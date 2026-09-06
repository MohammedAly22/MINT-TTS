"""ASR-based intelligibility scoring (WER / CER on synthesised audio).

Intelligibility is the metric that actually breaks first when a model is given
too little compute, which makes it the sharpest signal for locating C*.
"""

from __future__ import annotations

import re
import warnings

import torch

from .metrics import cer, wer

_PUNCT = re.compile(r"[^a-z' ]+")


def normalise_transcript(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


class ASRScorer:
    """Backends: `wav2vec2` (torchaudio bundle), `whisper` (faster-whisper), `none`."""

    def __init__(self, backend: str = "wav2vec2", device: str = "cpu",
                 model_name: str = "WAV2VEC2_ASR_BASE_960H", whisper_size: str = "tiny.en"):
        self.backend = backend
        self.device = torch.device(device)
        self.model = None
        self.sample_rate = 16000
        if backend == "none":
            return
        try:
            if backend == "wav2vec2":
                import torchaudio.pipelines as pipelines

                bundle = getattr(pipelines, model_name)
                self.model = bundle.get_model().to(self.device).eval()
                self.labels = bundle.get_labels()
                self.sample_rate = bundle.sample_rate
            elif backend == "whisper":
                from faster_whisper import WhisperModel

                self.model = WhisperModel(whisper_size, device=str(self.device), compute_type="int8")
            else:
                raise ValueError(f"Unknown ASR backend '{backend}'")
        except Exception as exc:  # pragma: no cover - optional/network dependency
            warnings.warn(f"ASR backend '{backend}' unavailable ({exc}); WER/CER disabled.")
            self.backend = "none"
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def transcribe(self, wav: torch.Tensor, sample_rate: int) -> str:
        if not self.available:
            return ""
        import torchaudio.functional as AF

        wav = wav.detach().float().cpu()
        if wav.dim() > 1:
            wav = wav.mean(0) if wav.shape[0] <= 2 else wav.squeeze()
        if sample_rate != self.sample_rate:
            wav = AF.resample(wav, sample_rate, self.sample_rate)
        if self.backend == "wav2vec2":
            emission, _ = self.model(wav.unsqueeze(0).to(self.device))
            ids = emission[0].argmax(-1)
            return self._ctc_decode(ids)
        segments, _ = self.model.transcribe(wav.numpy(), language="en", beam_size=1)
        return normalise_transcript(" ".join(s.text for s in segments))

    def _ctc_decode(self, ids: torch.Tensor) -> str:
        out, prev = [], -1
        for i in ids.tolist():
            if i != prev and self.labels[i] != "-":
                out.append(self.labels[i])
            prev = i
        return normalise_transcript("".join(out).replace("|", " "))

    def score(self, wav: torch.Tensor, sample_rate: int, reference: str) -> dict:
        if not self.available:
            return {"wer": float("nan"), "cer": float("nan"), "hypothesis": ""}
        hyp = self.transcribe(wav, sample_rate)
        ref = normalise_transcript(reference)
        return {"wer": wer(ref, hyp), "cer": cer(ref, hyp), "hypothesis": hyp, "reference": ref}
