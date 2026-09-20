"""Inference API.

    syn = Synthesizer.from_checkpoint("runs/exp2/checkpoints/best.pt")
    result = syn("The record is broken by the record broker.", quality=0.9)
    result.save("out.wav")

`quality` and `hardware` are the user-facing budget knobs: the model spends
the least computation it believes is needed to hit that quality on that class
of device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import Config, load_config
from ..data.audio import save_wav
from ..models.tts import AdaptiveTTS, build_model
from ..models.vocoder import Vocoder
from ..text.tokenizer import Encoded, TextProcessor, build_text_processor
from ..utils.common import resolve_device
from ..utils.flops import FlopReport


@dataclass
class SynthesisResult:
    text: str
    wav: torch.Tensor | None
    mel: torch.Tensor
    encoded: Encoded
    token_complexity: np.ndarray
    frame_complexity: np.ndarray
    word_complexity: np.ndarray
    durations: np.ndarray
    flops: FlopReport
    sample_rate: int
    latency_ms: float = 0.0
    extras: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        if self.wav is None:
            raise RuntimeError("No waveform: this result was produced with return_audio=False")
        save_wav(path, self.wav, self.sample_rate)
        return Path(path)

    def summary(self) -> str:
        return (
            f"{self.text}\n"
            f"  tokens={len(self.encoded.ids)}  frames={self.mel.shape[-1]}\n"
            f"  encoder compute={self.token_complexity.mean():.3f}  "
            f"decoder compute={self.frame_complexity.mean():.3f}\n"
            f"  FLOPs={self.flops.total:.3e}  saving vs dense={self.flops.saving * 100:.1f}%"
        )


class Synthesizer:
    def __init__(self, cfg: Config, model: AdaptiveTTS, text_processor: TextProcessor,
                 device: torch.device, vocoder: Vocoder | None = None):
        self.cfg = cfg
        self.model = model.to(device).eval()
        self.tp = text_processor
        self.device = device
        self.vocoder = vocoder
        # Set here as well as in `_init_semantic`, so a Synthesizer built
        # directly (tests, notebooks) is usable without that call.
        self.semantic_encoder = None

    # -- construction -----------------------------------------------------
    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, config: str | Path | None = None,
                        device: str = "auto", vocoder: bool = True,
                        overrides: list[str] | None = None, use_ema: bool = True) -> "Synthesizer":
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if config is not None:
            cfg = load_config(config, overrides)
        else:
            cfg = Config(ckpt["config"])
            for item in overrides or []:
                key, raw = item.split("=", 1)
                from ..config import parse_value

                cfg.set_path(key.strip(), parse_value(raw.strip()))
        dev = resolve_device(device)

        symbols_file = Path(checkpoint).parent.parent / "symbols.json"
        if not symbols_file.exists():
            symbols_file = Path(cfg.data.preprocessed_dir) / "symbols.json"
        tp = build_text_processor(cfg, symbols=symbols_file if symbols_file.exists() else None)
        tp.freeze()

        model = build_model(cfg, tp.vocab_size)
        state = ckpt["model"]
        if use_ema and "ema" in ckpt and ckpt["ema"].get("shadow"):
            shadow = ckpt["ema"]["shadow"]
            state = {k: (shadow[k].to(v.dtype) if k in shadow else v) for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        voc = None
        if vocoder:
            try:
                voc = Vocoder(cfg, device=dev)
            except Exception as exc:
                print(f"[warn] vocoder unavailable ({exc}); returning mel only")
        syn = cls(cfg, model, tp, dev, voc)
        syn._init_semantic()
        return syn

    def _init_semantic(self) -> None:
        """Load the frozen LM if this checkpoint was trained with semantics.

        Unlike training, inference has no cache to read from -- the sentence
        is new -- so the LM must run here. It is still frozen and still
        counted separately in the FLOP report.
        """
        self.semantic_encoder = None
        if not getattr(self.model, "use_semantic", False):
            return
        sem_cfg = self.cfg.model.get("semantic", {}) or {}
        try:
            from ..modules.semantic import SemanticEncoder

            self.semantic_encoder = SemanticEncoder(
                sem_cfg.get("model", "marbert"),
                layer=int(sem_cfg.get("layer", -1)),
                device=str(self.device),
            )
        except Exception as exc:
            # Silently synthesising without semantics would produce audio that
            # sounds plausible but resolves every homograph wrongly, which is
            # far worse than a loud failure.
            raise RuntimeError(
                f"This checkpoint was trained with semantic conditioning but the "
                f"language model could not be loaded ({exc}). Synthesising "
                f"without it would disable homograph disambiguation. "
                f"Install `transformers` and ensure "
                f"'{sem_cfg.get('model', 'marbert')}' is available."
            ) from exc

    def _semantic_for(self, enc):
        """(semantic, word_index) tensors for one encoded utterance."""
        if self.semantic_encoder is None:
            return None, None
        feats = self.semantic_encoder.encode(enc.words)
        sem = torch.from_numpy(feats.vectors).float().unsqueeze(0).to(self.device)
        if sem.shape[1] == 0:
            sem = torch.zeros(1, 1, self.semantic_encoder.hidden_size, device=self.device)
        widx = torch.tensor(enc.word_ids, dtype=torch.long, device=self.device)
        widx = widx.clamp(0, sem.shape[1] - 1).unsqueeze(0)
        return sem, widx

    # -- synthesis --------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, text: str, quality: float = 0.9, hardware: float = 1.0,
                 hard_routing: bool = True, return_audio: bool = True,
                 duration_scale: float = 1.0, encoder_max_steps: int | None = None,
                 decoder_max_steps: int | None = None, speaker: int = 0,
                 emotion: int = 0, reference_wav: str | Path | None = None,
                 reference_mel: torch.Tensor | None = None) -> SynthesisResult:
        import time

        enc = self.tp.encode(text)
        tokens = torch.tensor(enc.ids, dtype=torch.long, device=self.device).unsqueeze(0)
        lens = torch.tensor([len(enc.ids)], dtype=torch.long, device=self.device)
        budget = torch.tensor([[float(quality), float(hardware)]], device=self.device)
        spk = torch.tensor([speaker], device=self.device) if self.model.speaker_emb else None
        emo = torch.tensor([emotion], device=self.device) if self.model.emotion_emb else None

        sem, widx = self._semantic_for(enc)
        ref, ref_lens = self._reference(reference_wav, reference_mel)

        t0 = time.perf_counter()
        out = self.model(tokens, lens, budget=budget, hard=hard_routing,
                         duration_scale=duration_scale, speakers=spk, emotions=emo,
                         encoder_max_steps=encoder_max_steps,
                         decoder_max_steps=decoder_max_steps,
                         semantic=sem, word_index=widx,
                         reference_mel=ref, reference_lens=ref_lens)
        latency = (time.perf_counter() - t0) * 1000.0

        L = int(out.mel_mask[0].sum().item())
        mel = out.mel_post[0, :, :L]
        c_tok = out.encoder_router.complexity[0, : len(enc.ids)].float().cpu().numpy()
        c_frame = out.decoder_router.complexity[0, :L].float().cpu().numpy()
        from ..training.monitors import aggregate_by_word

        word_c = aggregate_by_word(c_tok, enc.word_ids, len(enc.words))

        wav = None
        if return_audio and self.vocoder is not None:
            wav = self.vocoder.to_wav(mel)
        return SynthesisResult(
            text=text, wav=wav, mel=mel.cpu(), encoded=enc,
            token_complexity=c_tok, frame_complexity=c_frame, word_complexity=word_c,
            durations=out.duration_rounded[0, : len(enc.ids)].cpu().numpy(),
            flops=self.model.flops(out), sample_rate=self.cfg.audio.sample_rate,
            latency_ms=latency,
        )

    def _reference(self, reference_wav, reference_mel):
        """Reference mel for voice cloning, from a wav path or a tensor."""
        if not getattr(self.model, "use_reference", False):
            return None, None
        if reference_mel is None and reference_wav is not None:
            from ..data.audio import AudioConfig, load_wav, mel_spectrogram

            ac = AudioConfig.from_cfg(self.cfg.audio)
            reference_mel = mel_spectrogram(load_wav(reference_wav, ac), ac)
        if reference_mel is None:
            return None, None          # the model falls back to its unknown token
        if reference_mel.dim() == 2:
            reference_mel = reference_mel.unsqueeze(0)
        reference_mel = reference_mel.to(self.device)
        lens = torch.tensor([reference_mel.shape[-1]], dtype=torch.long, device=self.device)
        return reference_mel, lens

    def batch(self, texts: list[str], **kwargs) -> list[SynthesisResult]:
        return [self(t, **kwargs) for t in texts]
