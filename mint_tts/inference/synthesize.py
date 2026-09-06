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
        return cls(cfg, model, tp, dev, voc)

    # -- synthesis --------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, text: str, quality: float = 0.9, hardware: float = 1.0,
                 hard_routing: bool = True, return_audio: bool = True,
                 duration_scale: float = 1.0, encoder_max_steps: int | None = None,
                 decoder_max_steps: int | None = None, speaker: int = 0,
                 emotion: int = 0) -> SynthesisResult:
        import time

        enc = self.tp.encode(text)
        tokens = torch.tensor(enc.ids, dtype=torch.long, device=self.device).unsqueeze(0)
        lens = torch.tensor([len(enc.ids)], dtype=torch.long, device=self.device)
        budget = torch.tensor([[float(quality), float(hardware)]], device=self.device)
        spk = torch.tensor([speaker], device=self.device) if self.model.speaker_emb else None
        emo = torch.tensor([emotion], device=self.device) if self.model.emotion_emb else None

        t0 = time.perf_counter()
        out = self.model(tokens, lens, budget=budget, hard=hard_routing,
                         duration_scale=duration_scale, speakers=spk, emotions=emo,
                         encoder_max_steps=encoder_max_steps,
                         decoder_max_steps=decoder_max_steps)
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

    def batch(self, texts: list[str], **kwargs) -> list[SynthesisResult]:
        return [self(t, **kwargs) for t in texts]
