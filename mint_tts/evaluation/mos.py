"""Perceptual quality estimation.

`utmos` (a MOS-prediction network) is the preferred backend; it correlates
reasonably with human MOS and is cheap enough to run every validation epoch.
When it is unavailable we fall back to a transparent *MOS proxy* built from
objective distortions -- clearly labelled as a proxy, because publishing a
proxy as MOS would be dishonest. Human MOS is collected on a subset only,
via `scripts/export_mos_test.py`.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch


class MOSPredictor:
    def __init__(self, backend: str = "proxy", device: str = "cpu"):
        self.backend = backend
        self.device = torch.device(device)
        self.model = None
        if backend == "utmos":
            try:
                self.model = torch.hub.load(
                    "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
                ).to(self.device).eval()
                self.sample_rate = 16000
            except Exception as exc:  # pragma: no cover - network dependency
                warnings.warn(f"UTMOS unavailable ({exc}); using the objective MOS proxy.")
                self.backend = "proxy"

    @property
    def name(self) -> str:
        return "utmos" if self.backend == "utmos" and self.model is not None else "mos_proxy"

    @torch.inference_mode()
    def score_wav(self, wav: torch.Tensor, sample_rate: int) -> float:
        if self.backend != "utmos" or self.model is None:
            return float("nan")
        import torchaudio.functional as AF

        wav = wav.detach().float().cpu()
        if wav.dim() > 1:
            wav = wav.mean(0)
        if sample_rate != self.sample_rate:
            wav = AF.resample(wav, sample_rate, self.sample_rate)
        return float(self.model(wav.unsqueeze(0).to(self.device), self.sample_rate))

    @staticmethod
    def proxy_from_metrics(mcd: float, cer: float | None = None) -> float:
        """Map objective distortions to a 1-5 pseudo-MOS.

        Calibrated so that MCD ~= 3 dB with near-zero CER lands around 4.3 and
        MCD ~= 10 dB lands near 1.5. Use for *relative* comparisons only.
        """
        if not np.isfinite(mcd):
            return float("nan")
        score = 5.0 - 0.42 * float(mcd)
        if cer is not None and np.isfinite(cer):
            score -= 3.0 * float(cer)
        return float(np.clip(score, 1.0, 5.0))


def quality_score(metrics: dict, weights: dict | None = None) -> float:
    """Single scalar Q in [0, 1] used to locate the minimum-compute point.

    Combines mel distortion, intelligibility and (when available) predicted
    MOS. Every term is mapped to "higher is better" and clipped to [0, 1].
    """
    w = {"mcd": 0.4, "cer": 0.4, "mos": 0.2}
    if weights:
        w.update(weights)
    terms, total_w = 0.0, 0.0

    mcd = metrics.get("mcd", float("nan"))
    if np.isfinite(mcd):
        terms += w["mcd"] * float(np.clip(1.0 - (mcd - 2.0) / 8.0, 0.0, 1.0))
        total_w += w["mcd"]
    c = metrics.get("cer", float("nan"))
    if np.isfinite(c):
        terms += w["cer"] * float(np.clip(1.0 - c / 0.3, 0.0, 1.0))
        total_w += w["cer"]
    mos = metrics.get("mos", float("nan"))
    if np.isfinite(mos):
        terms += w["mos"] * float(np.clip((mos - 1.0) / 4.0, 0.0, 1.0))
        total_w += w["mos"]
    return terms / total_w if total_w > 0 else float("nan")
