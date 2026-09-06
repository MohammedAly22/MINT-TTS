"""Objective speech metrics used both for validation and for building the
per-utterance compute curves."""

from __future__ import annotations

import numpy as np
import torch


def mel_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Length-tolerant L1 between two (n_mels, T) log-mels."""
    T = min(pred.shape[-1], target.shape[-1])
    return float((pred[..., :T] - target[..., :T]).abs().mean())


def mel_cepstral_distortion(pred: torch.Tensor, target: torch.Tensor, n_mfcc: int = 13) -> float:
    """MCD-style distortion from log-mels via a DCT (no external deps).

    Standard formulation: 10/ln(10) * sqrt(2 * sum_i (c_i - c_i')^2), applied
    to cepstra derived from the 80-bin log-mel rather than from a WORLD/SPTK
    mel-cepstral analysis, and without DTW.

    That makes the numbers **not comparable to published MCD**, which usually
    sits at 4-8 dB. On this scale, measured on LJSpeech-like features:

        identical mels          0
        mild noise              ~2
        two DIFFERENT utterances ~52   <- the "no information" level
        constant mean spectrum  ~220

    Use `chance_mcd` to measure that reference on your own data instead of
    trusting the number in isolation: a model scoring near it has learned
    nothing utterance-specific, however good the mel loss looks.
    """
    T = min(pred.shape[-1], target.shape[-1])
    if T < 2:
        return float("nan")
    a, b = pred[..., :T], target[..., :T]
    ca = _dct(a.transpose(0, 1))[:, 1:n_mfcc]
    cb = _dct(b.transpose(0, 1))[:, 1:n_mfcc]
    diff = ((ca - cb) ** 2).sum(-1)
    return float((10.0 / np.log(10.0) * torch.sqrt(2.0 * diff)).mean())


def _dct(x: torch.Tensor) -> torch.Tensor:
    """Type-II orthonormal DCT along the last dim."""
    N = x.shape[-1]
    n = torch.arange(N, device=x.device, dtype=x.dtype)
    k = n.unsqueeze(1)
    basis = torch.cos(np.pi / N * (n.unsqueeze(0) + 0.5) * k)
    scale = torch.full((N,), np.sqrt(2.0 / N), device=x.device, dtype=x.dtype)
    scale[0] = np.sqrt(1.0 / N)
    return torch.matmul(x, basis.T) * scale


def chance_mcd(mels: list, n_pairs: int = 40, seed: int = 0) -> float:
    """MCD between *unrelated* utterances -- the no-information reference.

    Anchoring quality to this makes the score meaningful across datasets and
    feature settings, instead of relying on a constant calibrated elsewhere.
    """
    import random

    if len(mels) < 2:
        return float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_pairs):
        i, j = rng.sample(range(len(mels)), 2)
        v = mel_cepstral_distortion(mels[i], mels[j])
        if np.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def f0_rmse(pred_f0: torch.Tensor, target_f0: torch.Tensor) -> float:
    T = min(pred_f0.numel(), target_f0.numel())
    a, b = pred_f0[:T], target_f0[:T]
    voiced = (a > 0) & (b > 0)
    if voiced.sum() == 0:
        return float("nan")
    return float(torch.sqrt(((a[voiced] - b[voiced]) ** 2).mean()))


def duration_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.float() - target.float()).abs().mean())


def spectral_convergence(pred: torch.Tensor, target: torch.Tensor) -> float:
    T = min(pred.shape[-1], target.shape[-1])
    a, b = pred[..., :T], target[..., :T]
    return float(torch.norm(a - b) / torch.norm(b).clamp_min(1e-8))


def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return float("nan")
    return levenshtein(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
    if not ref:
        return float("nan")
    return levenshtein(ref, hyp) / len(ref)


def aggregate(values: list[float]) -> dict:
    arr = np.array([v for v in values if v == v and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "median": float(np.median(arr)), "n": int(arr.size),
    }
