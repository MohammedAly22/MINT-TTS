"""Per-utterance compute-quality curves and the minimum-compute label C*.

This is the "generated supervision" step. For every utterance we synthesise it
repeatedly under increasing compute budgets and measure quality, producing

    Q_x(C)  -- quality as a function of compute for utterance x

and then

    C*(x, q) = min { C : Q_x(C) >= q * max_C Q_x(C) }

No human annotation is involved: the labels come from the model's own
behaviour under a controlled ablation. They serve three purposes:

1. evidence -- easy and hard utterances should have visibly different curves;
2. evaluation -- reporting mean C* is more informative than mean FLOPs;
3. supervision -- `loss.compute.c_star_weight` can distil these labels back
   into the router (see docs/EXPERIMENTS.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .asr import ASRScorer
from .metrics import mel_cepstral_distortion, mel_l1
from .mos import MOSPredictor, quality_score


def _steps_for(fraction: float, max_steps: int) -> int:
    return max(1, min(max_steps, int(math.ceil(fraction * max_steps))))


@torch.inference_mode()
def utterance_curve(
    model,
    mel_target: torch.Tensor,
    tokens: torch.Tensor,
    token_lens: torch.Tensor,
    text: str,
    points: list[float],
    device: torch.device,
    vocoder=None,
    asr: ASRScorer | None = None,
    mos: MOSPredictor | None = None,
    sweep: str = "steps",
    sample_rate: int = 22050,
    hard: bool = True,
) -> dict:
    """Returns {'compute': [...], 'quality': [...], 'metrics': [...]} for one utterance."""
    curve = {"compute": [], "encoder_compute": [], "decoder_compute": [],
             "quality": [], "metrics": []}
    for frac in points:
        if sweep == "steps":
            kwargs = dict(
                encoder_max_steps=_steps_for(frac, model.encoder.max_steps),
                decoder_max_steps=_steps_for(frac, model.decoder.max_steps),
            )
            budget = torch.tensor([[1.0, 1.0]], device=device)
        else:  # sweep == "budget": ask the router itself for cheaper output
            kwargs = {}
            budget = torch.tensor([[float(frac), 1.0]], device=device)

        out = model(tokens, token_lens, budget=budget, hard=hard, **kwargs)
        L = int(out.mel_mask[0].sum().item())
        pred = out.mel_post[0, :, :L].float().cpu()
        metrics = {
            "mel_l1": mel_l1(pred, mel_target),
            "mcd": mel_cepstral_distortion(pred, mel_target),
            "encoder_compute": float(out.encoder_router.per_utterance_depth()[0]
                                     / model.encoder.max_steps),
            "decoder_compute": float(out.decoder_router.per_utterance_depth()[0]
                                     / model.decoder.max_steps),
            "flops": float(model.flops(out).total),
            "flops_saving": float(model.flops(out).saving),
        }
        if vocoder is not None and asr is not None and asr.available:
            wav = vocoder.to_wav(pred.to(device))
            sc = asr.score(wav, sample_rate, text)
            metrics["wer"], metrics["cer"] = sc["wer"], sc["cer"]
            metrics["hypothesis"] = sc["hypothesis"]
            if mos is not None and mos.backend == "utmos":
                metrics["mos"] = mos.score_wav(wav, sample_rate)
        if "mos" not in metrics:
            metrics["mos"] = MOSPredictor.proxy_from_metrics(metrics["mcd"], metrics.get("cer"))

        # Fraction of the dense-equivalent FLOPs actually spent. A flat 50/50
        # average of the two stacks hides the one under study: with a fixed
        # decoder it pins C* near 0.5 no matter what the encoder does, so the
        # per-stack numbers are kept alongside it.
        curve["compute"].append(float(1.0 - metrics["flops_saving"]))
        curve["encoder_compute"].append(metrics["encoder_compute"])
        curve["decoder_compute"].append(metrics["decoder_compute"])
        curve["quality"].append(quality_score(metrics))
        curve["metrics"].append(metrics)
    return curve


def minimum_compute(curve: dict, ratio: float = 0.98) -> dict:
    """C* = cheapest measured compute reaching `ratio` of the best quality."""
    q = np.array(curve["quality"], dtype=np.float64)
    c = np.array(curve["compute"], dtype=np.float64)
    if not np.isfinite(q).any():
        return {"c_star": float("nan"), "q_max": float("nan"), "threshold": float("nan")}
    q_max = float(np.nanmax(q))
    threshold = ratio * q_max
    ok = np.where(q >= threshold)[0]
    idx = int(ok[0]) if ok.size else int(np.nanargmax(q))
    out = {
        "c_star": float(c[idx]),
        "c_star_index": idx,
        "q_max": q_max,
        "threshold": float(threshold),
        "q_at_c_star": float(q[idx]),
    }
    for stack in ("encoder", "decoder"):
        vals = curve.get(f"{stack}_compute")
        if vals and idx < len(vals):
            out[f"c_star_{stack}"] = float(vals[idx])
    return out


def summarise(records: list[dict]) -> dict:
    cstars = np.array([r["c_star"] for r in records if np.isfinite(r.get("c_star", np.nan))])
    if cstars.size == 0:
        return {"n": 0}
    extra = {}
    for stack in ("encoder", "decoder"):
        v = np.array([r[f"c_star_{stack}"] for r in records if f"c_star_{stack}" in r])
        if v.size:
            extra[f"c_star_{stack}_mean"] = float(v.mean())
            extra[f"c_star_{stack}_std"] = float(v.std())
            extra[f"c_star_{stack}_unique"] = int(np.unique(np.round(v, 4)).size)
    # If nearly every utterance shares one C*, there is nothing to allocate --
    # the premise fails before any router gets a chance.
    extra["c_star_unique_values"] = int(np.unique(np.round(cstars, 4)).size)
    extra["c_star_share_at_mode"] = float(
        (np.round(cstars, 4) == np.round(np.median(cstars), 4)).mean())
    return {**extra,
        "n": int(cstars.size),
        "c_star_mean": float(cstars.mean()),
        "c_star_std": float(cstars.std()),
        "c_star_p10": float(np.percentile(cstars, 10)),
        "c_star_median": float(np.median(cstars)),
        "c_star_p90": float(np.percentile(cstars, 90)),
        "fraction_below_half": float((cstars < 0.5).mean()),
    }


def save_curves(path: str | Path, records: list[dict], summary: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "utterances": records}, indent=2),
                    encoding="utf-8")
    return path
