"""Build per-utterance compute-quality curves and minimum-compute labels C*.

    python scripts/compute_curve.py --checkpoint runs/exp2_token/checkpoints/best.pt \
        --index data/preprocessed/ljspeech/val.jsonl --limit 200

Writes `compute_curves.json` (curves + C* per utterance + dataset summary) and,
with --write-labels, injects `c_star` back into a copy of the index so that
`loss.compute.c_star_weight` can distil the labels into the router.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from mint_tts.evaluation.asr import ASRScorer
from mint_tts.evaluation.compute_curve import (
    minimum_compute,
    save_curves,
    summarise,
    utterance_curve,
)
from mint_tts.evaluation.mos import MOSPredictor
from mint_tts.inference.synthesize import Synthesizer
from mint_tts.utils import plotting
from mint_tts.utils.plotting import save_figure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--index", required=True, help="preprocessed .jsonl index")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--sweep", choices=["steps", "budget"], default="steps")
    ap.add_argument("--points", nargs="*", type=float, default=None)
    ap.add_argument("--ratio", type=float, default=None,
                    help="quality retention ratio defining C* (default: eval.quality_threshold_ratio)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/compute_curves.json")
    ap.add_argument("--write-labels", default=None,
                    help="path of a new .jsonl index with c_star added")
    ap.add_argument("--plot-examples", type=int, default=6)
    args = ap.parse_args()

    syn = Synthesizer.from_checkpoint(args.checkpoint, args.config, args.device,
                                      overrides=args.override)
    cfg, model, device = syn.cfg, syn.model, syn.device
    points = args.points or list(cfg.eval.get("compute_curve_points",
                                              [0.25, 0.5, 0.75, 1.0]))
    ratio = args.ratio if args.ratio is not None else float(
        cfg.eval.get("quality_threshold_ratio", 0.98))

    asr = ASRScorer(cfg.eval.get("asr_backend", "none"), device="cpu")
    mos = MOSPredictor(cfg.eval.get("mos_backend", "proxy"), device="cpu")

    rows = [json.loads(l) for l in Path(args.index).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i, row in enumerate(tqdm(rows, desc="compute curves")):
        mel_target = torch.from_numpy(np.load(row["mel"])).float()
        tokens = torch.tensor(row["tokens"], dtype=torch.long, device=device).unsqueeze(0)
        lens = torch.tensor([len(row["tokens"])], dtype=torch.long, device=device)
        text = row.get("clean_text", row["text"])
        curve = utterance_curve(
            model, mel_target, tokens, lens, text, points, device,
            vocoder=syn.vocoder, asr=asr, mos=mos, sweep=args.sweep,
            sample_rate=cfg.audio.sample_rate,
        )
        info = minimum_compute(curve, ratio)
        rec = {"uid": row["uid"], "text": text, "n_tokens": len(row["tokens"]),
               "n_frames": row["n_frames"], **info, "curve": curve}
        records.append(rec)
        if i < args.plot_examples:
            save_figure(plotting.plot_compute_curve(
                curve["compute"], curve["quality"], info["threshold"], info["c_star"],
                title=text[:70]), out_dir / f"curve_{i:03d}_{row['uid']}")

    summary = summarise(records)
    save_curves(args.out, records, summary)
    print(json.dumps(summary, indent=2))

    cs = [r["c_star"] for r in records if np.isfinite(r["c_star"])]
    lens_ = [r["n_tokens"] for r in records if np.isfinite(r["c_star"])]
    if cs:
        save_figure(plotting.plot_scatter(lens_, cs, "tokens", "C*",
                                          "minimum compute vs utterance length"),
                    out_dir / "c_star_vs_length")
        from mint_tts.training.monitors import _safe_corr

        corr = _safe_corr(lens_, cs)
        print(f"corr(length, C*) = {corr:.3f}   "
              f"(near 1.0 would mean compute is just a proxy for length)")

    if args.write_labels:
        by_uid = {r["uid"]: r["c_star"] for r in records}
        out_rows = []
        for row in rows:
            row = dict(row)
            if row["uid"] in by_uid and np.isfinite(by_uid[row["uid"]]):
                row["c_star"] = by_uid[row["uid"]]
            out_rows.append(row)
        Path(args.write_labels).write_text(
            "\n".join(json.dumps(r) for r in out_rows) + "\n", encoding="utf-8")
        print(f"Wrote labelled index to {args.write_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
