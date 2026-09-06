"""Synthesise speech and dump the complexity diagnostics.

    python scripts/synthesize.py --checkpoint runs/exp2_token/checkpoints/best.pt \
        --text "The record is broken by the record broker." --quality 0.9

    # sweep the quality budget to see C*(x, q) in action
    python scripts/synthesize.py --checkpoint ... --text "..." --sweep-quality
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mint_tts.inference.synthesize import Synthesizer
from mint_tts.utils import plotting
from mint_tts.utils.plotting import save_figure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None, help="one utterance per line")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--quality", type=float, default=0.9)
    ap.add_argument("--hardware", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--soft-routing", action="store_true",
                    help="use the differentiable path (no gather-based speedup)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--sweep-quality", action="store_true",
                    help="synthesise at q in {0.1..1.0} and plot compute vs q")
    ap.add_argument("--plots", action="store_true", default=True)
    args = ap.parse_args()

    texts = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        texts += [l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    if not texts:
        texts = ["The record is broken by the record broker.",
                 "Hello, how are you doing?"]

    syn = Synthesizer.from_checkpoint(args.checkpoint, args.config, args.device,
                                      overrides=args.override, use_ema=not args.no_ema)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    qualities = [round(x, 2) for x in np.arange(0.1, 1.01, 0.1)] if args.sweep_quality else [args.quality]

    for i, text in enumerate(texts):
        per_q = []
        for q in qualities:
            res = syn(text, quality=q, hardware=args.hardware, hard_routing=not args.soft_routing)
            tag = f"{i:03d}_q{q:g}"
            if res.wav is not None:
                res.save(out_dir / f"{tag}.wav")
            per_q.append({
                "quality_budget": q,
                "encoder_compute": float(res.token_complexity.mean()),
                "decoder_compute": float(res.frame_complexity.mean()) if res.frame_complexity.size else 0.0,
                "flops": res.flops.total,
                "flops_saving": res.flops.saving,
                "latency_ms": res.latency_ms,
                "frames": int(res.mel.shape[-1]),
            })
            if args.plots and abs(q - qualities[-1]) < 1e-9:
                enc = res.encoded
                save_figure(plotting.plot_token_complexity(
                    enc.tokens, res.token_complexity, title=text[:90],
                    max_steps=syn.model.encoder.max_steps,
                    words=enc.words, word_ids=enc.word_ids),
                    out_dir / f"{tag}_token_complexity")
                save_figure(plotting.plot_word_complexity(enc.words, res.word_complexity),
                            out_dir / f"{tag}_word_complexity")
                save_figure(plotting.plot_mel(res.mel.numpy()), out_dir / f"{tag}_mel")
                if res.frame_complexity.size:
                    save_figure(plotting.plot_frame_complexity(res.frame_complexity),
                                out_dir / f"{tag}_frame_complexity")
            print(res.summary())
        if args.sweep_quality:
            save_figure(plotting.plot_compute_curve(
                [p["encoder_compute"] for p in per_q], [p["quality_budget"] for p in per_q],
                title="requested quality vs allocated compute"),
                out_dir / f"{i:03d}_quality_sweep")
        report.append({"text": text, "points": per_q})

    (out_dir / "synthesis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {len(texts)} utterance(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
