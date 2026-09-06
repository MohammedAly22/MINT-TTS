"""Latency / FLOPs / memory benchmark across devices and quality budgets.

    python scripts/benchmark.py --checkpoint runs/exp2_token/checkpoints/best.pt \
        --devices cpu cuda --threads 4

Also usable without a checkpoint (random weights) to compare architectures
before training anything:

    python scripts/benchmark.py --config configs/exp2_token.yaml --random-init
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mint_tts.benchmarks.latency import benchmark_model, benchmark_vocoder
from mint_tts.config import Config, load_config
from mint_tts.models.tts import build_model
from mint_tts.models.vocoder import Vocoder
from mint_tts.text.tokenizer import build_text_processor
from mint_tts.utils.common import format_table
from mint_tts.utils.flops import count_parameters, human

DEFAULT_SENTENCES = [
    "Hello, how are you doing?",
    "The record is broken by the record broker.",
    "I went to the store and I bought some milk and some bread and some eggs.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--devices", nargs="*", default=None)
    ap.add_argument("--budgets", nargs="*", type=float, default=None)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--texts", nargs="*", default=DEFAULT_SENTENCES)
    ap.add_argument("--soft-routing", action="store_true")
    ap.add_argument("--vocoder", action="store_true", help="also benchmark the vocoder")
    ap.add_argument("--out", default="benchmarks/results.csv")
    args = ap.parse_args()

    if args.checkpoint and not args.random_init:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = load_config(args.config, args.override) if args.config else Config(ckpt["config"])
    elif args.config:
        ckpt, cfg = None, load_config(args.config, args.override)
    else:
        ap.error("pass --checkpoint or --config")

    # The embedding is sized by the vocabulary, so a checkpoint can only be
    # loaded against the symbol table it was trained with.
    symbols = None
    for candidate in ([Path(args.checkpoint).parent.parent / "symbols.json"]
                      if args.checkpoint else []) + [
                          Path(cfg.data.preprocessed_dir) / "symbols.json"]:
        if candidate.exists():
            symbols = candidate
            break
    tp = build_text_processor(cfg, symbols=symbols)
    tp.freeze()
    model = build_model(cfg, tp.vocab_size)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    b = cfg.benchmark
    devices = args.devices or list(b.get("devices", ["cpu"]))
    budgets = args.budgets or list(b.get("budgets", [0.1, 0.5, 0.9, 1.0]))
    batch_sizes = args.batch_sizes or list(b.get("batch_sizes", [1]))
    threads = args.threads or b.get("threads", 4)
    iters = args.iters or b.get("iters", 20)

    print(f"Parameters: {human(count_parameters(model, False))}")
    rows = []
    for dev_str in devices:
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            print(f"[skip] {dev_str}: CUDA not available")
            continue
        device = torch.device(dev_str)
        for text in args.texts:
            enc = tp.encode(text)
            for bs in batch_sizes:
                tokens = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
                lens = torch.tensor([len(enc.ids)] * bs, dtype=torch.long)
                for q in budgets:
                    res = benchmark_model(
                        model, tokens, lens, device, quality=q, hard=not args.soft_routing,
                        warmup=b.get("warmup_iters", 3), iters=iters,
                        hop_length=cfg.audio.hop_length, sample_rate=cfg.audio.sample_rate,
                        threads=threads,
                    )
                    row = res.as_row()
                    row["text"] = text[:40]
                    rows.append(row)
                    print(f"{dev_str:6s} bs={bs} q={q:<4g} "
                          f"{res.latency_ms_mean:7.1f} ms  RTF={res.rtf:6.3f}  "
                          f"FLOPs={human(res.flops):>8s}  saving={res.flops_saving * 100:5.1f}%  "
                          f"| {text[:36]}")

    if args.vocoder:
        try:
            voc = Vocoder(cfg, device=devices[0])
            mel = torch.randn(1, cfg.audio.n_mels, 400) * 0.5 - 5.0
            print(json.dumps(benchmark_vocoder(voc, mel.to(devices[0]),
                                               torch.device(devices[0]),
                                               hop_length=cfg.audio.hop_length,
                                               sample_rate=cfg.audio.sample_rate), indent=2))
        except Exception as exc:
            print(f"[warn] vocoder benchmark skipped: {exc}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out}")
        summary = [
            {"device": r["device"], "q": r["budget_q"], "ms": f"{r['latency_ms_mean']:.1f}",
             "rtf": f"{r['rtf']:.3f}", "saving": f"{r['flops_saving'] * 100:.1f}%",
             "enc_c": f"{r['encoder_compute']:.2f}", "dec_c": f"{r['decoder_compute']:.2f}"}
            for r in rows
        ]
        print(format_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
