"""Full evaluation of one or more checkpoints on a held-out split.

    python scripts/evaluate.py --checkpoints runs/exp0_dense/checkpoints/best.pt \
        runs/exp2_token/checkpoints/best.pt --index data/preprocessed/ljspeech/test.jsonl

Prints (and writes) the experiment matrix the project is built around:

    model            params   Q      MCD    WER    FLOPs/utt  RTF(cpu)  compute
    dense-8                  ...
    adaptive-token           ...
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

from mint_tts.benchmarks.latency import benchmark_model
from mint_tts.evaluation.asr import ASRScorer
from mint_tts.evaluation.metrics import aggregate, mel_cepstral_distortion, mel_l1
from mint_tts.evaluation.mos import MOSPredictor, quality_score
from mint_tts.inference.synthesize import Synthesizer
from mint_tts.utils.common import format_table
from mint_tts.utils.flops import count_parameters, human


@torch.inference_mode()
def evaluate_checkpoint(path: str, index: str, args) -> dict:
    syn = Synthesizer.from_checkpoint(path, args.config, args.device, overrides=args.override)
    cfg, model, device = syn.cfg, syn.model, syn.device
    asr = ASRScorer(args.asr or cfg.eval.get("asr_backend", "none"), device="cpu")
    mos = MOSPredictor(args.mos or cfg.eval.get("mos_backend", "proxy"), device="cpu")

    rows = [json.loads(l) for l in Path(index).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    mcds, l1s, wers, cers, moss, comps, flops_list = [], [], [], [], [], [], []
    for row in tqdm(rows, desc=Path(path).parent.parent.name):
        mel_target = torch.from_numpy(np.load(row["mel"])).float()
        tokens = torch.tensor(row["tokens"], dtype=torch.long, device=device).unsqueeze(0)
        lens = torch.tensor([len(row["tokens"])], dtype=torch.long, device=device)
        budget = torch.tensor([[args.quality, 1.0]], device=device)
        out = model(tokens, lens, budget=budget, hard=True)
        L = int(out.mel_mask[0].sum().item())
        pred = out.mel_post[0, :, :L].float().cpu()

        mcds.append(mel_cepstral_distortion(pred, mel_target))
        l1s.append(mel_l1(pred, mel_target))
        rep = model.flops(out)
        flops_list.append(rep.total)
        comps.append(float(out.encoder_router.per_utterance_depth()[0] / model.encoder.max_steps))
        if syn.vocoder is not None and asr.available:
            wav = syn.vocoder.to_wav(pred.to(device))
            sc = asr.score(wav, cfg.audio.sample_rate, row.get("clean_text", row["text"]))
            wers.append(sc["wer"])
            cers.append(sc["cer"])
            if mos.backend == "utmos":
                moss.append(mos.score_wav(wav, cfg.audio.sample_rate))

    metrics = {
        "checkpoint": path,
        "run": Path(path).parent.parent.name,
        "params": count_parameters(model, False),
        "routing": f"{model.encoder.routing}/{model.decoder.routing}",
        "max_steps": f"{model.encoder.max_steps}/{model.decoder.max_steps}",
        "shared": f"{model.encoder.share_weights}/{model.decoder.share_weights}",
        "mcd": aggregate(mcds)["mean"],
        "mel_l1": aggregate(l1s)["mean"],
        "compute_norm": float(np.mean(comps)),
        "flops_per_utt": float(np.mean(flops_list)),
        "n": len(rows),
    }
    if wers:
        metrics["wer"] = aggregate(wers)["mean"]
        metrics["cer"] = aggregate(cers)["mean"]
    if moss:
        metrics["mos"] = aggregate(moss)["mean"]
    else:
        metrics["mos_proxy"] = MOSPredictor.proxy_from_metrics(metrics["mcd"], metrics.get("cer"))
    metrics["Q"] = quality_score({
        "mcd": metrics["mcd"], "cer": metrics.get("cer", float("nan")),
        "mos": metrics.get("mos", metrics.get("mos_proxy", float("nan"))),
    })

    if args.benchmark:
        enc = syn.tp.encode("The record is broken by the record broker.")
        tk = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0)
        tl = torch.tensor([len(enc.ids)], dtype=torch.long)
        for dev in args.benchmark_devices:
            if dev.startswith("cuda") and not torch.cuda.is_available():
                continue
            res = benchmark_model(model, tk, tl, torch.device(dev), quality=args.quality,
                                  iters=10, hop_length=cfg.audio.hop_length,
                                  sample_rate=cfg.audio.sample_rate)
            metrics[f"rtf_{dev}"] = res.rtf
            metrics[f"ms_{dev}"] = res.latency_ms_mean
        model.to(device)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--quality", type=float, default=0.9)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--asr", default=None, help="none | wav2vec2 | whisper")
    ap.add_argument("--mos", default=None, help="proxy | utmos")
    ap.add_argument("--benchmark", action="store_true", help="also measure latency")
    ap.add_argument("--benchmark-devices", nargs="*", default=["cpu"])
    ap.add_argument("--out", default="outputs/evaluation.json")
    args = ap.parse_args()

    results = [evaluate_checkpoint(c, args.index, args) for c in args.checkpoints]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    table = []
    for r in results:
        row = {
            "run": r["run"], "routing": r["routing"], "params": human(r["params"]),
            "Q": f"{r['Q']:.3f}", "MCD": f"{r['mcd']:.2f}",
            "compute": f"{r['compute_norm']:.2f}", "FLOPs/utt": human(r["flops_per_utt"]),
        }
        if "wer" in r:
            row["WER"] = f"{r['wer']:.3f}"
        for k in r:
            if k.startswith("rtf_"):
                row[k] = f"{r[k]:.3f}"
        table.append(row)
    print("\n" + format_table(table))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
