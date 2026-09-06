"""Extract mel / pitch / energy features and cache tokenised text.

    python scripts/preprocess.py --config configs/exp2_token.yaml --workers 4

Expects filelists produced by `prepare_dataset.py` (or your own manifest in
any of the formats documented in docs/DATA_FORMAT.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mint_tts.config import load_config
from mint_tts.data.preprocess import run_preprocess
from mint_tts.utils.logging_utils import get_logger


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--filelist-prefix", default=None,
                    help="defaults to filelists/<dataset>_<split>.txt derived from data.manifest")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug: only N utterances per split")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    log = get_logger("preprocess")
    out_dir = Path(cfg.data.preprocessed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.filelist_prefix
    if prefix is None:
        manifest = Path(cfg.data.manifest)
        stem = manifest.stem
        for suffix in ("_all", "_train", "_val", "_test"):
            stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem
        prefix = str(manifest.parent / stem)

    results = {}
    # train first: it writes stats.json used by every split
    for split in sorted(args.splits, key=lambda s: 0 if s == "train" else 1):
        manifest = Path(f"{prefix}_{split}.txt")
        if not manifest.exists():
            log.warning(f"skipping '{split}': {manifest} not found")
            continue
        log.info(f"preprocessing {split} from {manifest}")
        res = run_preprocess(cfg, str(manifest), str(out_dir), split, args.workers, args.limit)
        results[split] = res
        log.info(f"  ok={res['n_ok']} errors={res['n_error']} -> {res['index']}")
        if res["stats"]:
            log.info(f"  {res['stats']['n_utterances']} utterances, "
                     f"{res['stats']['total_hours']:.2f} hours")

    (out_dir / "preprocess_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"Done. Features in {out_dir}")
    if any(r["n_error"] for r in results.values()):
        log.warning("Some files failed; see *_errors.json in the output directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
