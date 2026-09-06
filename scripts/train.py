"""Train an AdaptiveTTS model.

    python scripts/train.py --config configs/exp2_token.yaml
    python scripts/train.py --config configs/exp0_dense.yaml --override train.batch_size=8
    python scripts/train.py --config configs/exp2_token.yaml --resume runs/exp2_token/checkpoints/best.pt

Monitor with:  tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mint_tts.config import load_config
from mint_tts.training.trainer import Trainer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    if args.run_name:
        cfg.log.run_name = args.run_name
    Trainer(cfg, resume=args.resume).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
