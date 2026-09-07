"""Fetch a pretrained HiFi-GAN generator and verify it loads.

The vocoder must be identical across every experiment, so it is downloaded
once and then treated as frozen infrastructure.

    # from a direct URL to a jik876/hifi-gan style checkpoint
    python scripts/download_vocoder.py --url https://.../g_02500000 --config-url https://.../config.json

    # from a Hugging Face repo containing the same files
    python scripts/download_vocoder.py --hf-repo <user>/<repo> --hf-file generator_v1

    # or verify a file you downloaded by hand
    python scripts/download_vocoder.py --local ~/Downloads/g_02500000

Verified source (V1 architecture, LJSpeech, 22.05 kHz / hop 256 -- matches this
repo's mel settings exactly):

    python scripts/download_vocoder.py --hf-repo speechbrain/tts-hifigan-ljspeech         --hf-file generator.ckpt

Also usable:
  * official release: https://github.com/jik876/hifi-gan (LJ_V1 / LJ_FT_T2_V1,
    on Google Drive -- download by hand, then pass --local)

Until a checkpoint is in place the repo falls back to Griffin-Lim, which is
intelligible but not publication quality: set `vocoder.name=hifigan` in the
config once this script reports success.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mint_tts.models.vocoder import (
    DEFAULT_HIFIGAN_V1,
    DEFAULT_HIFIGAN_V3,
    HiFiGANGenerator,
    normalise_generator_state,
)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n        -> {dest}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    return dest


def normalise_state_dict(state: dict) -> dict:
    return normalise_generator_state(state)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    ap.add_argument("--config-url", default=None)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--hf-file", default="generator.ckpt")
    ap.add_argument("--hf-config", default="config.json")
    ap.add_argument("--local", default=None)
    ap.add_argument("--variant", choices=["v1", "v3"], default="v1")
    ap.add_argument("--out-dir", default="pretrained/hifigan")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"generator_{args.variant}"
    cfg_path = out_dir / "config.json"

    if args.local:
        shutil.copy(args.local, ckpt_path)
    elif args.url:
        download(args.url, ckpt_path)
        if args.config_url:
            download(args.config_url, cfg_path)
    elif args.hf_repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("pip install huggingface_hub to use --hf-repo", file=sys.stderr)
            return 1
        shutil.copy(hf_hub_download(args.hf_repo, args.hf_file), ckpt_path)
        try:
            shutil.copy(hf_hub_download(args.hf_repo, args.hf_config), cfg_path)
        except Exception:
            print("[warn] no config.json in the repo; using the built-in default")
    else:
        print(__doc__)
        print("Nothing to do: pass --url, --hf-repo or --local.", file=sys.stderr)
        return 1

    h = dict(DEFAULT_HIFIGAN_V1 if args.variant == "v1" else DEFAULT_HIFIGAN_V3)
    if cfg_path.exists():
        h.update(json.loads(cfg_path.read_text(encoding="utf-8")))
    else:
        cfg_path.write_text(json.dumps(h, indent=2), encoding="utf-8")

    gen = HiFiGANGenerator(h)
    state = normalise_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    missing, unexpected = gen.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] {len(missing)} missing / {len(unexpected)} unexpected keys.")
        print("       The checkpoint may use a different HiFi-GAN variant; try --variant v3 "
              "or supply the matching --config-url.")
        if missing:
            print("       first missing:", missing[:5])
        return 1

    gen.remove_weight_norm()
    with torch.inference_mode():
        wav = gen(torch.randn(1, h.get("num_mels", 80), 40))
    print(f"OK: loaded {ckpt_path} and synthesised {wav.shape[-1]} samples from 40 mel frames.")
    print("Set in your config:\n  vocoder:\n    name: hifigan\n"
          f"    checkpoint: {ckpt_path.as_posix()}\n    config: {cfg_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
