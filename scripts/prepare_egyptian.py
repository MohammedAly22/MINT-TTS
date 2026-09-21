"""Download and lay out the 100-hour single-speaker Egyptian Arabic corpus.

    python scripts/prepare_egyptian.py --out data/egyptian
    python scripts/prepare_egyptian.py --out data/egyptian --limit 200   # dry run

Source: https://huggingface.co/datasets/ehabnegm/100-hour-Egyptian-dataset-single-speaker
(15,653 clips, 24 kHz mono, one speaker, ~71.8 h across the official splits;
transcripts are undiacritised and unpunctuated Egyptian Arabic.)

Why this does not use `load_dataset`
------------------------------------
The obvious implementation is `load_dataset(repo, split="train")` and then
`.select(range(limit))`. It is wrong here, in two ways that cost real time:

**`load_dataset` downloads everything before returning.** Subsetting afterwards
cannot help -- by the time `.select()` runs, all 12 GB is already on disk. A
`--limit` applied that way is not a dry run, it is a full download followed by
a subset.

**This repo stores audio as 15,653 individual .wav files**, not packed parquet
shards, so `load_dataset` issues one HTTP request per clip and the Hub rate
limits (HTTP 429) after a few thousand. Each retry then sleeps ~90 s.

So this script reads the small `metadata/*.jsonl` files first (three files,
a few MB), applies every filter and `--limit` to that, and only then downloads
the wavs that actually survive. A `--limit 200` run fetches 200 wavs instead of
15,653, and a filtered full run never downloads the clips it is going to drop.

Downloads run in a thread pool (`--workers`), which is what makes a full run
reasonable; `--workers 1` is the polite fallback if the Hub still rate limits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASET = "ehabnegm/100-hour-Egyptian-dataset-single-speaker"

# Metadata file -> the split name MINT-TTS uses.
METADATA = {"train": "train", "dev": "val", "test": "test"}

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Whitespace only. Orthographic normalisation belongs to the frontend.

    Doing it here as well would let the filelist and the model's view of the
    text drift apart, and the filelist is what a human inspects when a sample
    sounds wrong.
    """
    return _WS.sub(" ", str(text)).strip()


def load_metadata(repo: str, log=print) -> dict[str, list[dict]]:
    """Fetch the three small metadata files. No audio is downloaded here."""
    from huggingface_hub import hf_hub_download

    out: dict[str, list[dict]] = {}
    for remote, split in METADATA.items():
        path = hf_hub_download(repo, f"metadata/{remote}.jsonl", repo_type="dataset")
        rows = [json.loads(l) for l in
                Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        out[split] = rows
        log(f"  metadata/{remote}.jsonl -> {len(rows)} rows ({split})")
    return out


def select_rows(rows: list[dict], args) -> tuple[list[dict], dict[str, int]]:
    """Apply every filter BEFORE any audio is fetched."""
    keep, dropped = [], {"confidence": 0, "duration": 0, "promo": 0, "empty": 0}
    for r in rows:
        if not _clean(r.get("text", "")):
            dropped["empty"] += 1
            continue
        if args.drop_promo and bool(r.get("promo", False)):
            dropped["promo"] += 1
            continue
        conf = r.get("confidence")
        if conf is not None and float(conf) < args.min_confidence:
            dropped["confidence"] += 1
            continue
        dur = r.get("duration")
        if dur is not None and not (args.min_duration <= float(dur) <= args.max_duration):
            dropped["duration"] += 1
            continue
        keep.append(r)
    return keep, dropped


def fetch_one(repo: str, row: dict, wav_dir: Path, args) -> tuple[str, str] | None:
    """Download and convert one clip. Returns (relative path, text) or None."""
    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    uid = str(row.get("id") or Path(row["audio_path"]).stem)
    uid = uid.replace("/", "_").replace(" ", "_")
    dest = wav_dir / f"{uid}.wav"
    text = _clean(row.get("text", ""))

    if args.skip_existing and dest.exists():
        return f"wavs/{uid}.wav", text
    try:
        src = hf_hub_download(repo, row["audio_path"], repo_type="dataset")
        wav, sr = sf.read(src, dtype="float32", always_2d=False)
        if wav.ndim > 1:                       # to mono
            wav = wav.mean(1)
        if sr != args.sample_rate:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=args.sample_rate)
        sf.write(dest, np.asarray(wav, dtype="float32"), args.sample_rate,
                 subtype="PCM_16")
        return f"wavs/{uid}.wav", text
    except Exception as exc:                   # one bad clip must not kill the run
        return ("ERROR", f"{uid}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/egyptian", help="where the wavs land")
    ap.add_argument("--filelists", default="filelists")
    ap.add_argument("--prefix", default="egyptian")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop clips whose ASR confidence is below this (0-1). "
                         "The transcripts are already good -- median 0.975, "
                         "minimum 0.557 -- so 0.0 keeps all 71.8 h while 0.95 "
                         "costs 6.4 h and 0.97 costs 26.5 h. Raise it only if "
                         "alignment struggles.")
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--max-duration", type=float, default=20.0)
    ap.add_argument("--drop-promo", action="store_true",
                    help="drop clips flagged as sponsor segments")
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch this many clips PER SPLIT (a real dry run: "
                         "the audio for the rest is never downloaded)")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel downloads; lower it if the Hub rate limits")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not re-download wavs already on disk (resume)")
    args = ap.parse_args()

    try:
        # Fail now with a clear message rather than inside a worker thread,
        # where the traceback would be buried among concurrent downloads.
        import importlib

        for mod in ("soundfile", "huggingface_hub"):
            importlib.import_module(mod)
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}).\n"
              "  pip install huggingface_hub soundfile librosa", file=sys.stderr)
        return 1

    out = Path(args.out)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading metadata from {args.dataset} (no audio yet) ...")
    meta = load_metadata(args.dataset)

    # Filter first. Everything dropped here is never downloaded.
    planned: dict[str, list[dict]] = {}
    total_dropped = {"confidence": 0, "duration": 0, "promo": 0, "empty": 0}
    for split, rows in meta.items():
        keep, dropped = select_rows(rows, args)
        for k, v in dropped.items():
            total_dropped[k] += v
        if args.limit:
            keep = keep[: args.limit]
        planned[split] = keep

    n_planned = sum(len(v) for v in planned.values())
    n_total = sum(len(v) for v in meta.values())
    print(f"\nselected {n_planned} of {n_total} clips"
          + (f"  (--limit {args.limit} per split)" if args.limit else ""))
    for reason, n in total_dropped.items():
        if n:
            print(f"  filtered out ({reason}): {n}")
    hours = sum(float(r.get("duration", 0) or 0) for v in planned.values() for r in v) / 3600
    print(f"  ~{hours:.2f} hours to download\n")

    buckets: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    errors: list[str] = []
    done = 0
    for split, rows in planned.items():
        if not rows:
            continue
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(fetch_one, args.dataset, r, wav_dir, args): r
                       for r in rows}
            for fut in as_completed(futures):
                res = fut.result()
                done += 1
                if done % 200 == 0 or done == n_planned:
                    print(f"  {done}/{n_planned} clips", flush=True)
                if res is None:
                    continue
                if res[0] == "ERROR":
                    errors.append(res[1])
                else:
                    buckets[split].append(res)

    # Keep the order stable so reruns produce identical filelists.
    for split in buckets:
        buckets[split].sort()

    fl = Path(args.filelists)
    fl.mkdir(parents=True, exist_ok=True)
    everything = buckets["train"] + buckets["val"] + buckets["test"]
    for name, items in [*buckets.items(), ("all", everything)]:
        path = fl / f"{args.prefix}_{name}.txt"
        path.write_text("\n".join(f"{a}|{t}" for a, t in items) + "\n", encoding="utf-8")
        print(f"{path}  {len(items):>7d} utterances")

    if errors:
        err_path = out / "download_errors.txt"
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"\n{len(errors)} clips failed; see {err_path}")
        print("Re-run with --skip-existing to retry only the missing ones.")

    # How much of the corpus is structurally hard. If this is near zero the
    # homograph experiment cannot be run on this data, and it is far better to
    # learn that now than after a training run.
    try:
        from mint_tts.text.arabic import ARABIC_WORD_RE, ArabicNormalizer
        from mint_tts.text.homographs_ar import clitic_gender_ambiguity

        norm = ArabicNormalizer()
        n_clitic = n_words = n_utts = 0
        for _, text in everything:
            words = ARABIC_WORD_RE.findall(norm(text))
            n_words += len(words)
            hits = sum(1 for w in words if clitic_gender_ambiguity(w))
            n_clitic += hits
            n_utts += int(hits > 0)
        if n_words:
            print("\nstructural ambiguity (unwritten-vowel clitics):")
            print(f"  words      : {n_clitic} / {n_words} "
                  f"({100 * n_clitic / n_words:.2f}%)")
            print(f"  utterances : {n_utts} / {len(everything)} "
                  f"({100 * n_utts / max(len(everything), 1):.1f}%)")
            print("  (this is the orthographic prior only -- the real ambiguity")
            print("   measurement is scripts/mine_ambiguity.py)")
    except Exception as exc:  # pragma: no cover
        print(f"(coverage check skipped: {exc})")

    print("\nNext:")
    print(f"  1. data.root: {out.as_posix()}")
    print(f"     data.manifest: {fl.as_posix()}/{args.prefix}_all.txt")
    print("  2. python scripts/preprocess.py     --config configs/egyptian_homograph.yaml --workers 8")
    print("  3. python scripts/mine_ambiguity.py --config configs/egyptian_homograph.yaml")
    print("  4. python scripts/train.py          --config configs/egyptian_homograph.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
