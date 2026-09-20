"""Download and lay out the 100-hour single-speaker Egyptian Arabic corpus.

    python scripts/prepare_egyptian.py --out data/egyptian

Source: https://huggingface.co/datasets/ehabnegm/100-hour-Egyptian-dataset-single-speaker
(15,653 clips, 24 kHz mono, one speaker, ~71.8 h usable across the official
splits; transcripts are undiacritised and unpunctuated Egyptian Arabic.)

What this writes
----------------
    data/egyptian/wavs/<id>.wav          24 kHz mono PCM
    filelists/egyptian_{train,val,test,all}.txt     audio|text

The corpus carries its own `split` column naming disjoint *source videos* per
partition, and that is what is used here rather than a random split. It
matters: clips from one video share a recording session, a topic and a
speaking style, so splitting randomly would put near-duplicates of training
clips into validation and make every validation number optimistic.

Filtering
---------
`--min-confidence` drops clips whose ASR confidence is below a threshold. The
transcripts are machine-generated, so a low-confidence clip is one where the
text may not match the audio -- and a TTS model trained on mismatched pairs
learns to ignore its input. The default of 0.0 keeps everything; 0.8 is a
reasonable starting point if alignment struggles.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASET = "ehabnegm/100-hour-Egyptian-dataset-single-speaker"

# The split column's values, mapped to the names MINT-TTS uses.
SPLIT_MAP = {"train": "train", "dev": "val", "validation": "val", "test": "test"}


def _clean(text: str) -> str:
    """Whitespace only. Orthographic normalisation belongs to the frontend.

    Doing it here as well would mean the filelist and the model's view of the
    text could drift apart, and the filelist is what a human inspects when a
    sample sounds wrong.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/egyptian", help="where the wavs land")
    ap.add_argument("--filelists", default="filelists")
    ap.add_argument("--prefix", default="egyptian")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="drop clips whose ASR confidence is below this (0-1)")
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--max-duration", type=float, default=20.0)
    ap.add_argument("--drop-promo", action="store_true",
                    help="drop clips flagged as sponsor segments")
    ap.add_argument("--limit", type=int, default=None, help="for a smoke test")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not rewrite wavs that are already on disk")
    args = ap.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        from datasets import load_dataset
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}).\n"
              "  pip install datasets soundfile librosa", file=sys.stderr)
        return 1

    out = Path(args.out)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"  {len(ds)} rows, columns: {ds.column_names}")

    buckets: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    kept = 0
    skipped = {"confidence": 0, "duration": 0, "promo": 0, "empty": 0, "split": 0}

    for i, row in enumerate(ds):
        if i % 1000 == 0:
            print(f"  {i}/{len(ds)} ...", flush=True)

        split = SPLIT_MAP.get(str(row.get("split", "train")).lower())
        if split is None:
            skipped["split"] += 1
            continue
        text = _clean(row.get("text", ""))
        if not text:
            skipped["empty"] += 1
            continue
        if args.drop_promo and bool(row.get("promo", False)):
            skipped["promo"] += 1
            continue
        conf = row.get("confidence", None)
        if conf is not None and float(conf) < args.min_confidence:
            skipped["confidence"] += 1
            continue

        audio = row["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        dur = len(wav) / max(sr, 1)
        if not (args.min_duration <= dur <= args.max_duration):
            skipped["duration"] += 1
            continue

        if sr != args.sample_rate:
            try:
                import librosa

                wav = librosa.resample(wav, orig_sr=sr, target_sr=args.sample_rate)
            except ImportError:
                print("ERROR: clips are not at the requested rate and librosa is "
                      "missing. pip install librosa", file=sys.stderr)
                return 1

        uid = str(row.get("id", f"eg{i:06d}")).replace("/", "_").replace(" ", "_")
        path = wav_dir / f"{uid}.wav"
        if not (args.skip_existing and path.exists()):
            sf.write(path, wav, args.sample_rate, subtype="PCM_16")
        buckets[split].append((f"wavs/{uid}.wav", text))
        kept += 1

    fl = Path(args.filelists)
    fl.mkdir(parents=True, exist_ok=True)
    everything = buckets["train"] + buckets["val"] + buckets["test"]
    for name, items in [*buckets.items(), ("all", everything)]:
        path = fl / f"{args.prefix}_{name}.txt"
        path.write_text(
            "\n".join(f"{a}|{t}" for a, t in items) + "\n", encoding="utf-8")
        print(f"{path}  {len(items):>7d} utterances")

    print(f"\nkept {kept} / {len(ds)}")
    for reason, n in skipped.items():
        if n:
            print(f"  dropped ({reason}): {n}")

    # A quick look at how much of the corpus is actually hard. If this is
    # near zero the homograph experiment cannot be run on this data at all,
    # and it is far better to learn that now than after a training run.
    try:
        from mint_tts.text.arabic import ARABIC_WORD_RE, ArabicNormalizer
        from mint_tts.text.homographs_ar import difficulty_profile

        norm = ArabicNormalizer()
        n_hard_words = n_words = n_hard_utts = 0
        for _, text in everything:
            words = ARABIC_WORD_RE.findall(norm(text))
            d = difficulty_profile(words)
            n_words += len(words)
            hard = sum(1 for x in d if x > 0.5)
            n_hard_words += hard
            n_hard_utts += int(hard > 0)
        if n_words:
            print("\nhomograph coverage:")
            print(f"  ambiguous words : {n_hard_words} / {n_words} "
                  f"({100 * n_hard_words / n_words:.2f}%)")
            print(f"  utterances with >=1 : {n_hard_utts} / {len(everything)} "
                  f"({100 * n_hard_utts / max(len(everything), 1):.1f}%)")
    except Exception as exc:  # pragma: no cover
        print(f"(coverage check skipped: {exc})")

    print("\nNext:")
    print(f"  1. data.root: {out.as_posix()}")
    print(f"     data.manifest: {fl.as_posix()}/{args.prefix}_all.txt")
    print("  2. python scripts/preprocess.py --config configs/egyptian_homograph.yaml --workers 8")
    print("  3. python scripts/train.py --config configs/egyptian_homograph.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
