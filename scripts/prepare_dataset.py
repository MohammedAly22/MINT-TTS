"""Turn a downloaded corpus into MINT-TTS filelists.

    python scripts/prepare_dataset.py --dataset ljspeech    --root data/LJSpeech-1.1
    python scripts/prepare_dataset.py --dataset vctk        --root data/VCTK-Corpus-0.92
    python scripts/prepare_dataset.py --dataset libritts    --root data/LibriTTS
    python scripts/prepare_dataset.py --dataset librispeech --root data/LibriSpeech

Writes `filelists/<prefix>_{train,val,test,all}.txt`, plus a short report of
speaker counts and total duration.

Downloads
---------
LJSpeech    https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2   (2.6 GB, 24 h, 1 speaker)
VCTK        https://datashare.ed.ac.uk/handle/10283/3443                (11 GB, 44 h, 110 speakers)
LibriTTS    https://openslr.org/60/                                     (per-subset, 585 h total)
LibriSpeech https://openslr.org/12/                                     (ASR corpus; see the note below)

A note on LibriSpeech: its transcripts are upper-case and stripped of all
punctuation, so a model trained on it cannot learn phrasing or intonation
cues that punctuation carries. Prefer **LibriTTS**, which is the same audio
re-released for synthesis with true casing and punctuation. LibriSpeech
support is here because it is what people usually have on disk.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# --------------------------------------------------------------------------
# per-corpus collectors: each returns rows of (audio, speaker, text)
# --------------------------------------------------------------------------
def collect_ljspeech(root: Path, normalised_text: bool = True) -> list[tuple[str, str, str]]:
    meta = root / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found (expected LJSpeech-1.1 layout)")
    rows = []
    with meta.open("r", encoding="utf-8", newline="") as fh:
        for parts in csv.reader(fh, delimiter="|", quoting=csv.QUOTE_NONE):
            if len(parts) < 2:
                continue
            uid = parts[0]
            text = parts[2] if (normalised_text and len(parts) > 2 and parts[2]) else parts[1]
            wav = root / "wavs" / f"{uid}.wav"
            if wav.exists():
                rows.append((f"wavs/{uid}.wav", "LJ", text.strip()))
    return rows


def collect_vctk(root: Path, mic: str = "mic1") -> list[tuple[str, str, str]]:
    """Handles both VCTK 0.92 (flac, _mic1/_mic2) and the older wav48 layout."""
    txt_dir = next((d for d in (root / "txt", root / "transcripts") if d.is_dir()), None)
    if txt_dir is None:
        raise FileNotFoundError(f"No txt/ directory under {root}")
    audio_dirs = [d for d in (root / "wav48_silence_trimmed", root / "wav48") if d.is_dir()]
    if not audio_dirs:
        raise FileNotFoundError(f"No wav48_silence_trimmed/ or wav48/ under {root}")
    audio_dir = audio_dirs[0]

    index: dict[str, Path] = {}
    for path in audio_dir.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        stem = path.stem
        if "_mic" in stem:
            if not stem.endswith(f"_{mic}"):
                continue
            stem = stem.rsplit("_mic", 1)[0]
        index.setdefault(stem, path)

    rows = []
    for txt in sorted(txt_dir.rglob("*.txt")):
        audio = index.get(txt.stem)
        if audio is None:
            continue
        text = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            rows.append((_rel(audio, root), txt.stem.split("_")[0], text))
    return rows


def collect_libritts(root: Path, subsets: list[str] | None = None,
                     original_text: bool = False) -> list[tuple[str, str, str]]:
    suffix = ".original.txt" if original_text else ".normalized.txt"
    search_dirs = [root / s for s in subsets] if subsets else [root]
    rows = []
    for base in search_dirs:
        if not base.is_dir():
            continue
        for audio in sorted(base.rglob("*.wav")):
            txt = audio.parent / (audio.stem + suffix)
            if not txt.exists():
                continue
            text = txt.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                rows.append((_rel(audio, root), audio.stem.split("_")[0], text))
    return rows


def collect_librispeech(root: Path, subsets: list[str] | None = None,
                        titlecase: bool = True) -> list[tuple[str, str, str]]:
    search_dirs = [root / s for s in subsets] if subsets else [root]
    rows = []
    for base in search_dirs:
        if not base.is_dir():
            continue
        for trans in sorted(base.rglob("*.trans.txt")):
            for line in trans.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                uid, _, text = line.partition(" ")
                audio = trans.parent / f"{uid}.flac"
                if not audio.exists():
                    audio = trans.parent / f"{uid}.wav"
                if not audio.exists():
                    continue
                text = text.strip()
                if titlecase:
                    # LibriSpeech is ALL CAPS; lowering it stops the acronym
                    # rule from spelling every word out letter by letter.
                    text = text.lower().capitalize()
                rows.append((_rel(audio, root), uid.split("-")[0], text))
    return rows


COLLECTORS = {
    "ljspeech": collect_ljspeech,
    "vctk": collect_vctk,
    "libritts": collect_libritts,
    "librispeech": collect_librispeech,
}


# --------------------------------------------------------------------------
def split_rows(rows, val_size: int, test_size: int, seed: int, by_speaker: bool):
    """Hold out utterances at random, keeping every speaker in every split.

    Speaker-disjoint splits are a *different* experiment (zero-shot speakers);
    `--speaker-disjoint` switches to that.
    """
    rng = random.Random(seed)
    if not by_speaker:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        return (shuffled[val_size + test_size:], shuffled[:val_size],
                shuffled[val_size: val_size + test_size])

    speakers = sorted({r[1] for r in rows})
    rng.shuffle(speakers)
    n_val = max(1, len(speakers) // 20)
    val_spk, test_spk = set(speakers[:n_val]), set(speakers[n_val: 2 * n_val])
    train = [r for r in rows if r[1] not in val_spk and r[1] not in test_spk]
    val = [r for r in rows if r[1] in val_spk][:val_size]
    test = [r for r in rows if r[1] in test_spk][:test_size]
    return train, val, test


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(COLLECTORS))
    ap.add_argument("--root", required=True, help="corpus directory")
    ap.add_argument("--out", default="filelists")
    ap.add_argument("--prefix", default=None, help="defaults to the dataset name")
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="LibriTTS/LibriSpeech subsets, e.g. train-clean-100")
    ap.add_argument("--mic", default="mic1", help="VCTK microphone (mic1 | mic2)")
    ap.add_argument("--raw-text", action="store_true",
                    help="LJSpeech: use the raw column; LibriTTS: use .original.txt")
    ap.add_argument("--speaker-disjoint", action="store_true",
                    help="hold out whole speakers instead of utterances")
    ap.add_argument("--min-chars", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory.\n\n{__doc__}", file=sys.stderr)
        return 1

    kwargs = {}
    if args.dataset == "ljspeech":
        kwargs["normalised_text"] = not args.raw_text
    elif args.dataset == "vctk":
        kwargs["mic"] = args.mic
    elif args.dataset == "libritts":
        kwargs["subsets"] = args.subsets
        kwargs["original_text"] = args.raw_text
    elif args.dataset == "librispeech":
        kwargs["subsets"] = args.subsets

    rows = COLLECTORS[args.dataset](root, **kwargs)
    rows = [r for r in rows if len(re.sub(r"\s+", "", r[2])) >= args.min_chars]
    if not rows:
        print(f"ERROR: no usable utterances found under {root}", file=sys.stderr)
        return 1

    train, val, test = split_rows(rows, args.val_size, args.test_size,
                                  args.seed, args.speaker_disjoint)
    prefix = args.prefix or args.dataset
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    single_speaker = len({r[1] for r in rows}) == 1
    for name, items in [("train", train), ("val", val), ("test", test), ("all", rows)]:
        path = out_dir / f"{prefix}_{name}.txt"
        lines = [f"{a}|{t}" if single_speaker else f"{a}|{s}|{t}" for a, s, t in items]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{path}  {len(items):>7d} utterances")

    speakers = sorted({r[1] for r in rows})
    print(f"\nspeakers : {len(speakers)}" + ("" if len(speakers) > 12 else f"  {speakers}"))
    print(f"columns  : {'[audio, text]' if single_speaker else '[audio, speaker, text]'}")
    print("\nNext:")
    print(f"  1. set data.root: {root.as_posix()}")
    print(f"     set data.manifest: {out_dir.as_posix()}/{prefix}_all.txt")
    if not single_speaker:
        print("     set data.columns: [audio, speaker, text]")
        print("     set model.n_speakers: auto")
    print("  2. python scripts/preprocess.py --config configs/<your>.yaml --workers 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
