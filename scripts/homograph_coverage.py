"""Does the training corpus actually contain both readings of each homograph?

A model cannot learn a distinction it has never heard. Before concluding that
"the model failed to disambiguate", check whether the evidence was there at
all: if `record` appears twelve times and eleven are the noun, no architecture
and no amount of training will teach the verb.

    python scripts/homograph_coverage.py --filelist filelists/ljspeech_train.txt

Reports, per homograph: how often it occurs, and how the occurrences split
between readings. The split is *estimated* with g2p_en's POS-aware G2P, which
is itself imperfect (see scripts/inspect_frontend.py) -- so treat the split as
indicative and the raw counts as solid. When the minority reading has only a
handful of instances, that alone settles it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mint_tts.data.manifest import read_manifest
from mint_tts.text.normalizer import TextNormalizer
from mint_tts.training.homograph import DEFAULT_PAIRS

# Words worth checking beyond the probe set: classic English heteronyms.
EXTRA_WORDS = [
    "bass", "bow", "dove", "invalid", "object", "project", "produce", "refuse",
    "permit", "content", "contract", "minute", "number", "row", "sow", "tear",
    "subject", "conduct", "console", "convert", "rebel", "separate", "moderate",
    "estimate", "graduate", "delegate", "advocate", "alternate", "appropriate",
]


def occurrences(texts: list[str], words: set[str]) -> dict[str, list[str]]:
    """Sentences containing each target word (normalised, word-boundary match)."""
    norm = TextNormalizer()
    found: dict[str, list[str]] = defaultdict(list)
    patterns = {w: re.compile(rf"\b{re.escape(w)}\b") for w in words}
    for raw in texts:
        text = norm(raw)
        for w, pat in patterns.items():
            if pat.search(text):
                found[w].append(text)
    return found


def estimate_readings(word: str, sentences: list[str], g2p) -> Counter:
    """Bucket occurrences by the pronunciation a POS-aware G2P predicts."""
    counts: Counter = Counter()
    if g2p is None:
        return counts
    for sent in sentences:
        try:
            flat = g2p(sent)
        except Exception:
            continue
        chunks, cur = [], []
        for tok in flat:
            if tok == " " or not re.match(r"^[A-Z]{1,3}\d?$", tok):
                if cur:
                    chunks.append(cur)
                cur = []
            else:
                cur.append(tok)
        if cur:
            chunks.append(cur)
        words = re.findall(r"[a-z']+", sent)
        if len(chunks) != len(words):
            continue
        for w, phones in zip(words, chunks):
            if w == word:
                counts[" ".join(phones)] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", required=True, help="training filelist or manifest")
    ap.add_argument("--columns", nargs="*", default=["audio", "text"])
    ap.add_argument("--min-minority", type=int, default=20,
                    help="instances of the rarer reading below which it is unlearnable")
    ap.add_argument("--extra-words", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    records = read_manifest(args.filelist, args.columns)
    texts = [r.text for r in records]
    words = {p.word.lower() for p in DEFAULT_PAIRS}
    words |= {w.lower() for w in (args.extra_words if args.extra_words is not None else EXTRA_WORDS)}

    print(f"{len(texts)} utterances from {args.filelist}\n")
    found = occurrences(texts, words)

    try:
        from g2p_en import G2p

        from mint_tts.text.phonemes import _ensure_nltk_data

        _ensure_nltk_data()
        g2p = G2p()
    except Exception as exc:
        print(f"[warn] g2p_en unavailable ({exc}); reporting raw counts only\n")
        g2p = None

    report, unlearnable, thin = {}, [], []
    print(f"{'word':<12}{'count':>7}  readings (estimated split)")
    print("-" * 74)
    for w in sorted(words):
        sents = found.get(w, [])
        readings = estimate_readings(w, sents, g2p)
        report[w] = {"count": len(sents), "readings": dict(readings)}
        if not sents:
            print(f"{w:<12}{0:>7}  -- never appears")
            unlearnable.append(w)
            continue
        top = readings.most_common()
        shown = "  ".join(f"{p}={n}" for p, n in top[:3]) if top else "(no split available)"
        print(f"{w:<12}{len(sents):>7}  {shown}")
        if len(top) < 2:
            unlearnable.append(w)
        elif sum(n for _, n in top[1:]) < args.min_minority:
            thin.append((w, sum(n for _, n in top[1:])))

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if unlearnable:
        print(f"\nOnly ever seen ONE reading ({len(unlearnable)}):")
        print("  " + ", ".join(unlearnable))
        print("  The corpus contains no evidence for the other pronunciation, so no")
        print("  model trained on it can learn the distinction. Getting these wrong")
        print("  is a property of the DATA, not of the architecture.")
    if thin:
        print(f"\nSecond reading present but thin (< {args.min_minority} instances):")
        for w, n in sorted(thin, key=lambda x: x[1]):
            print(f"  {w:<12} {n} instances of the rarer reading")
    solid = [w for w in sorted(words)
             if w not in unlearnable and w not in {t[0] for t in thin} and found.get(w)]
    if solid:
        print(f"\nEnough evidence to be worth testing ({len(solid)}):")
        print("  " + ", ".join(solid))
    else:
        print("\nNo homograph in this list has enough of both readings to be learnable")
        print("here. Evaluate disambiguation on a larger or more varied corpus")
        print("(LibriTTS), or add text-only pronunciation supervision.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
