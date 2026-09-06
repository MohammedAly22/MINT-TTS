"""Compare text frontends on homographs -- the experiment that decides which
one MINT-TTS should be trained with.

    python scripts/inspect_frontend.py
    python scripts/inspect_frontend.py --text "I read a book yesterday."
    python scripts/inspect_frontend.py --json outputs/frontend_report.json

For each probe sentence it prints the phones every backend produces for the
ambiguous word, and reports whether the backend gave the *same* pronunciation
in both contexts. A backend that never varies is not disambiguating -- it is
guessing, and whatever it guesses becomes a fixed input the acoustic model can
never recover from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mint_tts.text.normalizer import TextNormalizer
from mint_tts.text.symbols import PUNCTUATION, SPECIALS
from mint_tts.text.tokenizer import TextProcessor

# Minimal pairs: same spelling, different pronunciation, decided by context.
MINIMAL_PAIRS = [
    ("read", "I read a book yesterday.", "I will read a book tomorrow."),
    ("record", "Please record the album.", "She bought the record."),
    ("lead", "He will lead the team.", "The pipe is made of lead."),
    ("desert", "Do not desert your post.", "They crossed the desert."),
    ("live", "They live in Berlin.", "It was a live broadcast."),
    ("bass", "He plays the bass guitar.", "She caught a large bass."),
    ("present", "Please present your work.", "She opened the present."),
    ("wind", "Wind the clock carefully.", "The wind was strong."),
    ("tear", "A tear rolled down.", "Do not tear the paper."),
    ("close", "Please close the door.", "The station is close."),
]


def phones_for_word(tp: TextProcessor, sentence: str, target: str) -> list[str]:
    """Phones belonging to `target` only.

    Boundary and punctuation tokens share a word id with their neighbour, so
    they must be filtered out -- otherwise a word at the end of a sentence
    picks up ". <eos>" and looks different from the same word mid-sentence.
    """
    enc = tp.encode(sentence)
    target = TextNormalizer()(target).strip(".,")
    skip = set(SPECIALS) | set(PUNCTUATION)
    for i, word in enumerate(enc.words):
        if word == target:
            return [t for t, w in zip(enc.tokens, enc.word_ids)
                    if w == i and t not in skip]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="*", default=["char", "ipa", "arpabet"])
    ap.add_argument("--text", default=None, help="phonemise one sentence and exit")
    ap.add_argument("--json", default=None, help="also write the report here")
    args = ap.parse_args()

    processors = {}
    for name in args.backends:
        tp = TextProcessor(name)
        if tp.input_type != name:
            print(f"[warn] backend '{name}' unavailable, it fell back to '{tp.input_type}'")
        processors[name] = tp

    if args.text:
        for name, tp in processors.items():
            enc = tp.encode(args.text)
            print(f"{name:>8s}: {' '.join(enc.tokens)}")
        return 0

    report, summary = [], {name: {"varies": 0, "total": 0} for name in processors}
    print("=" * 78)
    print("HOMOGRAPH DISAMBIGUATION BY FRONTEND")
    print("=" * 78)

    for word, sent_a, sent_b in MINIMAL_PAIRS:
        print(f"\n'{word}'")
        print(f"   A: {sent_a}")
        print(f"   B: {sent_b}")
        entry = {"word": word, "sentence_a": sent_a, "sentence_b": sent_b, "backends": {}}
        for name, tp in processors.items():
            pa = phones_for_word(tp, sent_a, word)
            pb = phones_for_word(tp, sent_b, word)
            varies = bool(pa) and bool(pb) and pa != pb
            summary[name]["total"] += 1
            summary[name]["varies"] += int(varies)
            flag = "DIFFERS" if varies else "same   "
            print(f"      {name:>8s} [{flag}]  A={' '.join(pa) or '?':<22s} B={' '.join(pb) or '?'}")
            entry["backends"][name] = {"a": pa, "b": pb, "varies": varies}
        report.append(entry)

    print("\n" + "=" * 78)
    print("SUMMARY -- how often the frontend produced two different pronunciations")
    print("=" * 78)
    for name, s in summary.items():
        pct = 100.0 * s["varies"] / max(s["total"], 1)
        print(f"  {name:>8s}: {s['varies']}/{s['total']} pairs ({pct:.0f}%)")

    print(
        "\nHow to read this:\n"
        "  'same'    the frontend committed to ONE pronunciation for both contexts.\n"
        "            Half of those commitments are wrong, and the acoustic model\n"
        "            cannot undo them -- the information is already gone.\n"
        "  'DIFFERS' the frontend varied with context. That is necessary but not\n"
        "            sufficient: it can still vary in the WRONG direction, so check\n"
        "            the actual phones above rather than trusting the count.\n"
        "  char      never varies by construction; the ambiguity is handed to the\n"
        "            model intact, which is the setting the MINT-TTS hypothesis is\n"
        "            actually about.\n"
    )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"summary": summary, "pairs": report}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
