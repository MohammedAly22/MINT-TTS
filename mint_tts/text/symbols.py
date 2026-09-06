"""Special symbols shared by every frontend.

Phone inventories are *not* hard-coded: they are collected from the corpus
during preprocessing and written to `symbols.json`. That keeps IPA (whose
inventory depends on the language and on the espeak-ng version) working
without anyone maintaining a phone list by hand.
"""

from __future__ import annotations

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPACE = "<sp>"          # word boundary
SPECIALS = [PAD, BOS, EOS, UNK, SPACE]

PUNCTUATION = list("!'(),-.:;?\"")

LETTERS = list("abcdefghijklmnopqrstuvwxyz'")

# ARPAbet, kept only so `scripts/inspect_frontend.py` and tests can reason
# about g2p_en output; the live table is built from data.
ARPABET_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
                  "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
