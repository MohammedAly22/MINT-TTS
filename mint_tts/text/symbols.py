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

# Arabic letter inventory (no diacritics: the corpus is undiacritised and the
# normaliser strips any that appear). Hamza forms are listed even though the
# default normaliser folds most of them, so that a run with `normalise_alef:
# false` still finds them in the table.
ARABIC_LETTERS_LIST = [chr(c) for c in range(0x0621, 0x063B)] +                       [chr(c) for c in range(0x0641, 0x064B)] +                       ["ٱ", "ٹ", "پ", "چ", "ژ",
                       "ڤ", "ک", "گ", "ھ", "ہ",
                       "ی", "ے"]
