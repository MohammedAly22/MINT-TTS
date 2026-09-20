"""Arabic (and specifically Egyptian Arabic) text frontend.

The corpus this module exists for -- 100 h of single-speaker Egyptian Arabic --
is **undiacritised and unpunctuated**. That is not a defect to be patched
around; it is the experiment. A diacritised corpus hands the model the answer
(the flag and the science are spelled differently once tashkeel is present),
so nothing has to be inferred and the homograph question never arises.
Undiacritised text forces the model to recover the pronunciation from context,
which is exactly the claim MINT-TTS makes about where computation should go.

What this module provides
-------------------------
``ArabicNormalizer``      an ordered pipeline mirroring ``TextNormalizer`` but
                          for Arabic orthography: Arabic-Indic digits, tatweel,
                          Quranic marks, optional letter folding, and number
                          verbalisation in Egyptian Arabic (see numbers_ar.py).
``ArabicCharPhonemizer``  identity frontend over Arabic graphemes. Every
                          ambiguity survives into the model, which is the only
                          setting in which the hypothesis is testable.

Letter folding is a *choice with consequences*, so each part is a separate
flag rather than one "normalise" switch:

``normalise_alef``       the hamza-seated alefs fold to bare alef. Almost
                         always right for TTS: the seat is orthographic, and
                         Egyptian speech does not distinguish them in most
                         positions.
``normalise_ya``         alef maqsura folds to ya. Egyptian writing uses the
                         two interchangeably, so folding them shrinks the
                         vocabulary without losing a pronunciation distinction.
``normalise_ta_marbuta`` ta marbuta folds to ha. **Off by default.** In
                         Egyptian Arabic the two are frequently homophonous in
                         pause, but ta marbuta carries the feminine morphology
                         and surfaces as /t/ in construct state. Folding it
                         destroys evidence the model needs.
``strip_diacritics``     remove tashkeel if any is present. On by default so a
                         partially-diacritised corpus cannot leak the answer
                         for some utterances and not others -- an inconsistent
                         input distribution is worse than a uniformly hard one.
"""

from __future__ import annotations

import re
import unicodedata

from .numbers_ar import decimal_to_words, digits_to_words, number_to_words


# --------------------------------------------------------------------------
# character classes
# --------------------------------------------------------------------------
# Tashkeel (harakat) plus the Quranic annotation marks. Tatweel (U+0640) is a
# pure elongation glyph and is handled separately, because removing it is
# always safe while removing harakat is a deliberate experimental choice.
DIACRITICS = (
    "ًٌٍَُِّْٓٔ"
    "ٕٖٜٗ٘ٙٚٛٝٞ"
    "ٰٟۖۗۘۙۚۛۜ۟"
    "ۣ۪ۭ۠ۡۢۤۧۨ۫۬"
)
TATWEEL = "ـ"

_DIACRITIC_RE = re.compile(f"[{DIACRITICS}]")
_TATWEEL_RE = re.compile(f"{TATWEEL}+")

# Arabic-Indic and Extended Arabic-Indic digits -> ASCII.
_DIGIT_MAP = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}

# Arabic punctuation -> the ASCII equivalents the symbol table already holds,
# so one punctuation inventory serves every language.
_PUNCT_MAP = {
    "،": ",",    # Arabic comma
    "؛": ";",    # Arabic semicolon
    "؟": "?",    # Arabic question mark
    "٪": "%",
    "٫": ".",    # Arabic decimal separator
    "٬": ",",    # Arabic thousands separator
    "۔": ".",    # Arabic full stop
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "«": '"', "»": '"',
    "…": ".",
}

# Latin letters are kept: Egyptian speech is full of code-switched English,
# and the multilingual goal needs it to survive the frontend rather than be
# silently deleted.
ARABIC_LETTERS = "ء-غف-يٱ-ۓ"
_ARABIC_WORD_CHARS = ARABIC_LETTERS + "a-zA-Z0-9'"

# Tatweel deliberately absent: it is stripped before tokenisation.
ARABIC_WORD_RE = re.compile(f"[{_ARABIC_WORD_CHARS}]+")

_ALEF_RE = re.compile("[آأإٱ]")
_ALEF_BARE = "ا"
_YA_MAQSURA = "ى"
_YA = "ي"
_TA_MARBUTA = "ة"
_HA = "ه"
_WS_RE = re.compile(r"\s+")

# Standalone symbols that must be spoken rather than dropped.
_SYMBOL_WORDS = {
    "%": " في المية ",   # fi el-miyya
    "+": " زائد ",                       # zaid
    "=": " يساوي ",                 # yisawi
    "&": " و ",                                          # wa
    "@": " ات ",                                    # at
}

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def strip_diacritics(text: str) -> str:
    """Remove tashkeel. The corpus has none; this guards against mixed input."""
    return _DIACRITIC_RE.sub("", text)


def strip_tatweel(text: str) -> str:
    """Remove the kashida elongation glyph, which carries no pronunciation."""
    return _TATWEEL_RE.sub("", text)


def normalise_digits(text: str) -> str:
    """Arabic-Indic digits -> ASCII, so one number rule handles both."""
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def normalise_punctuation(text: str) -> str:
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return text


def normalise_alef(text: str) -> str:
    """Hamza-seated and wasla alefs fold to bare alef."""
    return _ALEF_RE.sub(_ALEF_BARE, text)


def normalise_ya(text: str) -> str:
    """Alef maqsura folds to ya."""
    return text.replace(_YA_MAQSURA, _YA)


def normalise_ta_marbuta(text: str) -> str:
    """Ta marbuta folds to ha. Lossy; see the module docstring before enabling."""
    return text.replace(_TA_MARBUTA, _HA)


def expand_numbers(text: str) -> str:
    """Verbalise digit runs in EGYPTIAN Arabic.

    Deliberately not ``num2words(lang="ar")``, which emits Modern Standard
    Arabic in the nominative case: it writes `3ishruun` where every speaker
    in this corpus says `3ishriin`, `ithnaan` for `itnein`, `mi'a` for
    `miyya`. The transcript is what the aligner maps onto the audio, so an
    MSA spelling of a number is a text/audio mismatch the model would be
    trained on. See text/numbers_ar.py.
    """
    def repl(m: re.Match) -> str:
        raw = m.group(0)
        try:
            if "." in raw:
                return f" {decimal_to_words(raw)} "
            return f" {number_to_words(int(raw))} "
        except Exception:
            # An unparseable run (absurdly long, say) is read digit by digit
            # rather than dropped, so nothing silently vanishes from the
            # transcript.
            return " " + digits_to_words(raw) + " "
    return _NUM_RE.sub(repl, text)


def expand_symbols(text: str) -> str:
    for sym, word in _SYMBOL_WORDS.items():
        text = text.replace(sym, word)
    return text


def collapse_whitespace(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?]){2,}", r"\1", text)
    return text.strip()


class ArabicNormalizer:
    """Ordered Arabic normalisation pipeline.

    Mirrors ``TextNormalizer``'s interface (``__call__``, ``trace``, ``skip``)
    so the two are interchangeable everywhere a normalizer is accepted.
    """

    def __init__(
        self,
        strip_diacritics: bool = True,
        strip_tatweel: bool = True,
        normalise_alef: bool = True,
        normalise_ya: bool = True,
        normalise_ta_marbuta: bool = False,
        expand_numbers: bool = True,
        keep_punctuation: str = "!'(),-.:;?\"",
        lowercase: bool = True,
        skip: tuple[str, ...] = (),
    ):
        self.flags = {
            "diacritics": strip_diacritics,
            "tatweel": strip_tatweel,
            "alef": normalise_alef,
            "ya": normalise_ya,
            "ta_marbuta": normalise_ta_marbuta,
            "numbers": expand_numbers,
        }
        self.keep_punctuation = keep_punctuation
        self.lowercase = lowercase
        self.skip = set(skip)
        allowed = re.escape(keep_punctuation)
        # Anything that is not an Arabic letter, a Latin letter/digit, a kept
        # punctuation mark or whitespace becomes a space. Emoji and stray
        # control characters are common in YouTube-derived transcripts.
        self._strip_re = re.compile(f"[^{ARABIC_LETTERS}a-zA-Z0-9\\s{allowed}]")

    @property
    def steps(self) -> list[tuple[str, object]]:
        """The active pipeline, in order.

        Order matters: digits must become ASCII before the number rule runs,
        and the symbol words are inserted before the final strip so that the
        Arabic letters they introduce survive it.
        """
        out: list[tuple[str, object]] = [
            ("unicode", lambda t: unicodedata.normalize("NFC", t)),
            ("punctuation", normalise_punctuation),
            ("digits", normalise_digits),
        ]
        if self.flags["tatweel"]:
            out.append(("tatweel", strip_tatweel))
        if self.flags["diacritics"]:
            out.append(("diacritics", strip_diacritics))
        if self.flags["numbers"]:
            out.append(("numbers", expand_numbers))
        out.append(("symbols", expand_symbols))
        if self.flags["alef"]:
            out.append(("alef", normalise_alef))
        if self.flags["ya"]:
            out.append(("ya", normalise_ya))
        if self.flags["ta_marbuta"]:
            out.append(("ta_marbuta", normalise_ta_marbuta))
        return out

    def normalize(self, text: str) -> str:
        for name, fn in self.steps:
            if name not in self.skip:
                text = fn(text)
        if self.lowercase:
            # Only affects code-switched Latin; Arabic has no case.
            text = text.lower()
        text = self._strip_re.sub(" ", text)
        return collapse_whitespace(text)

    __call__ = normalize

    def trace(self, text: str) -> list[tuple[str, str]]:
        """Return the text after each step -- for debugging a bad expansion."""
        out = [("input", text)]
        for name, fn in self.steps:
            if name in self.skip:
                continue
            text = fn(text)
            out.append((name, collapse_whitespace(text)))
        return out


def build_arabic_normalizer(cfg_text) -> ArabicNormalizer:
    """Construct an ArabicNormalizer from the ``text:`` config section."""
    a = dict(cfg_text.get("arabic", {}) or {})
    return ArabicNormalizer(
        strip_diacritics=bool(a.get("strip_diacritics", True)),
        strip_tatweel=bool(a.get("strip_tatweel", True)),
        normalise_alef=bool(a.get("normalise_alef", True)),
        normalise_ya=bool(a.get("normalise_ya", True)),
        normalise_ta_marbuta=bool(a.get("normalise_ta_marbuta", False)),
        expand_numbers=bool(a.get("expand_numbers", True)),
        keep_punctuation=cfg_text.get("keep_punctuation", "!'(),-.:;?\""),
        lowercase=bool(cfg_text.get("lowercase", True)),
        skip=tuple(cfg_text.get("skip_normalisation_steps", [])),
    )
