"""Grapheme-to-phoneme backends.

Three frontends, deliberately kept interchangeable because *which one you use
changes what the model has to learn*:

``char``     raw graphemes. Every ambiguity survives into the model.
``ipa``      espeak-ng via `phonemizer`. Phonetic, but context-free per word,
             so homographs are resolved by espeak's own (limited) rules.
``arpabet``  g2p_en: CMUdict plus an NLTK part-of-speech tagger and an explicit
             homograph list, so it *attempts* contextual disambiguation.

None of them is an oracle. Run `python scripts/inspect_frontend.py` to see
exactly what each produces on the homograph probe set before choosing --
measured behaviour beats assumptions here, and both phonemisers get some of
these wrong.

Every backend returns **one list of phones per input word**, which is what
keeps the word map intact for the complexity heatmaps. When a backend's
sentence-level output cannot be aligned back to the input words, we fall back
to per-word phonemisation for that utterance rather than silently mis-aligning.
"""

from __future__ import annotations

import logging
import re
import warnings
from functools import lru_cache

WORD_RE = re.compile(r"[a-zA-Z']+")


class PhonemizerBase:
    """Interface: words in, one list of phone symbols per word out."""

    name = "base"
    is_phonetic = False

    def phonemize_words(self, words: list[str], text: str | None = None) -> list[list[str]]:
        raise NotImplementedError

    def __call__(self, words: list[str], text: str | None = None) -> list[list[str]]:
        return self.phonemize_words(words, text)


class CharPhonemizer(PhonemizerBase):
    """Identity frontend: the model sees letters and must do the rest itself."""

    name = "char"
    is_phonetic = False

    def phonemize_words(self, words, text=None):
        return [list(w) for w in words]


class EspeakPhonemizer(PhonemizerBase):
    """espeak-ng IPA.

    Uses `espeakng-loader` when available so no system package is required --
    that is what makes this work on a bare Colab runtime. Falls back to a
    system-installed espeak-ng if the loader is missing.
    """

    name = "ipa"
    is_phonetic = True

    def __init__(self, language: str = "en-us", with_stress: bool = True,
                 preserve_punctuation: bool = False):
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        _bind_espeak_library()
        # phonemizer warns per utterance about the word-count drift we handle
        # ourselves below; the counter is reported once at the end instead.
        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        self.language = language
        self._sep = Separator(word="|", syllable="", phone=" ")
        self._backend = EspeakBackend(
            language, preserve_punctuation=preserve_punctuation, with_stress=with_stress,
            language_switch="remove-flags", words_mismatch="ignore",
        )
        self._misaligned = 0

    def _phonemize_text(self, text: str) -> list[list[str]]:
        out = self._backend.phonemize([text], separator=self._sep, strip=True)[0]
        chunks = [c.strip() for c in out.split("|")]
        return [c.split() for c in chunks if c.strip()]

    def _phonemize_each(self, words: list[str]) -> list[list[str]]:
        """One espeak call, one line per word: alignment is guaranteed 1:1."""
        out = self._backend.phonemize(list(words), separator=self._sep, strip=True)
        result = []
        for word, line in zip(words, out):
            phones = [p for chunk in line.split("|") for p in chunk.split()]
            result.append(phones or list(word))
        return result

    def phonemize_words(self, words, text=None):
        if not words:
            return []
        if text:
            chunks = self._phonemize_text(text)
            if len(chunks) == len(words):
                return chunks
            # espeak occasionally merges a pair of function words, so its
            # word count drifts from ours. Silently zipping the two would
            # mis-attribute every phone after the merge and quietly corrupt
            # the word map the heatmaps are built on.
            self._misaligned += 1
        return self._phonemize_each(words)


class ArpabetPhonemizer(PhonemizerBase):
    """g2p_en: CMUdict + POS tagging + a homograph list."""

    name = "arpabet"
    is_phonetic = True

    def __init__(self, strip_stress: bool = False):
        from g2p_en import G2p

        _ensure_nltk_data()
        self._g2p = G2p()
        self.strip_stress = strip_stress
        self._misaligned = 0

    def _clean(self, phones: list[str]) -> list[str]:
        if self.strip_stress:
            return [re.sub(r"\d$", "", p) for p in phones]
        return phones

    def phonemize_words(self, words, text=None):
        if not words:
            return []
        if text:
            flat = self._g2p(text)
            chunks, current = [], []
            for tok in flat:
                if tok == " ":
                    if current:
                        chunks.append(current)
                    current = []
                elif re.match(r"^[A-Z]{1,3}\d?$", tok):
                    current.append(tok)
                else:  # punctuation: ends the current word
                    if current:
                        chunks.append(current)
                    current = []
            if current:
                chunks.append(current)
            if len(chunks) == len(words):
                return [self._clean(c) for c in chunks]
            self._misaligned += 1
        return [self._clean([p for p in self._g2p(w) if p != " "]) or list(w) for w in words]


@lru_cache(maxsize=1)
def _bind_espeak_library() -> bool:
    """Point `phonemizer` at the pip-installed espeak-ng, if present."""
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        return True
    except Exception:
        return False  # fall back to a system espeak-ng install


@lru_cache(maxsize=1)
def _ensure_nltk_data() -> None:
    try:
        import nltk

        for pkg, path in [
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("cmudict", "corpora/cmudict"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:  # pragma: no cover
        pass


BACKENDS = {"char": CharPhonemizer, "ipa": EspeakPhonemizer, "arpabet": ArpabetPhonemizer}


def build_phonemizer(name: str, **kwargs) -> PhonemizerBase:
    """Build a backend, degrading to characters with a loud warning."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown phonemizer '{name}'. Options: {sorted(BACKENDS)}")
    if name == "char":
        return CharPhonemizer()
    try:
        return BACKENDS[name](**kwargs)
    except Exception as exc:
        warnings.warn(
            f"Phonemizer backend '{name}' is unavailable ({exc}). Falling back to "
            f"character input. Install it with:\n"
            f"  ipa     : pip install phonemizer espeakng-loader\n"
            f"  arpabet : pip install g2p_en nltk\n"
            f"Training will still run, but on a different input representation "
            f"than the config asked for."
        )
        return CharPhonemizer()
