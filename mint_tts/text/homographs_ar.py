"""Egyptian Arabic: what makes a word hard, and where the probes come from.

**There is deliberately no homograph word list here.** An earlier version of
this file contained one -- 47 hand-written entries -- and it was the wrong
design:

* it covered almost nothing of a 62k-word vocabulary;
* it encoded assumptions rather than the corpus (it marked `مصر` ambiguous,
  which in a corpus of history and popular-science programmes it is not);
* it would have to be rewritten by hand for every new dialect or language,
  which is exactly the manual supervision the rest of this repository avoids.

Ambiguity is **discovered from the data** instead -- see `text/ambiguity.py`
for the measurement and `scripts/mine_ambiguity.py` for the pass that runs it.
A word is ambiguous when its pronunciation varies across the corpus *and* the
surrounding context predicts which variant occurs. Nothing is declared.

What remains here is Egyptian-specific *linguistic structure* that is a fact
about the writing system rather than a claim about particular words:

``clitic_gender_ambiguity``  a word ending in a bare kaf carries a
                             second-person suffix whose vowel encodes the
                             addressee's gender, and that vowel is never
                             written. This is a property of Arabic
                             orthography, true of any word with that ending,
                             so it needs no list.
``sentence_has_feminine_cue`` whether anything in the sentence resolves it.

These are used to *describe* a sentence (in probe reports and coverage
checks), and as a small structural prior that applies before any mining has
been done. They are not a pronunciation dictionary and the model never
consults them at inference.

The probe sentences are likewise mined from the corpus rather than written
here: see ``build_probe_set``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# structural ambiguity: facts about the orthography, not about words
# --------------------------------------------------------------------------
_KAF = "ك"
_TA_YA = "تي"      # -ti, the 2fs verb ending
_YA = "ي"
_MIN_CLITIC_LEN = 4

# Explicit second-person feminine pronouns and vocatives. These are closed
# grammatical classes, not a lexicon of content words: Arabic has exactly
# these forms, and listing them is describing the grammar rather than
# guessing about vocabulary.
FEMININE_PRONOUNS = {
    "انتي",      # enti
    "إنتي",
    "ليكي",
    "بيكي",
}
FEMININE_VOCATIVES = {
    "بنت",            # bint
    "يابنت",
    "هانم",      # hanem
    "مدام",      # madam
    "ست",                  # sitt
}

# Words that end in kaf as part of the root rather than as a clitic. Short
# and closed enough to enumerate; everything longer is handled by the length
# rule below.
_KAF_FINAL_ROOTS = {
    "ملك",   # malik / mulk
    "سمك",   # samak -- fish
    "ضحك",   # dihik -- laughed
    "مسك",   # misik -- held
    "شبك",   # shabak
    "سلك",   # silk
    "فلك",   # falak
    "بنك",   # bank
}


def clitic_gender_ambiguity(word: str) -> bool:
    """Does this word end in a 2nd-person clitic whose vowel is unwritten?

    `عمرك` is *3omrak* to a man and *3omrik* to a woman; the spelling is
    identical because Arabic does not write short vowels. This is true of any
    word carrying that suffix, which is why it can be detected structurally
    instead of listed.
    """
    return (
        len(word) >= _MIN_CLITIC_LEN
        and word.endswith(_KAF)
        and word not in _KAF_FINAL_ROOTS
    )


def is_feminine_verb(word: str) -> bool:
    """2nd-person-feminine-singular verb: ends in -ti, or -i on a longer stem.

    `فكرتي` (you[f] thought), `تجاوبي` (you[f] answer). This is the evidence
    that resolves a clitic, and it is typically several words away -- which is
    the reason this project routes per token over full-sentence attention.
    """
    if len(word) >= 5 and word.endswith(_TA_YA):
        return True
    # A longer word ending in bare ya, with an imperfective prefix.
    return len(word) >= 6 and word.endswith(_YA) and word[0] in "تبه"


def sentence_has_feminine_cue(words: list[str]) -> bool:
    """Is there evidence in the sentence that the addressee is female?"""
    for w in words:
        if w in FEMININE_PRONOUNS or w in FEMININE_VOCATIVES:
            return True
        if is_feminine_verb(w):
            return True
    return False


def cue_distance(words: list[str]) -> int:
    """How far is the resolving cue from the clitic it resolves?

    Returns -1 when there is no clitic or no cue. This is the quantity that
    makes a sentence hard in the way this architecture is meant to address:
    a large distance means a token needs several rounds of attention before
    the evidence reaches it.
    """
    clitics = [i for i, w in enumerate(words) if clitic_gender_ambiguity(w)]
    cues = [i for i, w in enumerate(words)
            if is_feminine_verb(w) or w in FEMININE_PRONOUNS or w in FEMININE_VOCATIVES]
    if not clitics or not cues:
        return -1
    return min(abs(c - q) for c in clitics for q in cues)


def structural_difficulty(words: list[str]) -> list[float]:
    """A per-word prior from orthography alone, used before mining has run.

    Only 0.5 (an unwritten-vowel clitic) or 0.0. Deliberately coarse and
    deliberately *not* a claim about any particular vocabulary item: it says
    "this word's ending hides a vowel", which is a fact about the script.

    Once `ambiguity.json` exists it is used instead, because it is measured
    rather than assumed. This is the bootstrap.
    """
    return [0.5 if clitic_gender_ambiguity(w) else 0.0 for w in words]


# --------------------------------------------------------------------------
# probe sets, mined from the corpus
# --------------------------------------------------------------------------
@dataclass
class ArabicPair:
    """Two real corpus sentences containing the same word."""

    word: str
    a: str
    b: str
    kind: str = "mined"
    note: str = ""
    extras: dict = field(default_factory=dict)


@dataclass
class ProbeSet:
    pairs: list[ArabicPair] = field(default_factory=list)
    easy: list[str] = field(default_factory=list)
    hard: list[str] = field(default_factory=list)
    long_easy: list[str] = field(default_factory=list)


_WS = re.compile(r"\s+")


def build_probe_set(
    rows: list[dict],
    ambiguity,
    n_pairs: int = 10,
    n_easy: int = 3,
    n_long_easy: int = 2,
    min_words: int = 4,
    max_words: int = 18,
) -> ProbeSet:
    """Build the probe set from real corpus utterances.

    Hand-written probe sentences have the same defect as a hand-written
    lexicon: they test what the author imagined rather than what the corpus
    contains, and a model can look good on them while failing on the
    distribution it was actually trained on. Mining them means the probes are
    in-domain by construction.

    ``ambiguity`` is an ``AmbiguityTable``. Pairs are chosen for the words it
    scored highest, taking two utterances whose *contexts differ most*, so the
    pair is a genuine minimal pair rather than two similar sentences.
    """
    by_word: dict[str, list[dict]] = {}
    for row in rows:
        words = row.get("words", [])
        if not (min_words <= len(words) <= max_words):
            continue
        for w in set(words):
            by_word.setdefault(w, []).append(row)

    pairs: list[ArabicPair] = []
    for word, score in ambiguity.top(n_pairs * 6):
        if score <= 0 or len(pairs) >= n_pairs:
            continue
        candidates = by_word.get(word, [])
        if len(candidates) < 2:
            continue
        a, b = _most_different(candidates, word)
        if a is None:
            continue
        pairs.append(ArabicPair(
            word=word,
            a=_WS.sub(" ", a.get("clean_text", "")).strip(),
            b=_WS.sub(" ", b.get("clean_text", "")).strip(),
            kind="clitic" if clitic_gender_ambiguity(word) else "mined",
            note=f"mined: ambiguity={score:.2f}",
        ))

    # Easy / hard controls, ranked by the mined scores of their words.
    scored: list[tuple[float, int, dict]] = []
    for row in rows:
        words = row.get("words", [])
        if not (min_words <= len(words) <= max_words):
            continue
        vals = ambiguity.profile(words)
        scored.append((max(vals) if vals else 0.0, len(words), row))

    easy = [r for s, n, r in sorted(scored, key=lambda t: (t[0], t[1]))
            if s == 0.0][:n_easy]
    hard = [r for s, n, r in sorted(scored, key=lambda t: -t[0])][:n_pairs]
    long_easy = [r for s, n, r in
                 sorted(scored, key=lambda t: (t[0], -t[1])) if s == 0.0][:n_long_easy]

    text = lambda rs: [_WS.sub(" ", r.get("clean_text", "")).strip() for r in rs]
    return ProbeSet(pairs=pairs, easy=text(easy), hard=text(hard),
                    long_easy=text(long_easy))


def _most_different(candidates: list[dict], word: str):
    """Two utterances whose semantic context around `word` differs most.

    Falls back to the first two when the cached vectors are unavailable, so
    probe construction never hard-fails on a missing file.
    """
    import numpy as np
    from pathlib import Path

    vecs, rows = [], []
    for row in candidates[:40]:
        path = row.get("semantic")
        words = row.get("words", [])
        if not path or not Path(path).exists() or word not in words:
            continue
        try:
            arr = np.load(path)
        except Exception:
            continue
        i = words.index(word)
        if i < len(arr):
            vecs.append(arr[i])
            rows.append(row)
    if len(vecs) < 2:
        return (candidates[0], candidates[1]) if len(candidates) >= 2 else (None, None)
    v = np.stack(vecs)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    sims = v @ v.T
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    return rows[int(i)], rows[int(j)]
