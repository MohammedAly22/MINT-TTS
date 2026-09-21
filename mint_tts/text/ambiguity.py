"""Learned ambiguity: discovering homographs from the corpus, not a word list.

The problem with a hand-written lexicon
---------------------------------------
An earlier version of this file was a list of Egyptian homographs I wrote by
hand. That was the wrong design, for reasons that are worth stating plainly
because they generalise:

* **It does not scale.** 47 entries against a 62k-word vocabulary covers
  almost nothing, and every word outside it is treated as unambiguous.
* **It encodes assumptions, not data.** The list said `مصر` was ambiguous
  (Egypt / "he insisted"). In *this* corpus -- popular science and history
  programmes -- it is Egypt essentially every time. Marking it hard spends
  computation on a word that never needed it.
* **It does not transfer.** Moving to another dialect, another corpus or
  another language would mean writing another list by hand, which is exactly
  the kind of manual supervision the rest of this repository avoids (durations
  are learned by the aligner; compute labels are generated, not annotated).

So ambiguity is **measured from the data** instead.

The signal
----------
A homograph is a spelling whose *pronunciation varies with context*. Both
halves of that sentence are observable:

    pronunciation   the mel frames aligned to that word, which the aligner
                    already produces during training
    context         the frozen LM's contextual vector for that occurrence

For every word type with enough occurrences, we ask:

1. **Does its acoustic realisation vary more than a typical word's?**
   Measured as the spread of its per-occurrence acoustic embeddings. A word
   said the same way every time has a tight cluster.

2. **Is that variation predictable from context?** Measured by how well the
   LM vectors separate the acoustic clusters. This is the half that matters:
   *unpredictable* acoustic variation is just noise -- different sentence
   positions, different prosody, a cough -- and spending compute on it buys
   nothing. Variation the context explains is exactly what extra computation
   can resolve.

The product of the two is the ambiguity score. A word scores high only when
it is said in genuinely different ways **and** the surrounding words say which
way applies.

What this buys
--------------
Nothing is declared. `علم` is discovered to be ambiguous because the corpus
contains it pronounced two ways in predictably different contexts; `مصر` is
discovered not to be, because it is not. A word nobody thought of is found on
the same evidence as one that is obvious. Moving to another language changes
nothing in this file.

The result is still only a **prior on the compute penalty** -- it says where
thinking is cheap, never what the answer is. The model never sees these
scores at inference time and must still resolve the reading itself.

Bootstrapping
-------------
Acoustic evidence needs durations, which need a trained aligner, so this runs
as a *second pass*: train until alignment is healthy, mine the corpus, then
continue with the discovered scores. Before that pass exists there is a
text-only fallback (see ``ContextualAmbiguity``) that needs no audio and no
alignment, and uses the LM alone.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WordStats:
    """Everything measured about one word type."""

    word: str
    count: int = 0
    acoustic_spread: float = 0.0      # how much its pronunciation varies
    context_predictivity: float = 0.0  # how well context explains that variation
    ambiguity: float = 0.0             # the product -- the score that is used
    separation: float = 0.0            # cosine distance between the two clusters
    n_clusters: int = 1
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "word": self.word, "count": self.count,
            "acoustic_spread": round(self.acoustic_spread, 4),
            "separation": round(self.separation, 4),
            "context_predictivity": round(self.context_predictivity, 4),
            "ambiguity": round(self.ambiguity, 4),
            "n_clusters": self.n_clusters,
            "examples": self.examples[:4],
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _unit(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows, so comparisons are angular rather than by magnitude."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def _centre(x: np.ndarray) -> np.ndarray:
    """Subtract the mean vector before comparing -- removes anisotropy.

    Language-model embeddings are strongly anisotropic: every vector sits in
    a narrow cone, so raw cosine between any two of them is ~0.95-0.99.
    Measured on MARBERTv2's last layer, two completely UNRELATED Arabic words
    score 0.979 -- the same as the two readings of a homograph. Any spread or
    separation computed on the raw vectors is then dominated by that shared
    mean direction rather than by what distinguishes the occurrences.

    Subtracting the per-word mean removes the cone and leaves the differences
    the measurement is actually about. On the same data this turns a +0.011
    raw margin into a clean split (+0.135 within-reading vs -0.295 across).

    Applied only to the context (language-model) side. Acoustic descriptors
    are mel statistics, which are not anisotropic in this way, and centring
    them would discard the overall spectral level that legitimately
    distinguishes two pronunciations.
    """
    if len(x) < 2:
        return x
    return x - x.mean(0, keepdims=True)


def _mean_pairwise_distance(x: np.ndarray, max_pairs: int = 400,
                            rng: np.random.Generator | None = None) -> float:
    """Average cosine distance between occurrences: the spread of a cluster."""
    if len(x) < 2:
        return 0.0
    u = _unit(x)
    if len(u) * (len(u) - 1) // 2 <= max_pairs:
        sims = u @ u.T
        iu = np.triu_indices(len(u), k=1)
        return float(1.0 - sims[iu].mean())
    rng = rng or np.random.default_rng(0)
    i = rng.integers(0, len(u), max_pairs)
    j = rng.integers(0, len(u), max_pairs)
    keep = i != j
    if not keep.any():
        return 0.0
    return float(1.0 - (u[i[keep]] * u[j[keep]]).sum(-1).mean())


def _two_means(x: np.ndarray, iters: int = 25,
               rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Split occurrences into two clusters; return labels and the separation.

    Two rather than k: the question is "does this word have more than one
    pronunciation", and two clusters answer it. Choosing k per word would add
    a model-selection problem whose errors are hard to distinguish from the
    effect being measured.
    """
    rng = rng or np.random.default_rng(0)
    u = _unit(x)
    if len(u) < 4:
        return np.zeros(len(u), dtype=int), 0.0
    # Seed with the two most distant points, so the split starts from the
    # widest axis of variation rather than an arbitrary one.
    sims = u @ u.T
    a, b = np.unravel_index(np.argmin(sims), sims.shape)
    centres = u[[a, b]].copy()
    labels = np.zeros(len(u), dtype=int)
    for _ in range(iters):
        new = (u @ centres.T).argmax(1)
        if (new == labels).all():
            break
        labels = new
        for c in (0, 1):
            if (labels == c).any():
                centres[c] = _unit(u[labels == c].mean(0, keepdims=True))[0]
    if len(set(labels.tolist())) < 2:
        return labels, 0.0
    separation = float(1.0 - (centres[0] @ centres[1]))
    return labels, separation


def _predictivity(context: np.ndarray, labels: np.ndarray,
                  rng: np.random.Generator | None = None) -> float:
    """How well do the context vectors predict the acoustic cluster?

    A nearest-centroid classifier under leave-one-out, scored against the
    majority-class baseline. 0 means context says nothing about which
    pronunciation occurs (so the acoustic variation is noise); 1 means it
    decides it completely.

    This is the load-bearing measurement. Acoustic variation alone would
    flag any word that happens to appear in varied prosodic positions.
    """
    if len(context) < 4 or len(set(labels.tolist())) < 2:
        return 0.0
    u = _unit(_centre(context))
    majority = max((labels == c).mean() for c in (0, 1))
    correct = 0
    for i in range(len(u)):
        keep = np.ones(len(u), dtype=bool)
        keep[i] = False
        cents = []
        for c in (0, 1):
            sel = keep & (labels == c)
            if not sel.any():
                cents.append(None)
                continue
            cents.append(_unit(u[sel].mean(0, keepdims=True))[0])
        if cents[0] is None or cents[1] is None:
            continue
        pred = int((u[i] @ cents[1]) > (u[i] @ cents[0]))
        correct += int(pred == labels[i])
    accuracy = correct / len(u)
    # Rescale against the majority baseline: beating "always guess the
    # common reading" is what counts, not raw accuracy.
    if majority >= 0.999:
        return 0.0
    return float(max(0.0, (accuracy - majority) / (1.0 - majority)))


# --------------------------------------------------------------------------
# the miner
# --------------------------------------------------------------------------
class AmbiguityMiner:
    """Discovers which words are ambiguous, from acoustics + context.

    Usage (see scripts/mine_ambiguity.py)::

        miner = AmbiguityMiner(min_count=6)
        for occurrence in corpus:
            miner.add(word, acoustic_vec, context_vec, sentence)
        table = miner.finalise()
    """

    def __init__(
        self,
        min_count: int = 6,
        max_occurrences: int = 60,
        min_spread: float = 0.05,
        separation_scale: float = 0.5,
        seed: int = 0,
    ):
        self.min_count = int(min_count)
        self.max_occurrences = int(max_occurrences)
        self.min_spread = float(min_spread)
        # Cluster separation at which the acoustic half of the score
        # saturates. 0.5 cosine distance is a large acoustic difference --
        # comfortably more than prosodic variation on one pronunciation.
        self.separation_scale = float(separation_scale)
        self.rng = np.random.default_rng(seed)
        self._acoustic: dict[str, list[np.ndarray]] = defaultdict(list)
        self._context: dict[str, list[np.ndarray]] = defaultdict(list)
        self._sentences: dict[str, list[str]] = defaultdict(list)
        self._counts: dict[str, int] = defaultdict(int)

    def add(self, word: str, acoustic: np.ndarray, context: np.ndarray,
            sentence: str = "") -> None:
        """Record one occurrence of one word."""
        self._counts[word] += 1
        # Cap what is kept per word: a function word appearing 20k times would
        # otherwise dominate memory, and its statistics are settled long
        # before that.
        if len(self._acoustic[word]) >= self.max_occurrences:
            return
        self._acoustic[word].append(np.asarray(acoustic, dtype=np.float32))
        self._context[word].append(np.asarray(context, dtype=np.float32))
        if sentence:
            self._sentences[word].append(sentence)

    def finalise(self) -> dict[str, WordStats]:
        """Score every word type that has enough occurrences."""
        out: dict[str, WordStats] = {}
        spreads: list[float] = []

        for word, acoustics in self._acoustic.items():
            if len(acoustics) < self.min_count:
                continue
            a = np.stack(acoustics)
            spread = _mean_pairwise_distance(a, rng=self.rng)
            spreads.append(spread)
            out[word] = WordStats(word=word, count=self._counts[word],
                                  acoustic_spread=spread,
                                  examples=self._sentences.get(word, []))

        if not out:
            return out

        # Normalise spread against the corpus itself: "varies more than a
        # typical word" is the question, and an absolute threshold would
        # depend on the acoustic features' scale.
        median = float(np.median(spreads)) or 1e-6
        for word, st in out.items():
            a = np.stack(self._acoustic[word])
            c = np.stack(self._context[word])
            st.acoustic_spread = st.acoustic_spread / median

            if st.acoustic_spread < 1.0 or _mean_pairwise_distance(a, rng=self.rng) < self.min_spread:
                # Said the same way every time: nothing to disambiguate,
                # whatever its spelling looks like.
                st.ambiguity = 0.0
                continue

            labels, separation = _two_means(a, rng=self.rng)
            st.n_clusters = 2 if separation > 0 else 1
            st.context_predictivity = _predictivity(c, labels, rng=self.rng)

            # Both halves must hold. Acoustic variation that context cannot
            # predict is noise, and context that predicts nothing acoustic is
            # irrelevant to pronunciation.
            #
            # The acoustic half is the *cluster separation* -- how far apart
            # the two pronunciations are -- rather than spread above the
            # corpus median. Spread-above-median turned out to be a poor
            # scale: a genuine homograph with perfectly predicted clusters
            # scored 0.15, because even a clearly bimodal word sits only just
            # above a median that already includes ordinary prosodic
            # variation. Separation measures the thing being claimed
            # (two distinct pronunciations) directly, and on the same 0-1
            # cosine scale as everything else here.
            st.separation = separation
            st.ambiguity = float(
                np.clip(separation / self.separation_scale, 0.0, 1.0)
                * st.context_predictivity
            )
        return out


# --------------------------------------------------------------------------
# the table the training run consumes
# --------------------------------------------------------------------------
class AmbiguityTable:
    """Word -> ambiguity score, loaded from a mined JSON file.

    Deliberately a plain mapping with a default of 0. A word that was never
    mined (too rare, or unseen at inference) is treated as ordinary rather
    than as an error: the score only relaxes a compute penalty, so being
    wrong about a rare word costs a little efficiency, never correctness.
    """

    def __init__(self, scores: dict[str, float] | None = None, default: float = 0.0):
        self.scores = dict(scores or {})
        self.default = float(default)

    @classmethod
    def load(cls, path: str | Path) -> "AmbiguityTable":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        words = data.get("words", data)
        scores = {
            w: (v["ambiguity"] if isinstance(v, dict) else float(v))
            for w, v in words.items()
        }
        return cls(scores)

    @classmethod
    def from_stats(cls, stats: dict[str, WordStats]) -> "AmbiguityTable":
        return cls({w: s.ambiguity for w, s in stats.items()})

    def save(self, path: str | Path, stats: dict[str, WordStats] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "words": ({w: s.to_dict() for w, s in stats.items()} if stats
                      else dict(self.scores)),
            "n_words": len(stats or self.scores),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def score(self, word: str) -> float:
        return self.scores.get(word, self.default)

    def profile(self, words: list[str]) -> list[float]:
        return [self.score(w) for w in words]

    def __len__(self) -> int:
        return len(self.scores)

    def top(self, n: int = 30) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: -kv[1])[:n]


# --------------------------------------------------------------------------
# text-only bootstrap, for before the aligner is trained
# --------------------------------------------------------------------------
class ContextualAmbiguity:
    """Ambiguity from the language model alone -- no audio, no alignment.

    This exists so the first training run has a difficulty signal before any
    acoustic evidence is available. The measure is the spread of a word's
    *contextual* LM vectors across the corpus: a word whose meaning shifts
    with context (`علم` as flag vs science) gets vectors that spread out,
    while a word with one stable sense does not.

    It is weaker evidence than the acoustic version, because a spread of
    meaning does not always imply a spread of *pronunciation* -- English
    `bank` has two meanings and one pronunciation, and Arabic has the same
    pattern. It is used only to bootstrap; ``AmbiguityMiner`` supersedes it
    once durations exist.
    """

    def __init__(self, min_count: int = 6, max_occurrences: int = 40,
                 separation_scale: float = 0.6, seed: int = 0):
        self.min_count = int(min_count)
        self.max_occurrences = int(max_occurrences)
        # Centred LM vectors separate further than mel descriptors do, so the
        # saturation point is higher here than in the acoustic miner.
        self.separation_scale = float(separation_scale)
        self.rng = np.random.default_rng(seed)
        self._context: dict[str, list[np.ndarray]] = defaultdict(list)
        self._counts: dict[str, int] = defaultdict(int)

    def add(self, word: str, context: np.ndarray) -> None:
        self._counts[word] += 1
        if len(self._context[word]) < self.max_occurrences:
            self._context[word].append(np.asarray(context, dtype=np.float32))

    def finalise(self) -> dict[str, WordStats]:
        """Score by how much better the split is than one on shuffled data.

        Two earlier attempts were both wrong, and the reason is worth keeping
        because it is easy to walk into again:

        *Spread above the corpus median* conflated "two clean senses" with
        "diffuse noise" -- both are large spreads -- and needed hundreds of
        word types before the median meant anything.

        *Raw cluster separation* is worse, and fails silently. In 768
        dimensions random points are nearly orthogonal, so after centring,
        2-means finds a separation of ~1.999 on a single structureless
        Gaussian blob, at every sample size tested up to n=80. Separation
        alone measures the dimensionality, not the structure.

        So the split is scored against its own null: the same clustering run
        on the same vectors with their dimensions independently shuffled,
        which destroys any real covariance while preserving the marginals.
        A word with two genuine senses beats that null; a word with one sense
        does not.

        This remains the weaker of the two measurements, and it is a
        bootstrap: `AmbiguityMiner` supersedes it as soon as the aligner can
        supply acoustic evidence, which is the thing actually being claimed.
        """
        out: dict[str, WordStats] = {}
        for word, vecs in self._context.items():
            if len(vecs) < self.min_count:
                continue
            x = _centre(np.stack(vecs))
            labels, separation = _two_means(x, rng=self.rng)

            # Within-cluster tightness relative to the null: how much of the
            # variance the split actually explains.
            observed = _split_quality(x, labels)
            null = float(np.mean([
                _split_quality(*_shuffled_split(x, self.rng)) for _ in range(8)
            ]))
            gain = max(0.0, observed - null) / max(1.0 - null, 1e-6)

            if len(set(labels.tolist())) < 2:
                balance = 0.0
            else:
                minority = min((labels == c).mean() for c in (0, 1))
                balance = float(minority / 0.5)

            st = WordStats(word=word, count=self._counts[word],
                           separation=separation)
            st.n_clusters = 2 if gain > 0.1 else 1
            st.context_predictivity = gain
            st.ambiguity = float(np.clip(gain, 0.0, 1.0) * balance)
            out[word] = st
        return out


def _split_quality(x: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of total variance explained by a two-way split (0-1)."""
    if len(set(labels.tolist())) < 2:
        return 0.0
    total = float(((x - x.mean(0)) ** 2).sum())
    if total <= 0:
        return 0.0
    within = 0.0
    for c in (0, 1):
        sel = labels == c
        if sel.any():
            within += float(((x[sel] - x[sel].mean(0)) ** 2).sum())
    return float(np.clip(1.0 - within / total, 0.0, 1.0))


def _shuffled_split(x: np.ndarray, rng: np.random.Generator):
    """The same clustering on dimension-shuffled data: the null hypothesis.

    Shuffling each dimension independently keeps every marginal distribution
    intact while destroying the covariance that a real sense distinction
    would create. Whatever separation 2-means finds here is an artefact of
    dimensionality and sample size, and is exactly what must be subtracted.
    """
    y = np.stack([rng.permutation(col) for col in x.T], axis=1)
    labels, _ = _two_means(y, rng=rng)
    return y, labels
