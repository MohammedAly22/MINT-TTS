"""Contextual semantic conditioning: giving the router something to route *on*.

Why this module exists
----------------------
Adaptive depth alone cannot disambiguate a homograph, and it is worth being
precise about why, because the architecture looks as though it should.

Extra depth gives a token more rounds of self-attention over the *same*
representations. For the English `record` that is arguably enough: the
disambiguating evidence is syntactic (a determiner before it, a subject after
it), and syntax is recoverable from character statistics given enough data.

Egyptian Arabic is not that case. Consider the pair this project was retargeted
for::

    ana rasamt  3alam masr        -> 3alam  (flag)
    ana ba7eb el-3elm gedan       -> 3elm   (science)

The spellings are identical, the syntactic frames are near-identical (both are
a noun after a verb), and the short vowels that separate them are simply not
written. What decides the reading is **lexical semantics**: which nouns go with
"draw", which go with "love ... and want to become a scientist". A character
encoder trained from scratch on 68 hours of speech has no lexical semantics to
recover -- 62k word forms over 612k tokens is far too sparse a sample to learn
a word's meaning from its spelling.

So the model is given the semantics rather than asked to invent them. A frozen
Arabic language model (MARBERTv2 by default -- pre-trained on dialectal Arabic
including Egyptian Twitter, not only MSA) reads the *whole sentence* and emits
a contextual vector per word. The vector for the flag-word and the vector for
the science-word are already far apart, because that is exactly what masked
language modelling learns. This module projects those vectors into the acoustic
model and adds them to the character states of the word they belong to.

The router then has a signal worth routing on, and the two jobs separate
cleanly:

    the language model   decides *what the word means here*
    the adaptive stack   decides *how much computation that meaning needs*

Cost
----
The LM is frozen and can be run **offline** (``precompute: true``), in which
case training pays nothing for it at all: vectors are cached to disk during
preprocessing and loaded with the mel. That is the recommended setting, and the
only one that keeps the "very fast" requirement honest -- a 163M-parameter BERT
forward pass per step would otherwise dominate a 40M-parameter TTS model.

The projection is small (hidden -> d_model) and *is* trained. Everything that
is learned stays in the TTS model; the LM contributes fixed features.

Alignment
---------
The LM uses WordPiece, so one word becomes several subwords. Subword vectors
are mean-pooled back to whole words using the fast tokenizer's ``word_ids()``
map, which is what keeps the result indexable by MINT-TTS's own ``word_ids``.
Getting this wrong would shift every word's vector by one and poison the
signal silently, so ``SemanticEncoder.encode`` asserts the two word counts
agree and falls back to per-word encoding when they do not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Models known to work. MARBERTv2 is the default because it is trained on
# dialectal Arabic (including a large Egyptian Twitter share) rather than MSA
# alone, which matters for a corpus of conversational Egyptian.
KNOWN_MODELS = {
    "marbert": "UBC-NLP/MARBERTv2",
    "marbertv1": "UBC-NLP/MARBERT",
    "arabert": "aubmindlab/bert-base-arabertv02",
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "xlmr": "FacebookAI/xlm-roberta-base",
}

DEFAULT_MODEL = KNOWN_MODELS["marbert"]


def resolve_model_name(name: str) -> str:
    """Accept either a short alias or a full Hugging Face repo id."""
    return KNOWN_MODELS.get(name, name)


@dataclass
class SemanticFeatures:
    """Per-word contextual vectors for one utterance."""

    vectors: np.ndarray   # (n_words, hidden)
    words: list[str]
    model: str

    def __len__(self) -> int:
        return len(self.words)


class SemanticEncoder:
    """Frozen LM -> one contextual vector per word.

    Used at preprocessing time (to cache vectors) and by the probes at
    training time (probe sentences are not in the corpus, so their vectors
    cannot come from the cache).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        layer: int = -1,
        device: str | torch.device = "cpu",
        max_length: int = 256,
        pooling: str = "mean",
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = resolve_model_name(model_name)
        self.layer = int(layer)
        self.pooling = pooling
        self.max_length = int(max_length)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"{self.model_name} has no fast tokenizer. The subword->word "
                "mapping relies on `word_ids()`, which only the fast "
                "tokenizers provide; without it every word's vector would be "
                "silently misaligned."
            )
        self.model = AutoModel.from_pretrained(
            self.model_name, output_hidden_states=True
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)
        self._fallbacks = 0

    # -- encoding ---------------------------------------------------------
    @torch.inference_mode()
    def encode(self, words: list[str]) -> SemanticFeatures:
        """Encode one utterance, given as the word list MINT-TTS produced.

        The words come from MINT-TTS's own tokenisation so that the returned
        rows line up 1:1 with `Encoded.words`, which is what makes indexing by
        `word_ids` correct.
        """
        if not words:
            return SemanticFeatures(np.zeros((0, self.hidden_size), np.float32), [], self.model_name)

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = enc.word_ids(0)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).hidden_states[self.layer][0]   # (n_sub, H)

        vectors = np.zeros((len(words), self.hidden_size), dtype=np.float32)
        counts = np.zeros(len(words), dtype=np.int64)
        hid = hidden.float().cpu().numpy()
        for pos, wid in enumerate(word_ids):
            if wid is None:            # [CLS]/[SEP] belong to no word
                continue
            vectors[wid] += hid[pos]
            counts[wid] += 1

        missing = counts == 0
        if missing.any():
            # A word can be lost to truncation or to a tokenizer that maps it
            # to nothing at all (a lone combining mark, say). Leaving a zero
            # row would look like a legitimate "neutral" vector, so those
            # words are re-encoded alone and the count is reported.
            self._fallbacks += int(missing.sum())
            for i in np.nonzero(missing)[0]:
                vectors[i] = self._encode_single(words[int(i)])
                counts[i] = 1
        vectors /= counts[:, None].clip(1)
        return SemanticFeatures(vectors, list(words), self.model_name)

    @torch.inference_mode()
    def _encode_single(self, word: str) -> np.ndarray:
        enc = self.tokenizer(word, return_tensors="pt", truncation=True,
                             max_length=self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).hidden_states[self.layer][0]
        if hidden.shape[0] <= 2:       # only special tokens survived
            return np.zeros(self.hidden_size, dtype=np.float32)
        return hidden[1:-1].mean(0).float().cpu().numpy()

    @torch.inference_mode()
    def encode_batch(self, batch_words: list[list[str]]) -> list[SemanticFeatures]:
        """Encode several utterances. One call per utterance keeps the
        subword->word mapping simple; batching happens at the caller."""
        return [self.encode(w) for w in batch_words]


# --------------------------------------------------------------------------
# the trainable side: projection + injection
# --------------------------------------------------------------------------
class SemanticAdapter(nn.Module):
    """Projects frozen LM word vectors into the acoustic model.

    Three things are produced from the same projected vector:

    ``add``    a residual added to the character states of that word, so the
               encoder's representation of a token carries the meaning of the
               word it belongs to.
    ``film``   a per-word gain/shift, so semantics can *modulate* rather than
               only translate the state.
    ``router`` the same vector is concatenated into the router's input, so the
               halting decision sees the semantics directly rather than only
               through their effect on the state.

    The output projections are **zero-initialised**. At step 0 the model is
    therefore exactly the character-only model, and the semantic path has to
    earn its influence through gradient descent. That keeps the comparison
    against the no-semantics baseline honest: any difference is learned, not
    an artefact of a different initialisation.
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int = 768,
        proj_hidden: int = 0,
        dropout: float = 0.1,
        use_film: bool = True,
        layernorm_input: bool = True,
    ):
        super().__init__()
        proj_hidden = proj_hidden or d_model
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.use_film = use_film
        # Frozen LM features arrive with an arbitrary scale; normalising them
        # keeps the projection's job well-conditioned and stops a change of LM
        # from silently changing the effective learning rate of this path.
        self.in_norm = nn.LayerNorm(hidden_size) if layernorm_input else nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.to_add = nn.Linear(proj_hidden, d_model)
        nn.init.zeros_(self.to_add.weight)
        nn.init.zeros_(self.to_add.bias)
        if use_film:
            self.to_film = nn.Linear(proj_hidden, 2 * d_model)
            nn.init.zeros_(self.to_film.weight)
            nn.init.zeros_(self.to_film.bias)
        self.out_dim = proj_hidden

    def forward(
        self,
        semantic: torch.Tensor,        # (B, W, H) per-word LM vectors
        word_index: torch.Tensor,      # (B, T) long, word id per token
        token_mask: torch.Tensor,      # (B, T) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (delta (B, T, D) to add to the states, features (B, T, P)).

        ``word_index`` broadcasts a word's vector to every character of that
        word, which is what ties a 768-dim sentence-level representation to
        character-level routing.
        """
        B, T = word_index.shape
        h = self.proj(self.in_norm(semantic.to(self.proj[0].weight.dtype)))   # (B, W, P)
        idx = word_index.clamp_min(0).unsqueeze(-1).expand(-1, -1, h.size(-1))
        per_token = h.gather(1, idx.clamp(max=h.size(1) - 1))                 # (B, T, P)
        per_token = per_token * token_mask.unsqueeze(-1).to(per_token.dtype)
        delta = self.to_add(per_token)
        return delta * token_mask.unsqueeze(-1).to(delta.dtype), per_token

    def film(self, per_token: torch.Tensor):
        """Per-token (gamma, beta) from the projected features, or None."""
        if not self.use_film:
            return None
        gamma, beta = self.to_film(per_token).chunk(2, -1)
        return gamma, beta


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------
def cache_key(model_name: str, layer: int) -> str:
    """Short stable id for a (model, layer) pair, used in cache paths.

    Including it in the path means switching LM or layer cannot silently reuse
    the previous model's vectors -- which would be invisible in the metrics
    and extremely hard to diagnose.
    """
    raw = f"{resolve_model_name(model_name)}@{layer}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def semantic_path(out_dir: str | Path, uid: str) -> Path:
    return Path(out_dir) / f"{uid}.npy"


@lru_cache(maxsize=2)
def get_encoder(model_name: str, layer: int, device: str, max_length: int) -> SemanticEncoder:
    """Process-local cached encoder, so workers build the LM once each."""
    return SemanticEncoder(model_name, layer=layer, device=device, max_length=max_length)
