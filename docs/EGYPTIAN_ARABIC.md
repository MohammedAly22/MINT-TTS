# Egyptian Arabic: the homograph experiment

> **Why this corpus.** On LJSpeech the homograph question was hard to ask
> cleanly: English has a few hundred homographs, most of them resolvable from
> syntax, and the `ipa` frontend resolved them before the model ever saw them.
> Undiacritised Arabic makes the same question unavoidable — the short vowels
> that distinguish readings are *systematically* absent from the orthography,
> so every sentence carries some of this ambiguity.

---

## The problem, precisely

Arabic is written without short vowels. The reader supplies them from context.
Three distinct kinds of ambiguity result, and they fail for different reasons:

### Lexical

The same consonant skeleton spells unrelated words.

| written | reading | meaning |
|---|---|---|
| `علم` | `عَلَم` *3alam* | flag |
| `علم` | `عِلْم` *3elm* | science, knowledge |
| `علم` | `عَلَّم` *3allem* | he taught |

> انا رسمت **علم** مصر → *3alam* (flag)
> انا بحب ال**علم** جدا و نفسي ابقا عالم → *3elm* (science)

What decides it is **lexical semantics**: which nouns go with "draw", which go
with "love … and want to become a scientist". Syntax does not help — both are
nouns in the same position.

### Morphological

The same skeleton spells different inflections of one root. Active and passive
are frequently distinguished by vowels alone, so the ambiguity changes *who did
what*:

| written | reading | meaning |
|---|---|---|
| `ضرب` | `ضَرَب` | he hit |
| `ضرب` | `ضُرِب` | he was hit |

### Clitic / agreement — the hardest class

A second-person suffix agrees with the addressee's gender, and that agreement
is carried entirely by an unwritten short vowel:

| written | to a man | to a woman |
|---|---|---|
| `عمرك` | `عُمرَك` *3omrak* | `عُمرِك` *3omrik* |
| `هسيبك` | `هسيبَك` *hasiibak* | `هسيبِك` *hasiibik* |

> **عمرك** فكرت**ي** الراجل بتاع غزل البنات بينفخ الكيس ازاي؟ **هسيبك** تجاوب**ي** و تخمن**ي**

The only evidence that the addressee is a woman is the feminine verb `فكرتي`
— **four words after** `عمرك` — and `تجاوبي` / `تخمني` later still. The
disambiguating signal is non-local, which is precisely why per-token routing
over full-sentence attention is the right shape of solution: a token needs
enough passes for distant evidence to reach it.

---

## Why adaptive depth alone is not enough

It is worth being explicit, because the architecture looks as though it should
suffice.

Extra depth gives a token more rounds of self-attention over the **same**
representations. For English `record` that is arguably enough — the evidence is
syntactic, and syntax is recoverable from character statistics given data.

Egyptian Arabic is not that case. The corpus has 612k word tokens over 62k
unique forms, and the decisive evidence is lexical meaning. **A character
encoder trained from scratch on 68 hours cannot learn what a word means from
its spelling** at that sample size. More depth over meaningless representations
produces more of nothing.

So the model is given the semantics rather than asked to invent them, and the
two jobs are separated:

```
frozen MARBERTv2  ->  what does this word mean here?
adaptive routing  ->  how much computation does that meaning need?
```

---

## The three additions

### 1. `ar_char` frontend

Raw Arabic graphemes. **There is deliberately no Arabic G2P.** Every available
Arabic grapheme-to-phoneme tool needs diacritics to decide short vowels, so
running one on undiacritised text would make it guess — *context-free*,
resolving the ambiguity to a single reading before the model saw it. That is
exactly the failure the English `ipa` frontend already demonstrated.

The normaliser (`mint_tts/text/arabic.py`) handles Arabic-Indic digits,
tatweel, Quranic marks, Arabic punctuation and number verbalisation. Letter
folding is per-flag, because each choice has consequences:

| flag | default | why |
|---|---|---|
| `strip_diacritics` | on | a partially diacritised corpus would leak the answer for some utterances and not others |
| `normalise_alef` | on | the hamza seat is orthographic; Egyptian speech does not distinguish them in most positions |
| `normalise_ya` | on | ya and alef maqsura are used interchangeably in Egyptian writing |
| `normalise_ta_marbuta` | **off** | it carries feminine morphology and surfaces as /t/ in construct state — folding it destroys evidence the agreement cases need |

Latin letters survive the frontend, so code-switched English ("laptop",
"اوكي") reaches the model rather than being deleted.

### 2. Frozen contextual semantics

`mint_tts/modules/semantic.py`. A frozen MARBERTv2 reads the whole sentence and
emits one vector per word. MARBERT is chosen over AraBERT because it is
pre-trained on dialectal Arabic including a large Egyptian Twitter share,
rather than MSA alone.

The `SemanticAdapter` projects those vectors and injects them three ways:

* **added** to the character states of the word they belong to (via `word_ids`),
* **FiLM** gain/shift, so semantics modulate rather than only translate,
* **into the router**, so the halting decision sees meaning directly rather
  than only through its effect on the state.

Two properties matter:

**Zero-initialised.** Every output head starts at zero, so at step 0 the model
is *exactly* the character-only model. Any difference from the no-semantics
control is therefore learned, not an initialisation artefact. `tests/test_arabic.py`
asserts this holds to 1e-6, and separately that the path does escape the zero
init during training — a permanently dead path would defeat the design
silently.

**Precomputed.** With `semantic.precompute: true` the LM runs once during
preprocessing and the vectors are cached to disk. Training never runs BERT: a
163M-parameter forward pass per step would otherwise dominate a ~18M-parameter
acoustic model and make "fast" untrue. The cache path includes a hash of
`(model, layer)`, so switching LM cannot silently reuse the old vectors.

**Subword alignment.** MARBERT uses WordPiece, so one word becomes several
subwords. They are mean-pooled back to whole words using the fast tokenizer's
`word_ids()` map. Getting this wrong would shift every vector by one word and
poison the signal invisibly, so the encoder asserts the counts agree and falls
back to per-word encoding when they do not.

### 3. Difficulty-aware compute

`mint_tts/text/homographs_ar.py` scores each word:

| score | meaning |
|---|---|
| 0.0 | ordinary word |
| 0.5 | ambiguous clitic, nothing in the sentence resolves it |
| 0.7 | ambiguous clitic **and** the sentence carries the resolving cue |
| 1.0 | a listed homograph with two or more readings |

The compute penalty then prices each token by `1 - relief * difficulty`. At
`difficulty_relief: 0.75` an ambiguous token pays 25% of what an ordinary token
pays per step — depth where it is needed becomes affordable, while easy words
stay under full pressure.

A uniform penalty asks the router to make *every* token cheap, which is the
wrong objective. The claim is not that computation is wasteful; it is that
computation should go where the difficulty is.

`difficulty_contrast_weight` adds a hinge requiring hard tokens to run at least
`difficulty_margin` deeper than easy ones — because the relief term alone can
be satisfied by a router that is simply uniformly shallow.

**The lexicon is a prior, not an oracle.** It is hand-built and partial. It is
used for a training-time penalty weight and for building probe sets; the model
never consults it at inference time and must generalise from context. An
unlisted homograph simply gets the default score.

---

## Running it

```bash
python scripts/prepare_egyptian.py --out data/egyptian
python scripts/preprocess.py --config configs/egyptian_homograph.yaml --workers 8
python scripts/train.py      --config configs/egyptian_homograph.yaml
```

| config | what it is |
|---|---|
| `egyptian_homograph.yaml` | the experiment (H200-sized) |
| `egyptian_homograph_small.yaml` | same, for a 24 GB card |
| `egyptian_nosemantic.yaml` | **control**: no MARBERT. Isolates the semantic path |
| `egyptian_dense.yaml` | **control**: no routing. Quality ceiling |

Preprocessing is shared across all four — only training re-runs.

---

## Reading the result

The metrics, in the order they can invalidate each other:

| metric | healthy | meaning |
|---|---|---|
| `align/entropy_ratio` | falls below 0.3 | **check first** — at ~1.0 the aligner is at chance and nothing downstream means anything |
| `val/mcd_vs_chance` | well below 1.0 | at ≥1.0 the audio carries no information about which sentence was asked for |
| `semantic/delta_norm` | rises above 0 | the LM path escaped its zero init. If it stays at 0, any homograph result comes from somewhere else |
| `compute/difficulty_contrast` | > 0 | **the claim**: ambiguous tokens get more depth than ordinary ones |
| `compute/encoder_depth_spread` | > 0 | at ~0 the router collapsed to a constant, whatever the mean says |
| `homograph/divergence_ratio` | > 1 | ambiguous words are *rendered differently* across contexts |
| `probe/length_corr` | not ~1 | at ~1 the router only learned "longer = more" |

### The controls that make it interpretable

A positive number on the main run means little alone.

* **Tongue twisters must stay cheap.** They are hard phonetics with easy
  semantics. If they cost as much as the homographs, the router is tracking
  articulatory or character-level difficulty, not ambiguity. That is a real
  finding, but a different claim than this experiment set out to test.
* **`long_easy` must stay cheap.** Otherwise compute tracks length.
* **`egyptian_nosemantic` must be weaker.** If it separates the readings just
  as well, the character encoder was sufficient and MARBERT is dead weight
  worth deleting.

### And then listen

`homograph/divergence_ratio` says two renderings **differ**. It cannot say they
differ *correctly* — a model could pronounce both readings wrongly but
differently and score well. Only an Egyptian speaker can settle it, which is
why the probe writes both renditions to TensorBoard every `probe_every` steps.

Every automatic metric here is necessary; none is sufficient.

---

## Forward-looking pieces (wired, inert on this run)

Enabled now so that adding them later is a data change rather than an
architecture change that invalidates every checkpoint trained before it.

**Zero-shot voice cloning.** `model.reference_encoder` maps a reference mel to
a speaker vector through strided convs and attentive pooling, replacing the
lookup-table embedding that can only reproduce voices seen in training. On this
single-speaker corpus it learns one voice, but the conditioning pathway, the
shapes, the checkpoint layout and the inference API are already the
multi-speaker ones.

The reference is always a *different* utterance by the same speaker — encoding
the target would hand the model the mel it is being asked to predict.
`speaker_dropout` replaces it with a learned "unknown" token some of the time,
so synthesis without any reference keeps working.

**Multilingual / code-switching.** `model.n_languages > 1` adds a language
embedding to the same conditioning path. The frontend already preserves Latin
script through Arabic normalisation, so code-switched text survives today.
