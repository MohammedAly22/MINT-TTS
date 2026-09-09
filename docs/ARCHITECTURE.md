# Architecture

<div align="center">
<img src="../assets/architecture.svg" alt="MINT-TTS architecture" width="1000"/>
</div>

The ASCII sketch the diagram replaces is kept below for terminals.


```
                       TEXT
                         |
                 [char / phoneme]
                         v
              +---------------------+
              | Embedding + ConvNet |   local mixing (non-adaptive)
              +----------+----------+
                         v
              +---------------------+
              |  LINGUISTIC STACK   |   adaptive depth, per TOKEN
              |  shared block x N   |<--- router sees (state, q, h)
              +----------+----------+
                         |  c_1 c_2 ... c_T   <- complexity heatmap
                         v
        +----------------+-----------------+
        | aligner (train) | duration pred. |   pitch / energy predictors
        +----------------+-----------------+
                         v
                 length regulator
                         v
              +---------------------+
              |   ACOUSTIC STACK    |   adaptive depth, per FRAME
              |  shared block x M   |
              +----------+----------+
                         v
                  linear + postnet
                         v
                    MEL SPECTROGRAM
                         v
                 FIXED HiFi-GAN vocoder
                         v
                       AUDIO
```

## The text frontend

```
raw text
   |  TextNormalizer  -- ordered rules: currency before numbers, dates before
   v                     ordinals, phones before digit groups (num2words)
normalised text
   |  Phonemizer      -- char | espeak-ng IPA | g2p_en ARPAbet
   v                     one phone list PER WORD, so the word map survives
tokens + word_ids + words
   |  SymbolTable     -- built from the corpus, saved to symbols.json,
   v                     frozen at train time
token ids
```

Keeping one phone list *per word* is what makes the heatmaps readable: every
token knows which word it came from, so `probe/word_complexity` can say
"`record` cost 6.1 steps" instead of "index 27 cost 6.1 steps".

The symbol table is data-derived rather than hard-coded. IPA inventories depend
on the language and the espeak-ng version, so hand-maintaining a phone list
would rot; instead preprocessing collects every symbol it produced and writes
`symbols.json`, and training loads and freezes it.

## The adaptive stack

`mint_tts/modules/adaptive.py` implements all three routing modes behind
one class, so the dense baseline and the adaptive model differ by a config flag
rather than by codebase:

| `routing` | Halting decision | Experiment |
|---|---|---|
| `fixed` | none — every token takes every step | 0 (baseline) |
| `sentence` | one per utterance | 1 |
| `token` | one per token (ACT) | 2–5 |

Halting follows Adaptive Computation Time: at step $n$ the router emits
$p_n \in (0,1)$ per position, cumulative halting probability is accumulated, and
the output is the convex combination of intermediate states weighted by the
halting distribution. Two quantities come out of this, and they are not the
same number:

$$ \text{ponder}_t = n^{\text{updates}}_t + r_t, \qquad c_t = \frac{n^{\text{updates}}_t}{N} \in (0, 1] $$

`ponder` is the differentiable surrogate the compute penalty acts on; $c_t$ is
the executed depth, which is what the FLOP counter and the heatmaps report.

Two details matter more than they look:

* **The final step forces a halt.** A token that never crosses the threshold
  would have `ponder == N` exactly — a constant, with zero gradient — so the
  router could never be pushed to halt earlier no matter how strong the compute
  penalty. Forcing the halt makes the remainder non-zero and differentiable.
  (It also means `ponder` can reach `N + 1`; the *reported* compute fraction is
  `n_updates / N`, the exact executed depth the FLOP counter is built from.)
* **The dense FLOP reference excludes padding.** Otherwise a ragged batch shows
  a "saving" that is only the padding it never computed.

### Shared vs independent blocks

`share_weights: true` re-applies **one** block (Universal-Transformer style), so
$N$ steps of depth cost $1\times$ the parameters. `share_weights: false` gives
each step its own block. Experiment 3 compares them at matched parameter count;
the interesting outcome is shared blocks matching independent ones, because
depth then becomes a pure inference-time knob.

### Two forward paths

Training uses the differentiable path: every step is computed for all
positions, masked and weighted. There is no speedup — that is fine, the speedup
claim is about inference.

Inference uses `forward_active`: at each step only the *still-running*
positions are gathered and pushed through the block. Halted positions keep
frozen keys/values in a cache so attention still sees the whole sequence while
queries and FFN are computed only for live positions. This is where FLOPs
actually disappear.

One consequence: the position-wise sublayer inside an adaptive stack must be
token-independent, so it is a linear FFN rather than the convolutional FFN of
FastSpeech. Local convolutional mixing lives in the encoder prenet and the
mel postnet, which are dense and cheap.

## Budget conditioning

The router additionally receives $(q, h)$ through a small MLP that produces

* a bias on the halting logit (global "be cheaper / be richer" pressure), and
* FiLM parameters modulating the router's view of the state.

Both heads are zero-initialised, so an untrained model behaves exactly like an
unconditioned ACT model and conditioning is learned rather than imposed.

During training $q \sim U[0,1]$ and $h \sim U[0.3, 1]$ per example, and the
compute penalty is scaled by $(1-q)$. One checkpoint therefore learns the whole
trade-off curve instead of a single operating point.

## Alignment

Durations are learned in-model with a RAD-TTS-style forward-sum aligner plus
monotonic alignment search — no Montreal Forced Aligner, no TextGrids, no
external toolchain. `docs/MONITORING.md` explains how to read the alignment
plots, which are the first thing to check when a run goes wrong.

## Parameter budget

With `d_model=384`, `ff_dim=1536` a block is ~1.8M parameters.

| Config | Encoder | Decoder | Total (approx.) |
|---|---|---|---|
| shared, N=8 / M=6 (`exp2_token`) | 1 block | 1 block | 9.8M |
| independent, N=8 / M=6 (`exp0_dense`, `exp3_independent`) | 8 blocks | 6 blocks | 31.0M |
| shared at `d_model=768` (`exp2_token_matched`) | 1 block | 1 block | 31.2M |

To reach the 100–200M regime, raise `d_model`/`ff_dim` (both variants) rather
than only the depth — and always compare shared vs independent at *matched
parameters*, not matched depth.
