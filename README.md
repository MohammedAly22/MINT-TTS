# MINT-TTS

**Minimal Inference Needed for Text-to-Speech**

*Speech synthesis reformulated as a resource-allocation problem.*

---

Most TTS systems spend the same computation on every token of every sentence.
That is obviously wrong.

*"Hello, how are you doing?"* is trivial — high-frequency words, one obvious
prosody, no ambiguity. *"The record is broken by the record broker."* is not:
the same spelling carries two pronunciations and only the syntax decides which.
And length is not the answer either — *"I went to the store and I bought some
milk and some bread and some eggs"* is long and completely easy.

MINT-TTS learns **how much computation each token and each acoustic frame
actually needs**, under a quality budget you choose at inference time:

$$\text{Text} \;\rightarrow\; \text{Information requirements} \;\rightarrow\; \text{Minimum compute} \;\rightarrow\; \text{Audio}$$

The learned quantity is $C^{*}(x, q, h)$: the least computation that still
reaches quality $q$ for utterance $x$ on a device with capability $h$.

New to speech synthesis? **[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)**
explains the whole system from scratch and defines every term in the logs —
probe, C\*, MCD, mcd/chance, ponder, depth spread.

Then read [`docs/HYPOTHESIS.md`](docs/HYPOTHESIS.md): it states the claim and,
just as importantly, what would falsify it.

---

## Quickstart

```bash
git clone https://github.com/MohammedAly22/MINT-TTS.git && cd MINT-TTS
pip install -r requirements.txt

# see how each text frontend handles homographs (no data or model needed)
python scripts/inspect_frontend.py

# run the test suite
python -m pytest -q
```

Then train on LJSpeech:

```bash
# 1. data (2.6 GB)
curl -L -o ljs.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf ljs.tar.bz2 -C data/
python scripts/prepare_dataset.py --dataset ljspeech --root data/LJSpeech-1.1

# 2. features + cached phonemisation (~10 min with 4 workers)
python scripts/preprocess.py --config configs/exp2_token.yaml --workers 4

# 3. the headline experiment
python scripts/train.py --config configs/exp2_token.yaml

# 4. watch it think
tensorboard --logdir runs
```

Colab: [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb)
runs all of the above, TensorBoard included, on a free GPU.

---

## Architecture

```
                            TEXT
                              |
              +---------------v----------------+
              |  Normaliser                    |  numbers, currency, dates,
              |  num2words + rule pipeline     |  emails, URLs, phones,
              +---------------+----------------+  addresses, acronyms
                              |
              +---------------v----------------+
              |  Frontend  char | ipa | arpabet|  espeak-ng or g2p_en
              +---------------+----------------+
                              |  tokens + word map
              +---------------v----------------+
              |  Embedding + Conv prenet       |  local mixing (dense)
              +---------------+----------------+
                              |
        ======================v=========================
        |  LINGUISTIC STACK - adaptive depth per TOKEN  |
        |                                               |
        |      +--------------+     router sees         |
        |   -->| shared block |--->  (state, q, h)      |
        |   |  +------+-------+          |              |
        |   |         |          halt?   |              |
        |   +---------+<-------- no -----+              |
        ======================+=========================
                              |  c1 c2 ... cT  <- complexity heatmap
              +---------------v----------------+
              |  Aligner (train) -> durations  |  forward-sum + MAS
              |  Duration / pitch / energy     |
              +---------------+----------------+
                              |
                      length regulator
                              |
        ======================v=========================
        |  ACOUSTIC STACK - adaptive depth per FRAME    |
        ======================+=========================
                              |
                    Linear + Conv postnet
                              |
                      MEL SPECTROGRAM
                              |
                  FIXED HiFi-GAN vocoder
                              |
                            AUDIO
```

### What makes it adaptive

Both stacks are the same class, `AdaptiveStack`, with three routing modes:

| `routing` | Halting decision | Used by |
|---|---|---|
| `fixed` | none — every position takes every step | the dense baseline |
| `sentence` | one decision per utterance | Experiment 1 |
| `token` | one decision per token/frame (ACT) | Experiments 2–5 |

Because the baseline and the adaptive model are **the same code with a
different flag**, the comparison is a controlled experiment rather than two
codebases that happen to be benchmarked together.

Halting follows Adaptive Computation Time. At step $n$ the router emits a
halting probability per position; the output is the convex combination of
intermediate states weighted by the halting distribution. Two numbers come out,
and they are not the same:

$$\text{ponder}_t = n^{\text{updates}}_t + r_t \qquad\qquad c_t = \frac{n^{\text{updates}}_t}{N}$$

`ponder` is the differentiable objective the compute penalty acts on; $c_t$ is
the executed depth — what the FLOP counter and every heatmap report.

### Depth, not experts

The router does not ask *"which expert owns this token?"* (MoE). It asks
*"how many times must this representation be transformed before it is good
enough?"* With `share_weights: true` one block is re-applied, so extra depth
costs **no extra parameters** — depth becomes a pure inference-time knob.

### The budget is an input

The router additionally receives $(q, h)$ through a small MLP producing a
halting-logit bias and FiLM parameters. Both heads are zero-initialised, so an
untrained model behaves like plain ACT and the conditioning is *learned*, not
imposed. During training $q \sim U[0,1]$, $h \sim U[0.3,1]$ and the compute
penalty is scaled by $(1-q)$ — so one checkpoint spans the whole quality/compute
curve instead of a single operating point.

```python
syn("The record is broken by the record broker.", quality=0.3)   # cheap
syn("The record is broken by the record broker.", quality=0.95)  # careful
```

### Where the savings actually come from

Training uses a dense masked path (no speedup — that is fine, the claim is
about inference). Inference uses `forward_active`: at each step only the
still-running positions are gathered and pushed through the block, while halted
positions keep frozen keys/values in a cache so attention still sees the whole
sequence.

`tests/test_adaptive.py` asserts the two paths are **numerically identical**
(~1e-6). If they ever drift, every inference-time measurement would be of a
different model than the one that was trained.

Two subtleties that are easy to get wrong and are handled explicitly:

* **The final step forces a halt.** Otherwise a position that never crosses the
  threshold has `ponder == N` exactly — a constant, with zero gradient — and no
  compute penalty could ever move it.
* **Shared weights make halting free; independent weights do not.** A cached key
  is only valid if every step uses the same projection. With per-step weights,
  keys must be re-projected for all readable positions, which the FLOP counter
  tracks separately as `kv_token_steps`.

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## The text frontend, and the homograph question

Three interchangeable frontends, because **which one you use changes what the
model has to learn**:

| `text.input_type` | Backend | Homographs |
|---|---|---|
| `char` | none | untouched — the model gets the ambiguity intact |
| `ipa` | espeak-ng (`phonemizer` + `espeakng-loader`, no system install) | resolved by espeak's own rules, which are largely context-free |
| `arpabet` | `g2p_en` (CMUdict + POS tagger + homograph list) | *attempts* contextual disambiguation |

`python scripts/inspect_frontend.py` measures this on ten minimal pairs. The
result is worth knowing before you choose:

```
'read'   A: I read a book yesterday.      B: I will read a book tomorrow.
     ipa [same   ]  A=r i: d           B=r i: d
 arpabet [DIFFERS]  A=R IY1 D          B=R EH1 D      <- varies, but backwards

'lead'   A: He will lead the team.        B: The pipe is made of lead.
     ipa [same   ]  A=l i: d           B=l i: d
 arpabet [same   ]  A=L EH1 D          B=L EH1 D      <- wrong for the verb
```

**Neither phonemiser is an oracle.** espeak is context-free on most of these
pairs; g2p_en varies more often but sometimes in the wrong direction. So the
recommendation is not "use espeak" or "use g2p_en" — it is:

> Treat the frontend as an **experimental variable**, not a settled choice.

Three ready-made configs do exactly that, identical except for the input
representation: `configs/frontend_char.yaml`, `frontend_ipa.yaml`,
`frontend_arpabet.yaml`. The default is `ipa` — phonetic enough to train well,
context-free enough that the ambiguity survives for the model to spend compute
on. If `probe/contrast` rises under `char`/`ipa` but not under `arpabet`, that
is direct evidence the router is doing disambiguation work rather than
inheriting it from the frontend.

### Normalisation

An ordered rule pipeline built on `num2words`. Order matters — currency before
bare numbers, dates before ordinals, phone numbers before digit groups:

```
"Dr. Smith paid $1,250.75 on 3/15/2024."
  -> "doctor smith paid one thousand two hundred fifty dollars and seventy five
      cents on march fifteenth twenty twenty four."

"Call +1 (555) 123-4567 or email a.smith@mit.edu."
  -> "call plus one, five five five, one two three, four five six seven or
      email a dot smith at mit dot ee dee you."

"15 Oak St., Apt. 4B; take Oak Dr. north."
  -> "fifteen oak street, apartment four bee; take oak drive north."
```

Covers numbers, ordinals, years, currency, percentages, temperatures, units,
dates, times, phone numbers, emails, URLs, street addresses, unit designators,
titles, general abbreviations and acronyms (`FBI` spelled out, `NASA` kept as a
word). `TextNormalizer(...).trace(text)` shows the output after every step when
a rule misfires.

---

## Monitoring

Every `log.probe_every` steps the trainer runs a fixed probe set and logs:

| Panel | What it tells you |
|---|---|
| `probe/*/token_complexity` | per-token compute heatmap, labelled with the real tokens |
| `probe/*/word_complexity` | aggregated per word — where `record` vs `the` shows up |
| `probe/*/halting` | halting probability at every step (steps × tokens) |
| `probe/*/frame_complexity` | per-frame acoustic compute |
| `probe/compute_by_group` | easy vs homograph vs long-easy, side by side |
| **`probe/contrast`** | one number: compute on ambiguous words minus the rest |
| **`probe/length_corr`** | correlation of compute with length — near 1.0 means the model cheated |
| `train/*/alignment_soft`, `alignment_hard` | aligner health; check this first when a run misbehaves |
| `align/entropy_ratio` | **check first**: ~1.0 means the aligner is at chance and nothing downstream is meaningful |
| `val/mcd_vs_chance` | ≥ 1.0 means the output carries no utterance-specific information |
| `val/wer`, `val/cer`, `val/quality_score` | quality |
| `compute/*`, `train/flops_saving` | what the compute penalty is doing |

Figures render with matplotlib by default — straight into TensorBoard, no
external binary. Set `log.figure_backend: plotly` for interactive versions
(hovering a cell reads `token='r' word='record' depth=6.0/8`), which is ideal
with W&B; its TensorBoard path additionally needs kaleido and a Chrome install.

Details: [`docs/MONITORING.md`](docs/MONITORING.md).

---

## Datasets

| Corpus | Speakers | Hours | Prepare |
|---|---|---|---|
| LJSpeech | 1 | 24 | `--dataset ljspeech --root data/LJSpeech-1.1` |
| VCTK | 110 | 44 | `--dataset vctk --root data/VCTK-Corpus-0.92` |
| LibriTTS | 2400+ | up to 585 | `--dataset libritts --root data/LibriTTS --subsets train-clean-100` |
| LibriSpeech | 2400+ | up to 960 | `--dataset librispeech --root data/LibriSpeech` |

```bash
python scripts/prepare_dataset.py --dataset vctk --root data/VCTK-Corpus-0.92
python scripts/preprocess.py --config configs/vctk_token.yaml --workers 4
python scripts/train.py      --config configs/vctk_token.yaml
```

Multi-speaker is wired end to end: `model.n_speakers: auto` reads the real
count from the preprocessed corpus, so it cannot silently disagree with the
data. `--speaker-disjoint` switches to held-out speakers when you want the
zero-shot question instead.

Your own corpus works through any of four manifest formats (pipe filelist, CSV
with emotion labels, JSONL, or the templates in [`filelists/`](filelists/)) —
see [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md). Only audio and a transcript
are required: durations are learned by the internal aligner, and the *compute*
labels are generated later by `scripts/compute_curve.py`, not annotated by hand.

---

## Inference

```python
from mint_tts.inference.synthesize import Synthesizer

syn = Synthesizer.from_checkpoint("runs/exp2_token/checkpoints/best.pt")
res = syn("The record is broken by the record broker.", quality=0.9)
res.save("out.wav")

print(res.summary())
for word, c in zip(res.encoded.words, res.word_complexity):
    print(f"{word:>10s}  {c:.3f}")   # where the compute went
```

```bash
python scripts/synthesize.py --checkpoint runs/exp2_token/checkpoints/best.pt \
    --text "I read a book yesterday." --quality 0.9

# the same sentence across the whole budget: C*(x, q)
python scripts/synthesize.py --checkpoint ... --text "..." --sweep-quality
```

---

## Repository layout

```
mint_tts/
├── config.py            YAML configs with _base_ inheritance + CLI overrides
├── text/
│   ├── normalizer.py    ordered rule pipeline (num2words, dates, emails, ...)
│   ├── phonemes.py      char / espeak-ng IPA / g2p_en ARPAbet backends
│   └── tokenizer.py     tokenisation + word map + persisted symbol table
├── data/                mel/pitch/energy extraction, manifests, dataset
├── modules/
│   ├── adaptive.py      <- ACT routing, budget conditioning, gathered fast path
│   ├── transformer.py   blocks with a dense path and an active-subset path
│   ├── aligner.py       forward-sum aligner + monotonic alignment search
│   └── variance.py      duration/pitch/energy predictors, length regulator
├── models/              AdaptiveTTS, HiFi-GAN / Griffin-Lim vocoders
├── losses/              reconstruction losses + the compute-allocation objective
├── evaluation/          MCD, WER/CER, MOS, per-utterance compute curves
├── benchmarks/          latency / RTF / FLOPs / memory measurement
├── training/            trainer, complexity probe, figure logging
└── inference/           Synthesizer API
configs/                 base.yaml + one file per experiment and dataset
scripts/                 prepare, preprocess, train, synthesize, benchmark,
                         evaluate, compute_curve, inspect_frontend
docs/                    hypothesis, architecture, data format, experiments, monitoring
tests/                   72 tests, ~6 s
```

---

## Experiments

Run in order; each is a stop/go decision, not a checklist.

| Config | Question |
|---|---|
| `exp0_dense.yaml` | quality ceiling and FLOP reference |
| `exp0_dense_shallow.yaml` | the cost floor the adaptive model should approach |
| `exp1_sentence.yaml` | does sentence-level adaptivity pay off at all? |
| **`experiment_char.yaml`** | **the live experiment**: character input, so homographs actually reach the model |
| `experiment_char_unified.yaml` | the same with an adaptive decoder, where 75% of the compute is |
| `exp2_token.yaml` | per-token allocation on phoneme input |
| `exp2_token_matched.yaml` | the same, widened to match the baseline's parameter count |
| `exp3_independent.yaml` | is weight sharing as good as per-step weights? |
| `exp4_acoustic.yaml` | per-frame allocation in the acoustic decoder |
| `exp5_unified.yaml` | both, with a hardware budget: full $C^{*}(x,q,h)$ |
| `frontend_{char,ipa,arpabet}.yaml` | does the frontend do the disambiguation, or the model? |
| `vctk_token.yaml` / `libritts_token.yaml` | does the policy transfer across speakers and scale? |
| `scale_dense_115m.yaml` / `scale_token_113m.yaml` | does the saving hold at ~115M parameters? |

Decision rules and tuning notes: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## Honest notes

* **The vocoder is frozen and shared** by every experiment; otherwise a quality
  delta could come from the vocoder rather than the acoustic model. Griffin-Lim
  is the default so the repo runs with zero downloads — it is intelligible but
  not publication quality. Fetch HiFi-GAN with `scripts/download_vocoder.py`
  and set `vocoder.name: hifigan` before reporting anything.
* **`mos_proxy` is not MOS.** It is a transparent function of MCD and CER,
  useful for ranking checkpoints. Use `eval.mos_backend: utmos` for a real MOS
  predictor, and human listeners for the final claim.
* **FLOPs are not speed.** `scripts/benchmark.py` reports latency, RTF, peak
  memory *and* FLOPs, because a saving that does not show up in wall-clock on a
  CPU or a GTX 1660 Ti is not a result.
* **Adaptive gains are inference-only.** Training still pays full dense cost.
* **No quality results are claimed yet.** What has been verified is that the
  pipeline runs end to end and that each piece does what it says. The
  hypothesis itself is open — that is the point of the repository.

Status and next steps: [`ROADMAP.md`](ROADMAP.md).

---

## Requirements

Python 3.9+, PyTorch 2.1+. A single mid-range GPU is enough for LJSpeech;
CPU-only works for inference and benchmarking. espeak-ng arrives through pip
(`espeakng-loader`) — no system package needed, which is what makes the IPA
frontend work on a bare Colab runtime.

## License

MIT.
