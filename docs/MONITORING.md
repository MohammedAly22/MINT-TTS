# Monitoring a run

```bash
tensorboard --logdir runs
# or mirror everything to W&B (renders plotly natively, no rasterisation cost):
python scripts/train.py --config configs/exp2_token.yaml \
    --override "log.backends=['tensorboard','wandb']" log.wandb_project=mint-tts
```

## Before anything else: can you see images?

If TensorBoard shows no **IMAGES** tab, figure export is broken and you are
flying blind on alignment and the complexity heatmaps. The usual cause is a
plotly/kaleido version mismatch — kaleido 1.x needs plotly ≥ 6.1.1, and some
environments (Colab) preinstall plotly 5.x:

```bash
pip install -U 'plotly>=6.1.1' 'kaleido>=1.0'   # then restart the runtime
```

The trainer checks this at startup and warns loudly. Training is still valid
without it — the `align/*` scalars below cover the critical diagnosis — but
fix it if you can.

## What the model actually produces

Mel spectrograms, not discrete codes: 80-bin log-mel at 22.05 kHz, hop 256,
matching the HiFi-GAN LJSpeech recipe. The vocoder is frozen and shared across
every experiment. `train/*/mel` shows predicted-over-target as a heatmap.

## Phases of a run

| Steps | What is happening | What to expect |
|---|---|---|
| 0 → `router_start_step` | **routing frozen at full depth** | `enc_depth` pinned at `max_steps`, `compute/lambda` = 0, `train/routing_frozen` = 1. Intended: an unpenalised router collapses to a constant depth on reconstruction loss alone, and drags alignment down with it. |
| `warmup_steps` → `+ ramp_steps` | compute penalty ramps in | `compute/lambda` rises, `enc_depth` starts falling, `train/flops_saving` rises |
| after | the actual experiment | depth should *differentiate* across tokens, not merely fall |

## Read the panels in this order

### 1. Alignment — nothing else means anything until this works

| Scalar | Healthy | Broken |
|---|---|---|
| `align/entropy_ratio` | falls from ~0.65 to well below 0.4 and keeps going | stays near **1.0** = attention is uniform, the aligner is at chance and only the prior is doing anything |
| `align/hard_agreement` | rises steadily (0.2 → 0.6+) | flat |
| `align/diagonality` | high; starts high because of the prior | collapses |
| `loss/forwardsum` | decreases steadily | flat, or **rising** |

`train/*/alignment_hard` should become a clean monotonic diagonal. If
`loss/forwardsum` is flat near `log(n_tokens)` (~5–6 for LJSpeech-length
sentences), alignment is at chance — stop and fix that before reading any
complexity number.

Alignment keys come from the token **embedding**, not the encoder output, so
routing changes cannot destabilise alignment. If you change that, expect the
two to fight.

### 2. Is the model producing *this* utterance, or an average one?

The check that catches a run which looks fine but has learned nothing:

| Scalar | Meaning |
|---|---|
| `val/mcd_chance` | MCD between two **unrelated** utterances, measured on your val set, logged once at startup |
| `val/mcd_vs_chance` | `val/mcd ÷ val/mcd_chance`. **≥ 1.0 means the output carries no utterance-specific information** — no better than playing a different sentence |
| `val/quality_score` | 1.0 = identical to target, 0.0 = chance |

A falling `loss/mel` together with `val/mcd_vs_chance` stuck near 1.0 means the
model is converging to the unconditional average spectrum — almost always an
alignment failure, not a capacity problem.

`val/mcd` is computed from log-mels by DCT, without DTW. **It is not comparable
to published MCD** (which sits at 4–8 dB); on this scale 0 is identical and
~50 is chance. That is why it is always reported against `mcd_chance`.

### 3. Compute allocation

| Scalar | Expect |
|---|---|
| `compute/lambda` | 0 during warmup, then ramps to `lambda_max` |
| `compute/combined_norm` | ~1.0, then descends once the penalty engages |
| `train/encoder_depth` | executed steps; should *spread across tokens*, not just fall |
| `train/flops_saving` | rises as depth falls |
| `val/mcd_vs_chance` | should stay flat while compute falls — that is the whole claim |

If quality degrades in lockstep with compute, `lambda_max` is too high.

### 4. The hypothesis

| Scalar | Good | Bad |
|---|---|---|
| `probe/contrast` | rises above 0 and holds | ~0: no linguistic signal in the allocation |
| `probe/length_corr` | clearly below 1 | ~1.0: the router only learned sentence length |
| `probe/compute_by_group` | `homograph` above `easy` / `long_easy` | all equal |
| `probe/*/encoder_compute` | differs across sentences **and** across `q` | identical everywhere: routing has collapsed |

A constant value across every probe sentence *and* every quality budget means
the router has collapsed to a constant policy — the degenerate solution, not a
result.

### 5. Audio

Three tags per example, and the difference between them matters:

| Tag | What it is |
|---|---|
| `train/*/audio_target_original` | the untouched source file |
| `train/*/audio_target_vocoded` | ground-truth mel through **the same vocoder** as the prediction — the ceiling this vocoder can reach |
| `train/*/audio_pred` | the model's mel through that vocoder |

With the default Griffin-Lim vocoder, `audio_target_vocoded` already sounds
rough and phasey while `audio_target_original` is clean. **That gap is the
vocoder, not the model.** Install HiFi-GAN (`scripts/download_vocoder.py`, then
`vocoder.name: hifigan`) before judging quality or reporting anything.

`probe/*/audio` is full inference — no ground-truth durations. Expect silence
or noise early on: it depends on the duration predictor, which depends on
alignment.

### 6. Reading the figures

In W&B they stay interactive: hover a heatmap cell to read
`token='ɹ'  word='record'  depth=6.0/8`. TensorBoard gets rasterised copies.
Standalone scripts write `.html` next to every `.png`.

## Cost of monitoring

Probes are cheap; rasterising figures is not — kaleido needs ~2.7 s for one
figure but ~0.6 s each when batched, so the logger buffers and flushes them in
one pass. Defaults (`probe_every: 2000`, `probe_max_figures: 4`) keep monitoring
near 1% of training time. W&B skips rasterisation entirely.

## Useful overrides

```bash
# see routing engage sooner on a first validation run
--override loss.compute.warmup_steps=4000 loss.compute.ramp_steps=2000

# fast iteration: probe often, no audio
--override log.probe_every=200 log.figure_every=200 log.log_audio=false

# intelligibility tracking on (downloads a wav2vec2 model once)
--override eval.asr_backend=wav2vec2 train.val_every=1000

# watch the whole trade-off curve during training
--override "log.probe_budgets=[0.1,0.3,0.5,0.7,0.9]"
```
