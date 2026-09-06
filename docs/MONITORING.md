# Monitoring a run

```bash
tensorboard --logdir runs
# or mirror everything to W&B:
python scripts/train.py --config configs/exp2_token.yaml \
    --override "log.backends=['tensorboard','wandb']" log.wandb_project=mint-tts
```

## Read the panels in this order

### 1. Alignment (`train/*/alignment_hard`, `alignment_soft`)

A clean monotonic diagonal from bottom-left to top-right. Typically appears
within 5–15k steps with the beta-binomial prior enabled.

If it is diffuse or fragmented, **stop reading everything else** — durations
are wrong, so mel loss, complexity and FLOP numbers are all measuring a broken
model. Check `data.use_attn_prior: true`, raise `loss.forwardsum`, and confirm
your transcripts actually match the audio.

### 2. Reconstruction (`loss/mel_post`, `train/*/mel`)

`loss/mel_post` falls steadily. The predicted mel should acquire visible
harmonic structure; a flat, blurry prediction late in training usually means
the aligner never converged (see above).

### 3. Compute (`compute/*`, `train/encoder_depth`, `train/flops_saving`)

* `compute/lambda` is 0 until `loss.compute.warmup_steps`, then ramps. Nothing
  about compute means anything before that point.
* `compute/combined_norm` should start at ~1.0 and descend once the penalty
  engages.
* `val/mcd` should stay roughly flat while it descends. Quality falling in
  lockstep with compute means the penalty is too strong.

### 4. The hypothesis (`probe/*`)

| Scalar | Good | Bad |
|---|---|---|
| `probe/contrast` | rises above 0 and holds | stays ~0: no linguistic signal in the allocation |
| `probe/length_corr` | clearly below 1 | ~1.0: the router only learned sentence length |
| `probe/*/encoder_compute` | differs across groups | identical everywhere: no allocation happening |
| `probe/*/flops_saving` | grows with training | flat at 0 |

The `easy` and `long_easy` probe sentences should end up *cheaper* than the
`homograph` ones. `long_easy` is deliberately long: if it is expensive, compute
is tracking length rather than difficulty.

### 5. The heatmaps

`probe/*/token_complexity` — one row, one cell per token, labelled with the
actual token strings. Bright = expensive.

`probe/*/word_complexity` — the same, aggregated per word. This is the figure
to put in a paper: on *"the record is broken by the record broker"*, both
`record` bars tall, `the`/`by` bars short.

`probe/*/halting` — steps × tokens. Shows *when* each token halted, not just
how deep it went. A vertical stripe means a token was still running long after
its neighbours stopped.

`probe/*/frame_complexity` — per-frame compute across the utterance. Compare it
against `train/*/mel`: peaks at onsets, valleys in silence.

### 6. Quality (`val/*`)

`val/mcd` (lower better), `val/wer` / `val/cer` (needs `eval.asr_backend:
wav2vec2`), `val/mos` or `val/mos_proxy`, and `val/quality_score` — the single
scalar in [0,1] combining them, and the same one used to locate `C*`.

### 7. Reading the figures

Figures are plotly. In W&B they stay interactive: hover a heatmap cell to read
`token='r'  word='record'  depth=6.0/8`, which is far more useful than
squinting at a 60-tick axis. TensorBoard has no plotly renderer, so figures are
rasterised with kaleido and logged as images; the labels are still there, just
not hoverable. Standalone scripts write `.html` next to every `.png`.

## Cost of monitoring

Probes and figures run a full forward pass per probe sentence per budget. With
12 sentences × 3 budgets that is 36 forwards. Defaults (`probe_every: 1000`)
keep it under ~1% of training time; lower it for a quick run, raise it if
figures dominate.

`log.log_audio: true` also runs the vocoder. With Griffin-Lim that is slow —
set `log_audio: false` until you have HiFi-GAN in place.

## Useful overrides

```bash
# fast iteration: probe often, no audio
--override log.probe_every=200 log.figure_every=200 log.log_audio=false

# quality tracking on (downloads a wav2vec2 model once)
--override eval.asr_backend=wav2vec2 train.val_every=1000

# see the trade-off curve during training
--override "log.probe_budgets=[0.1,0.3,0.5,0.7,0.9]"
```
