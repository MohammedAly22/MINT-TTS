# Experiment plan

Run these in order. Each one is a decision point: if it fails, stop and think,
do not proceed to the next.

```
Experiment 0   Dense baselines (deep + shallow)
      |
Experiment 1   Sentence-level adaptive        -> is there any headroom at all?
      |
Experiment 2   Token-level adaptive           -> the headline claim
      |
Experiment 3   Shared vs independent blocks   -> is depth free in parameters?
      |
Experiment 4   Acoustic (frame-level) routing
      |
Experiment 5   Unified linguistic + acoustic, with a hardware budget
```

## Experiment 0 — dense baselines

```bash
python scripts/train.py --config configs/exp0_dense.yaml
python scripts/train.py --config configs/exp0_dense_shallow.yaml
```

Two rows, not one. The deep baseline sets the quality ceiling; the shallow one
sets the cost the adaptive model is trying to approach. Without both, "adaptive
is cheaper" has no scale.

Routing is `fixed` and `loss.compute.enabled: false`, so this is the same code
running as an ordinary non-autoregressive TTS model.

## Experiment 1 — sentence-level adaptive

```bash
python scripts/train.py --config configs/exp1_sentence.yaml
```

One halting decision per utterance. The cheapest form of the hypothesis, and
the cheapest way to kill it: if a per-utterance budget cannot beat the dense
baseline on quality-per-FLOP, per-token routing will not save it.

**Watch:** `compute/combined_norm` should drop below 1.0 once `compute/lambda`
starts ramping (after `loss.compute.warmup_steps`) while `val/mcd` stays flat.
If quality collapses the moment the penalty turns on, lower `lambda_max`.

## Experiment 2 — token-level adaptive (headline)

```bash
python scripts/train.py --config configs/exp2_token.yaml
```

**Watch, in priority order:**

1. `train/*/alignment_hard` — a clean monotonic diagonal. If alignment is
   broken, everything downstream is noise. Fix this before reading any
   complexity plot.
2. `probe/contrast` — mean compute on ambiguous words minus the rest. It should
   climb above zero and stay there. This is the claim, reduced to one number.
3. `probe/length_corr` — if this sits near 1.0, the router learned "longer
   sentences get more compute", which is trivially true and scientifically
   empty. The `long_easy` probe sentence exists to expose exactly this.
4. `probe/*/word_complexity` — eyeball it. Both instances of `record` bright,
   `the` and `by` dark, is the picture we are looking for.
5. `val/quality_score` vs `val/compute_norm` — the trade-off itself.

### Parameter matching

`exp2_token` shares one block and lands at ~9.8M parameters against
`exp0_dense`'s ~31M. Some of any win therefore comes from size, not routing.
`configs/exp2_token_matched.yaml` widens the shared model to ~31.2M so the two
differ only in *how* compute is allocated. Train both: the small one shows what
weight sharing buys, the matched one isolates the routing claim.

```bash
python scripts/train.py --config configs/exp2_token_matched.yaml
```

## Experiment 3 — shared vs independent blocks

```bash
python scripts/train.py --config configs/exp3_independent.yaml
```

Compare against `exp2_token` **at matched parameter count**, not matched depth:
raise `d_model`/`ff_dim` on the shared variant until the totals agree
(`runs/*/model_summary.json` reports them). Comparing a 12M shared model with a
35M independent one and declaring a winner would be meaningless.

There is also a systematic inference asymmetry worth reporting: with shared
weights a halted position's cached keys stay valid forever, so halting is
completely free; with independent weights every step owns different projection
weights, so keys must be recomputed for all readable positions. Shared blocks
are therefore *structurally* cheaper under adaptive depth, not just smaller.
`kv_token_steps` in the FLOP report quantifies this.

## Experiment 4 — acoustic routing

```bash
python scripts/train.py --config configs/exp4_acoustic.yaml
```

Encoder fixed, decoder adaptive: compute is allocated per mel frame. Silence
and steady vowels should cost far less than transients and onsets.

**Watch:** `probe/*/frame_complexity` against the mel plot. Peaks should line
up with consonant onsets, valleys with silence and sustained vowels.

## Experiment 5 — unified, with a hardware budget

```bash
python scripts/train.py --config configs/exp5_unified.yaml
```

Both stacks adaptive, hardware cap active. Then verify the budget knobs
actually do something:

```bash
python scripts/synthesize.py --checkpoint runs/exp5_unified/checkpoints/best.pt \
    --text "The record is broken by the record broker." --sweep-quality
```

Compute should rise monotonically with `q`. If it is flat, the budget head
never learned — check that `q_range` is wide and `lambda_max` is not so small
that the penalty is irrelevant.

## Scaling probe (Phase 2)

`scale_dense_115m.yaml` (~116M, independent) and `scale_token_113m.yaml`
(~114M, shared adaptive) are a parameter-matched pair in the 100-200M regime
the project targets. The question is whether the *relative* saving grows with
scale — a bigger model has more redundancy to skip — or shrinks. Do not run
these before Experiment 2 has produced a clear answer at ~30M.

## Frontend ablation -- who resolves the homograph?

```bash
for f in char ipa arpabet; do
  python scripts/preprocess.py --config configs/frontend_$f.yaml --workers 4
  python scripts/train.py      --config configs/frontend_$f.yaml
done
```

The three configs are identical except for `text.input_type`, so the difference
between the runs is attributable to the input representation alone. Note that
each needs its own `preprocessed_dir`: the vocabularies differ.

Start by looking at what the frontends actually do:

```bash
python scripts/inspect_frontend.py --json outputs/frontend_report.json
```

Measured on ten minimal pairs, espeak-ng is context-free on most of them
("read" is /ri:d/ whether the sentence says *yesterday* or *tomorrow*), while
g2p_en varies more often but sometimes in the wrong direction. Neither is an
oracle, so the interesting comparison is:

* `probe/contrast` **rises under char/ipa** -> the router is spending compute
  where the input is genuinely ambiguous. That is the claim.
* `probe/contrast` **rises only under arpabet** -> the router is reacting to
  something the frontend already decided, not doing the work itself.
* quality is much worse under `char` -> the model cannot solve G2P and
  resource allocation at once at this data scale, which is a real and
  reportable limitation rather than a bug.

## Multi-speaker (Phase 2)

```bash
python scripts/prepare_dataset.py --dataset vctk --root data/VCTK-Corpus-0.92
python scripts/preprocess.py --config configs/vctk_token.yaml --workers 4
python scripts/train.py      --config configs/vctk_token.yaml
```

The question is whether the compute policy is a property of *language* or of
*speaker*. Two checks:

* per-speaker mean compute should not dominate per-token variation. If one
  speaker is uniformly expensive, the router has learned speaker identity, not
  difficulty.
* `probe/contrast` should survive. It is measured on fixed probe sentences, so
  it is directly comparable to the LJSpeech number.

`--speaker-disjoint` holds out whole speakers instead of utterances; that is
the zero-shot-speaker experiment and a different (harder) question.

## Compute curves and generated labels

```bash
python scripts/compute_curve.py \
    --checkpoint runs/exp2_token/checkpoints/best.pt \
    --index data/preprocessed/ljspeech/test.jsonl --limit 300 \
    --write-labels data/preprocessed/ljspeech/train_labelled.jsonl
```

This measures $Q_x(C)$ per utterance and derives $C^{*}$. Two things to read:

* the spread of `C*` — if every utterance needs the same compute, there is
  nothing to allocate and the premise is wrong;
* `corr(length, C*)` — printed at the end, and again the trivial-solution check.

To distil the labels back into the router, point `data.train_index` at the
labelled file and set `loss.compute.c_star_weight: 0.5`.

## Final evaluation

```bash
python scripts/evaluate.py \
    --checkpoints runs/exp0_dense/checkpoints/best.pt \
                  runs/exp0_dense_shallow/checkpoints/best.pt \
                  runs/exp1_sentence/checkpoints/best.pt \
                  runs/exp2_token/checkpoints/best.pt \
    --index data/preprocessed/ljspeech/test.jsonl \
    --asr wav2vec2 --benchmark --benchmark-devices cpu cuda
```

Which produces the table the whole project is about:

```
| run                | routing      | params | Q     | MCD  | compute | FLOPs/utt | WER   | rtf_cpu |
|--------------------|--------------|--------|-------|------|---------|-----------|-------|---------|
| exp0_dense         | fixed/fixed  |  ...   |  ...  | ...  |  1.00   |   ...     | ...   |  ...    |
| exp0_dense_shallow | fixed/fixed  |  ...   |  ...  | ...  |  1.00   |   ...     | ...   |  ...    |
| exp2_token         | token/fixed  |  ...   |  ...  | ...  |  0.4?   |   ...     | ...   |  ...    |
```

The result worth publishing is `exp2_token` matching `exp0_dense` in the Q and
MCD columns while sitting near `exp0_dense_shallow` in the FLOPs and rtf_cpu
columns. Anything less should be reported as such.

## Tuning notes

| Symptom | Likely cause | Fix |
|---|---|---|
| Compute pinned at 1.0 | penalty too weak or still in warmup | raise `lambda_max`, check `compute/lambda` is non-zero |
| Quality collapses when the penalty ramps | penalty too strong or ramped too fast | lower `lambda_max`, lengthen `ramp_steps` |
| Every token gets identical depth | router has no signal | raise `router_hidden`, try `diversity_weight: 0.01` |
| Alignment never forms a diagonal | prior off, or forwardsum weight too low | ensure `data.use_attn_prior: true`, raise `loss.forwardsum` |
| `probe/length_corr` ≈ 1.0 | trivial solution | this is a *result*, report it — then try token-normalised penalties |
| Duration predictor produces 1-frame outputs | untrained, or duration loss too low | train longer; check `loss/duration` is falling |
