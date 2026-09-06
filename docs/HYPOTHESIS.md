# Research hypothesis

*MINT-TTS: Minimal Inference Needed for Text-to-Speech.*

> A speech generator does not require uniform computation across utterances or
> linguistic regions. A learned controller can estimate the minimum computation
> necessary to preserve a target level of speech quality, enabling substantial
> reductions in inference cost without perceptible degradation.

## The reformulation

Conventional TTS learns

$$ \text{Text} \rightarrow \text{Audio} $$

This project learns

$$ \text{Text} \rightarrow \text{Information requirements} \rightarrow \text{Minimum compute} \rightarrow \text{Audio} $$

with compute treated as a **latent variable of the generation process**, not a
fixed property of the architecture. The quantity the model learns is

$$ C^{*}(x, q, h) $$

* $x$ — the utterance
* $q$ — the requested quality, in $[0, 1]$
* $h$ — the compute available on the target device, in $[0, 1]$

## Why this is not "just early exit"

Early exit, MoE and distillation all reduce cost at a *fixed* operating point.
Three things separate this formulation:

1. **The budget is an input.** One checkpoint spans the whole quality/compute
   curve because $q$ (and $h$) condition the router at inference time. There is
   no retraining per operating point.
2. **The label is generated, not assumed.** `scripts/compute_curve.py` measures
   $Q_x(C)$ for every utterance and derives $C^{*}$ empirically. That turns
   "how much compute does this sentence need?" from a hyperparameter into a
   measurable quantity with a dataset attached.
3. **Depth, not experts.** The router chooses *how many times* to transform a
   representation, reusing one shared block, so extra depth is free in
   parameters. The question becomes "how much processing does this token need"
   rather than "which specialist owns this token".

## What would falsify it

The hypothesis is designed to be killable on 24 hours of LJSpeech, cheaply:

| Observation | Verdict |
|---|---|
| Adaptive model matches deep dense quality at materially lower FLOPs **and** lower CPU latency | supported |
| Compute correlates ~1.0 with utterance length (`probe/length_corr`) | **failed** — the router learned "longer = more", which is trivial |
| `probe/contrast` stays ~0: homographs cost the same as function words | **failed** — no linguistic signal in the allocation |
| Adaptive model needs full depth everywhere to match quality | **failed** — no headroom exists at this scale |
| FLOP savings do not translate into wall-clock savings on CPU / 1660 Ti | **partially failed** — theoretically interesting, practically inert |

The last row is the reason `scripts/benchmark.py` reports latency, RTF, memory
*and* FLOPs. A model with 1B parameters and 100M active is not automatically
faster than a 300M dense model, and this repo refuses to pretend otherwise.

## Success criterion for Phase 1

> A ≤200M-parameter TTS model that reaches approximately the quality of a
> substantially larger dense baseline while using dramatically fewer FLOPs on
> easy utterances, and staying faster than real time on CPU and on a GTX
> 1660 Ti.

Concretely, on LJSpeech, the target is: adaptive model within noise of the deep
dense baseline on MCD/WER at `q = 0.9`, with mean normalised compute ≤ 0.5 and
CPU RTF ≤ that of the *shallow* dense baseline.
