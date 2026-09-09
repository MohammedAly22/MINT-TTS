# How MINT-TTS works, from A to Z

Written for someone new to speech synthesis. No prior TTS background assumed.
Every term that appears in the logs or the notebook is defined here.

---

## 1. What the machine is trying to do

You give it a sentence. It gives you a `.wav`. In between:

<div align="center">
<img src="../assets/pipeline.svg" alt="text to audio pipeline" width="820"/>
</div>


```
"The record is broken."
        |
   [1] normalise      "the record is broken."
        |
   [2] tokenise        t h e _ r e c o r d _ ...
        |
   [3] encode          a vector per character
        |
   [4] durations       how many frames does each character last?
        |
   [5] decode          a mel spectrogram
        |
   [6] vocoder         a waveform
```

**Frame.** Audio is cut into overlapping slices. Ours advance by 256 samples at
22050 Hz, so one frame ≈ 11.6 ms. A 2-second clip is about 172 frames.

**Mel spectrogram.** A picture of sound: 80 rows (frequency bands, spaced the
way human hearing is) × one column per frame. Bright = energy at that pitch and
moment. The model predicts this picture, not raw audio — pictures are far easier
to predict than 22050 numbers per second.

**Vocoder.** Turns the picture back into sound. We use **HiFi-GAN**, a small
neural network trained for exactly this. The fallback, **Griffin-Lim**, is a
classical algorithm that guesses the missing phase information; it is
intelligible but sounds metallic. The vocoder is *frozen* — identical in every
experiment — so quality differences between runs are always the model's doing.

---

<div align="center">
<img src="../assets/architecture.svg" alt="MINT-TTS architecture" width="1000"/>
</div>

## 2. The pieces, in order

### [1] Normalisation

Written text is not speakable. `$1,250.75` has no pronunciation until someone
decides it is "one thousand two hundred fifty dollars and seventy five cents".
The normaliser handles numbers, currency, dates, times, phone numbers, emails,
URLs, addresses, abbreviations and acronyms, in a fixed order (currency before
plain numbers, dates before ordinals, and so on).

### [2] Tokenisation and the frontend

Text becomes a sequence of symbols the model can look up. Which symbols is a
choice with real consequences:

| frontend | the model sees | homographs |
|---|---|---|
| `char` | letters: `r e c o r d` | **unresolved** — the model must work it out |
| `ipa` | espeak's phonemes: `ɹ ˈɛ k ɚ d` | espeak already chose, mostly ignoring context |
| `arpabet` | g2p_en's phonemes: `R EH1 K ER0 D` | attempts context, sometimes wrongly |

**Phoneme.** The smallest unit of pronunciation. English has ~40. "cat" = three
phonemes: /k/ /æ/ /t/.

**Homograph.** One spelling, two pronunciations, decided by context: *"I **read**
yesterday"* /red/ vs *"I will **read** tomorrow"* /reed/.

**G2P (grapheme-to-phoneme).** Letters → sounds. Hard in English because
spelling is irregular.

We train on `char` **on purpose**. With `ipa`, espeak resolves the ambiguity
before the model ever sees it — and then there is nothing for the model to spend
extra thought on. We measured exactly that, and it produced a null result.

### [3] The encoder — and where this project differs

A **transformer block** is the standard building block: it lets every position
look at every other position (*attention*) and then transforms each position on
its own (*feed-forward*). Stacking N blocks = "depth N".

Normally depth is fixed: 8 blocks means every token goes through all 8.
**MINT-TTS re-uses one shared block and decides per token how many times to
apply it.** "the" might go round twice; "record" might go round seven times.

**Why depth might matter.** Each pass is one round of information exchange
across the sentence. Deciding which "record" you meant requires evidence from
neighbouring words — so a token needs enough passes for that evidence to reach
it. "the" needs none.

### [4] Durations and the aligner

To turn 40 characters into 200 frames, the model must know how long each
character lasts. Nobody labels that by hand. The **aligner** learns it from the
audio during training:

- **Soft alignment** — a heat map: how strongly does frame 57 belong to
  character 12?
- **MAS (Monotonic Alignment Search)** — forces a sensible path through that
  map: never go backwards, never skip a character. Gives an integer duration per
  character.
- **Forward-sum loss** — trains the map by summing over *all* valid paths.
- **Beta-binomial prior** — a gentle nudge that characters early in the text
  belong to frames early in the audio.

At inference there is no audio, so a **duration predictor** (trained against
MAS output) supplies the numbers. Then the **length regulator** simply repeats
each character's vector that many times.

### [5] Decoder and postnet

The frame-rate vectors go through another stack, then a linear layer produces
the 80-row mel, and a small convolutional **postnet** adds a refinement pass.

The model also predicts **pitch** (F0, how high the voice is — measured in Hz)
and **energy** (loudness), which give it explicit control over prosody.

---

## 3. Adaptive computation, precisely

### ACT — Adaptive Computation Time

At each pass, a small network (the **router**) outputs a *halting probability*
for every token. These accumulate; once a token's total crosses ~1.0, it stops
and keeps its current vector. Everyone else continues.

Two numbers come out, and they are **not** the same:

| term | meaning |
|---|---|
| **`n_updates`** | how many passes a token actually took — a whole number. This drives the FLOP count and every heatmap. |
| **`ponder`** | `n_updates` + a fractional remainder. A smooth, differentiable stand-in used by the loss, because you cannot take a gradient through "how many times did the loop run". |

`ponder` can slightly exceed the maximum depth. That is normal, and it is why
the two are reported separately.

**Complexity** in the heatmaps = `n_updates / max_steps`, between 0 and 1.

### Why the router is frozen at first

With no pressure, the router drifts to whatever is convenient for reconstruction
and settles on one constant value for every token — and in one run it dragged
the alignment down with it. So routing is held at full depth until the compute
penalty switches on (`router_start_step`). Before then, `enc_depth` sitting at 8
is correct behaviour, not a stall.

### The budget: q and h

At inference you pass a **quality budget** `q` ∈ [0,1] and a **hardware budget**
`h` ∈ [0,1]. The router sees both. During training they are sampled randomly and
the penalty is scaled by (1−q), so one checkpoint learns the whole trade-off
curve rather than a single operating point.

Written compactly: the model learns **C\*(x, q, h)** — the least computation
that still reaches quality `q` for utterance `x` on a device of capability `h`.

---

## 4. What is being optimised

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{duration}} + \mathcal{L}_{\text{pitch}} + \mathcal{L}_{\text{energy}} + \mathcal{L}_{\text{align}}}_{\text{reconstruction: sound correct}} + \underbrace{\lambda(q)\cdot C}_{\text{spend less}}$$

- **`loss/mel`, `loss/mel_post`** — average error between predicted and true mel,
  before and after the postnet.
- **`loss/duration`** — error in predicted character lengths (in log space).
- **`loss/pitch`, `loss/energy`** — prosody prediction error.
- **`loss/forwardsum`** — the alignment objective. **The one to watch early.**
- **`loss/bin`** — sharpens the soft alignment towards the hard one. Starts late.
- **`C`** — normalised compute actually used, from `ponder`.
- **`λ(q)`** — the price of computation. Zero during warmup, then ramps to
  `lambda_max`, and is scaled by (1−q) so a high quality request is charged less.

In words: **say it correctly, using as little computation as you can get away
with** — where "as little" depends on the quality you asked for.

---

## 5. Every metric in the logs

### Alignment — check these first

| metric | means | good |
|---|---|---|
| `align/entropy_ratio` | how *spread out* the soft alignment is. 1.0 = every frame equally attached to every character, i.e. pure guessing | falls from ~0.65 to below 0.4 |
| `align/hard_agreement` | how much probability mass sits on the MAS path | rises, 0.2 → 0.8 |
| `align/diagonality` | does the path track the diagonal | stays high |
| `loss/forwardsum` | the alignment loss | falls steadily |

If `entropy_ratio` stays near 1.0, **stop**. Durations will be nonsense and
every number below it is meaningless.

### Quality

**MCD (Mel Cepstral Distortion)** — the distance between predicted and true
audio. 0 = identical, larger = worse.

> Ours is computed from log-mels with a DCT and no time-warping, so **it is not
> comparable to published MCD** (which sits at 4–8). On our scale, 0 is perfect
> and ~50 is meaningless output.

Because that scale is our own, it is always reported against a measured
reference:

| metric | means |
|---|---|
| `val/mcd_chance` | the MCD between two **unrelated** utterances, measured on *your* validation set at startup. The "no information" level. |
| **`val/mcd_vs_chance`** | `val/mcd ÷ val/mcd_chance`. **≥ 1.0 = the audio is no closer to your sentence than a random different sentence would be.** Below ~0.4 is healthy. |
| `val/quality_score` (`Q`) | 0 to 1: 1 = identical to target, 0 = chance |
| `val/wer`, `val/cer` | word / character error rate when an ASR system transcribes the generated audio. Measures intelligibility. Off by default. |
| `mos_proxy` | a stand-in for a human listening score, derived from MCD. **Not** MOS. Real MOS needs human listeners. |

### Compute

| metric | means |
|---|---|
| `compute/lambda` | current price of computation. 0 during warmup |
| `compute/encoder_norm` | fraction of maximum depth used, 1.0 = full |
| `train/encoder_depth` | mean passes per token |
| **`compute/encoder_depth_spread`** | standard deviation of depth *across tokens within a sentence*. **This is the experiment.** ~0 means the router picked one constant — no allocation, whatever the mean says |
| `compute/encoder_frac_at_min/max` | fraction of tokens stopping immediately / running to the end |
| `train/flops_saving` | arithmetic skipped versus running every token to full depth |
| `rtf` | **real-time factor**: seconds of compute per second of audio. 0.05 = 20× faster than real time |

### The probes

A **probe** is a fixed set of sentences the model synthesises every N steps, so
you can watch the *same* examples evolve. They are never trained on. They are
grouped:

- `easy` — trivial sentence
- `homograph` — contains an ambiguous word
- `tongue_twister` — hard phonetics, easy meaning
- `normalisation` — numbers, dates, emails
- `long_easy` — long but simple. **A control**: if this costs as much as the
  homographs, compute is tracking *length*, not difficulty.

| metric | means |
|---|---|
| **`probe/contrast`** | mean compute on ambiguous words **minus** mean compute on ordinary words. The hypothesis as one number: should be > 0 |
| **`probe/length_corr`** | correlation between sentence length and compute. Near 1.0 means the router only learned "longer = more", which is trivially true and scientifically empty |
| `probe/compute_by_group` | the groups side by side |

### The homograph probe

The complexity heatmaps show *where* compute went. They do not show whether it
*changed the pronunciation*. This probe synthesises both contexts of a minimal
pair, cuts out the mel frames of the ambiguous word in each, and compares them.

| metric | means |
|---|---|
| `homograph/<word>/divergence` | how differently the word was rendered across the two contexts |
| `homograph/<word>/control_divergence` | the same measure for ordinary words present in both sentences — the baseline amount of variation |
| **`homograph/divergence_ratio`** | divergence ÷ control. **> 1 means ambiguous words vary more than ordinary ones.** ≈ 1 means the model says "record" identically both times |
| `homograph/compute_advantage` | extra compute the ambiguous word received |

Differing is necessary but not sufficient — it could differ *wrongly*. That is
why both renditions are logged as audio: listen to them.

### The compute curve

`scripts/compute_curve.py` forces the model to run at each depth from 1 to 8 and
measures quality each time. From that curve:

**C\* ("C-star")** — the *cheapest* depth that still reaches 98% of the best
quality that utterance ever achieves. "What this sentence actually needs."

| metric | means |
|---|---|
| `c_star_encoder_mean` | average minimum encoder depth, as a fraction |
| **`c_star_encoder_unique`** | how many **distinct** C\* values appeared across utterances. **If this is 1, every sentence needs the same compute — there is nothing to allocate, and no router can invent headroom that does not exist.** |
| `c_star_share_at_mode` | fraction of utterances sharing the single most common C\* |
| `corr(length, C*)` | if near 1.0, C\* is just sentence length again |

This is the cheapest possible check on whether the whole premise holds, and it
runs on a mid-training checkpoint in under a minute.

---

## 6. The assumptions, stated plainly

1. **Not all utterances need the same computation.** Testable: `c_star_*_unique`.
2. **Difficulty is not length.** Testable: `probe/length_corr`, the `long_easy` group.
3. **Ambiguity is a form of difficulty that costs computation.** This is the
   least certain one. Resolving "record" is *one bit of information*, and bits
   are not FLOPs — there is no derivation from "ambiguous" to "needs more
   layers". The mechanistic version we can defend: depth = rounds of information
   exchange, and disambiguating evidence sits some distance away in the sentence.
4. **A model can learn to predict its own requirement before doing the work.**
   That is what the router is.
5. **Savings must survive contact with real hardware.** FLOPs are not seconds,
   which is why `rtf` and wall-clock latency are reported alongside.

## 7. What would prove it — and what would kill it

**Support:** the adaptive model matches the deep dense baseline on
`val/mcd_vs_chance` while using materially less compute, with
`compute/encoder_depth_spread` clearly above 0, `probe/contrast` above 0, and
`homograph/divergence_ratio` above 1.

**Failure modes, each with its own detector:**

| what happened | you would see |
|---|---|
| nothing to allocate | `c_star_encoder_unique` = 1 |
| router collapsed to a constant | `depth_spread` ≈ 0 |
| it only learned sentence length | `probe/length_corr` ≈ 1 |
| compute moved but pronunciation did not | `divergence_ratio` ≈ 1 |
| saving does not reach the clock | FLOPs drop, `rtf` does not |

Every one of those is a real result worth reporting. A hypothesis you cannot
kill is not a hypothesis.
