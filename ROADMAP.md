# Roadmap and status

Legend: `[x]` done and verified · `[~]` implemented, not yet validated on real
data · `[ ]` not started.

## Phase 0 — instrument (this repo)

- [x] Config system with `_base_` inheritance and CLI overrides
- [x] Text frontend: three interchangeable backends (char / espeak-ng IPA / g2p_en ARPAbet)
- [x] espeak-ng shipped through pip (`espeakng-loader`) so no system package is needed
- [x] Powerful normaliser: numbers, currency, dates, times, phones, emails, URLs,
      addresses, units, titles, acronyms (num2words-backed, order-sensitive pipeline)
- [x] Corpus-derived symbol tables persisted to `symbols.json`; training freezes the vocab
- [x] `scripts/inspect_frontend.py` measures homograph handling per backend
- [x] Word-level provenance carried through preprocessing for the heatmaps
- [x] Mel/pitch/energy extraction matching the HiFi-GAN LJSpeech recipe
- [x] Manifest formats: pipe filelist, CSV (emotion), JSONL, multi-speaker
- [x] Internal forward-sum aligner + monotonic alignment search (no MFA needed)
- [x] Adaptive stack with `fixed` / `sentence` / `token` routing behind one class
- [x] Shared and independent block variants
- [x] Budget conditioning on `(q, h)`, zero-initialised so it is learned not imposed
- [x] Gathered inference path with a KV cache — **numerically equal** to the training path (`tests/test_adaptive.py`)
- [x] Correct KV accounting for independent blocks (they cannot reuse cached keys)
- [x] Exact analytic FLOP counter driven by the executed token-steps
- [x] Compute-allocation loss with warmup, budget scaling and a hardware-cap hinge
- [x] Plotly figures: token/word complexity heatmaps, halting matrices, alignments, mel
- [x] Batched kaleido rasterisation so TensorBoard figures cost ~0.6 s each, not ~2.7 s
- [x] TensorBoard + W&B logging (W&B keeps plotly interactive)
- [x] `probe/contrast` and `probe/length_corr` scalars (the falsification metrics)
- [x] Per-utterance compute curves and generated `C*` labels
- [x] `c_star` distillation hook (`loss.compute.c_star_weight`)
- [x] Latency / RTF / memory / FLOP benchmark across devices and budgets
- [x] Evaluation script producing the experiment matrix
- [x] Griffin-Lim fallback vocoder so the repo runs with zero downloads
- [x] Multi-speaker end to end: VCTK / LibriTTS / LibriSpeech collectors,
      templates, configs, and `model.n_speakers: auto`
- [x] 72 unit tests covering routing equivalence, alignment, FLOPs, text and configs
- [ ] HiFi-GAN weights fetched and verified (needs a manual download — see `scripts/download_vocoder.py`)

## Fixed after the first LJSpeech run (2026-09-06)

The first real run exposed several problems that only appear at scale. All are
fixed; the run itself was diagnostic, not wasted.

- [x] **Alignment was coupled to routing.** Aligner keys came from the encoder
      output, which changes as the router learns. When routing collapsed around
      step 2.8k the forming alignment was destroyed. Keys now come from the
      token embedding, as in RAD-TTS/FastPitch.
- [x] **The router ran unconstrained during warmup** and collapsed to a
      constant depth (3.0 for every token, every sentence, every quality
      budget) on reconstruction loss alone. Routing is now held at full depth
      until the compute penalty engages (`router_start_step`).
- [x] **Aligner scale.** With a fixed temperature of 5e-4 against LayerNormed
      features the attention logits spanned ~0.01. Now a per-channel normalised
      distance with a learned scale: ~2x faster alignment convergence.
- [x] **`loss/forwardsum` weight 2.0** made it ~77% of the total loss; now 1.0.
- [x] **No IMAGES tab at all.** Colab preinstalls plotly 5.24 while pip pulls
      kaleido 1.x, which needs plotly >= 6.1.1 — static export failed silently.
      Versions pinned, plus a startup check that says so at step 0.
- [x] **Alignment health as scalars** (`align/entropy_ratio`,
      `align/diagonality`, `align/hard_agreement`) so a broken aligner is
      visible even when figures are unavailable.
- [x] **`val/quality_score` was pinned at 0.** Its thresholds assumed 4-8 dB
      literature MCD; this repo's DCT-of-log-mel MCD runs ~0-50. Quality is now
      anchored to `val/mcd_chance`, measured per dataset, and
      `val/mcd_vs_chance` >= 1.0 flags "no utterance-specific information".
- [x] **Preprocessing 28 min -> ~2 min.** Pitch extraction was 95% of the time
      (40 ms/utterance); replaced with a vectorised FFT autocorrelation (~15x
      faster, optional pyworld backend) and worker threads pinned to 1.
- [x] **Confusing target audio.** `audio_target` was the ground-truth mel put
      through Griffin-Lim. Now logged as `audio_target_vocoded` alongside
      `audio_target_original` (the untouched file), so the vocoder's ceiling is
      distinguishable from the model's error.
- [x] LR scheduler no longer advances on an AMP-skipped step.

## Phase 1 — test the hypothesis on LJSpeech (24 h, single speaker)

- [ ] **E0** Dense baseline `exp0_dense` trained to convergence
- [ ] **E0b** Shallow dense baseline `exp0_dense_shallow`
- [ ] **E1** Sentence-level adaptive `exp1_sentence`
- [ ] **E2** Token-level adaptive `exp2_token`
- [ ] **E2b** Token-level adaptive, parameter-matched `exp2_token_matched`
- [ ] **E3** Shared vs independent at matched parameters
- [ ] **E4** Acoustic (per-frame) routing
- [ ] **E5** Unified routing with hardware budget
- [ ] **F**  Frontend ablation: `frontend_char` vs `frontend_ipa` vs `frontend_arpabet`
- [ ] Compute curves on the test split; report mean `C*` and its spread
- [ ] CPU + GTX 1660 Ti benchmark table
- [ ] Human MOS on a 100-utterance subset (the only claim that needs listeners)

**Go/no-go:** E2 must beat E0 on quality-per-FLOP *and* show
`probe/contrast > 0` with `probe/length_corr` clearly below 1. If compute turns
out to be a pure function of length, the interesting claim is dead and the
honest move is to say so.

## Phase 2 — scale and generalise

- [ ] VCTK (multi-speaker, ~44 h): does allocation transfer across speakers and accents?
- [ ] LibriTTS (~585 h): does the effect survive scale, or was it a small-data artefact?
- [ ] Parameter scaling 10M → 31M → 115M → 200M: does the *relative* saving grow or shrink?
      (`scale_dense_115m.yaml` / `scale_token_113m.yaml` are the matched 115M pair)
- [ ] Emotion/style conditioning using the CSV template (embeddings already wired in)
- [ ] Phoneme-input vs character-input ablation — with a POS-aware G2P the homograph is
      resolved *before* the model sees it, so character mode is the harder and more
      interesting setting
- [ ] Router-target distillation: train with `c_star_weight > 0` from Phase 1 curves

## Phase 3 — vocoder and deployment

- [ ] Swap in and compare vocoders: HiFi-GAN v1 / v3, Vocos, BigVGAN
- [ ] Adaptive vocoder (Phase 2 of the original plan): frame-level routing inside the vocoder
- [ ] ONNX / TorchScript export of the gathered path
- [ ] int8 dynamic quantisation on CPU, measured not assumed
- [ ] Streaming / chunked inference
- [ ] Hardware-aware routing validated across ≥3 real devices (CPU, 1660 Ti, A100):
      does the same checkpoint pick genuinely different depths per device?

## Phase 4 — write-up

- [ ] Reproduce every table from a single `scripts/evaluate.py` invocation
- [ ] Release checkpoints and the compute-curve dataset
- [ ] Paper: the resource-allocation formulation, not "a fast TTS"

## Known limitations (as of now)

* Adaptive gains are only realised at inference; training still costs the full
  dense computation. Making training itself adaptive is open work.
* The gathered path pays a gather/scatter overhead, so the wall-clock saving
  trails the FLOP saving — most visible at small `d_model` and short sequences.
* Batched inference realises savings only when batch members halt together;
  batch size 1 (the low-end-device case) is where the effect is cleanest.
* `mos_proxy` is a proxy. UTMOS is a predictor. Neither is MOS.
* Pitch extraction uses torchaudio's detector; `pyworld` would be better and is
  listed as an optional dependency but is not wired in yet.
* Only English. The multilingual case (`lang` already exists in the manifest
  format) would need a different symbol table and G2P.
