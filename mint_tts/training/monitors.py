"""Training-time monitors.

`ComplexityProbe` is the scientific instrument of this repo. On a fixed set of
probe sentences it renders, every N steps:

* a per-token complexity heatmap with the token strings as labels,
* the per-step halting matrix,
* per-word aggregated compute,
* the frame-level (acoustic) compute trace,

and it reduces all of that to a handful of scalars you can watch on a curve:

* ``probe/contrast`` -- mean compute on *ambiguous* words minus mean compute on
  the rest. If the hypothesis holds, this rises above zero and stays there.
* ``probe/length_corr`` -- correlation between sentence length and compute. If
  this is ~1.0 the model has merely learned "longer = more", which is exactly
  the trivial solution we need to rule out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..utils import plotting
from ..utils.flops import human


@dataclass
class ProbeSentence:
    text: str
    group: str = "default"
    note: str = ""
    ambiguous_words: list[str] = field(default_factory=list)


def load_probe_sentences(cfg) -> list[ProbeSentence]:
    items = cfg.log.get("test_sentences", [])
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(ProbeSentence(text=it))
        else:
            out.append(ProbeSentence(
                text=it["text"],
                group=it.get("group", "default"),
                note=it.get("note", ""),
                ambiguous_words=list(it.get("ambiguous_words", [])),
            ))
    return out


class ComplexityProbe:
    def __init__(self, cfg, text_processor, device):
        self.cfg = cfg
        self.tp = text_processor
        self.device = device
        self.sentences = load_probe_sentences(cfg)
        self.encoded = [self.tp.encode(s.text) for s in self.sentences]
        self.budgets = list(cfg.log.get("probe_budgets", [0.2, 0.5, 0.9]))
        self.max_figs = int(cfg.log.get("probe_max_figures", 6))
        self.log_audio = bool(cfg.log.get("probe_audio", True))

    @torch.inference_mode()
    def run(self, model, logger, step: int, vocoder=None, hard: bool = True,
            routing_frozen: bool = False) -> dict:
        was_training = model.training
        model.eval()
        scalars: dict[str, float] = {}
        rows = []
        per_sentence_compute: list[float] = []
        lengths: list[int] = []
        amb_vals: list[float] = []
        other_vals: list[float] = []
        group_compute: dict[str, list[float]] = {}

        for si, (sent, enc) in enumerate(zip(self.sentences, self.encoded)):
            tokens = torch.tensor(enc.ids, dtype=torch.long, device=self.device).unsqueeze(0)
            lens = torch.tensor([len(enc.ids)], dtype=torch.long, device=self.device)
            for q in self.budgets:
                budget = torch.tensor([[float(q), 1.0]], device=self.device)
                # Mirror training: while routing is frozen the router is bypassed
                # entirely, so probing it would report a policy the model is not
                # being trained with.
                out = model(tokens, lens, budget=budget, hard=hard,
                            force_full_depth=routing_frozen)
                enc_r, dec_r = out.encoder_router, out.decoder_router
                c_tok = enc_r.complexity[0].float().cpu().numpy()[: len(enc.ids)]
                c_frame = dec_r.complexity[0].float().cpu().numpy()
                n_frames = int(out.mel_mask[0].sum().item())
                c_frame = c_frame[:n_frames]
                flops = model.flops(out)

                tag = f"probe/{si:02d}_{sent.group}/q{q:g}"
                scalars[f"{tag}/encoder_compute"] = float(c_tok.mean())
                scalars[f"{tag}/decoder_compute"] = float(c_frame.mean()) if c_frame.size else 0.0
                scalars[f"{tag}/flops"] = float(flops.total)
                scalars[f"{tag}/flops_saving"] = float(flops.saving)

                if abs(q - self.budgets[-1]) < 1e-9:
                    per_sentence_compute.append(float(c_tok.mean()))
                    group_compute.setdefault(sent.group, []).append(float(c_tok.mean()))
                    lengths.append(len(enc.words))
                    word_c = aggregate_by_word(c_tok, enc.word_ids, len(enc.words))
                    amb = {w.lower() for w in sent.ambiguous_words}
                    for w, val in zip(enc.words, word_c):
                        (amb_vals if w.lower() in amb else other_vals).append(float(val))

                if si < self.max_figs and abs(q - self.budgets[-1]) < 1e-9:
                    logger.log_figure(
                        f"{tag}/token_complexity",
                        plotting.plot_token_complexity(
                            enc.tokens, c_tok, title=sent.text[:90],
                            max_steps=model.encoder.max_steps,
                            words=enc.words, word_ids=enc.word_ids),
                        step,
                    )
                    halting = enc_r.halting_probs[0].float().cpu().numpy()[:, : len(enc.ids)]
                    logger.log_figure(
                        f"{tag}/halting",
                        plotting.plot_halting_matrix(halting, enc.tokens), step)
                    word_c = aggregate_by_word(c_tok, enc.word_ids, len(enc.words))
                    logger.log_figure(
                        f"{tag}/word_complexity",
                        plotting.plot_word_complexity(enc.words, word_c,
                                                      highlight=sent.ambiguous_words), step)
                    if c_frame.size:
                        logger.log_figure(
                            f"{tag}/frame_complexity",
                            plotting.plot_frame_complexity(c_frame), step)
                    if vocoder is not None and self.log_audio:
                        try:
                            wav = vocoder.to_wav(out.mel_post[0, :, :n_frames])
                            logger.log_audio(f"{tag}/audio", wav, step, self.cfg.audio.sample_rate)
                        except Exception:
                            pass

                rows.append([
                    sent.group, sent.text[:60], f"{q:g}", f"{c_tok.mean():.3f}",
                    f"{(c_frame.mean() if c_frame.size else 0):.3f}",
                    human(flops.total), f"{flops.saving * 100:.1f}%",
                ])

        if amb_vals and other_vals:
            contrast = float(np.mean(amb_vals) - np.mean(other_vals))
            scalars["probe/contrast"] = contrast
            scalars["probe/ambiguous_compute"] = float(np.mean(amb_vals))
            scalars["probe/other_compute"] = float(np.mean(other_vals))
        if len(per_sentence_compute) > 2:
            corr = _safe_corr(lengths, per_sentence_compute)
            scalars["probe/length_corr"] = corr if np.isfinite(corr) else 0.0
            logger.log_figure(
                "probe/compute_vs_length",
                plotting.plot_scatter(lengths, per_sentence_compute, "words", "mean compute",
                                      "compute vs sentence length",
                                      labels=[s.text[:48] for s in self.sentences]), step)
        if group_compute:
            logger.log_figure(
                "probe/compute_by_group",
                plotting.plot_group_comparison(
                    {g: float(np.mean(v)) for g, v in group_compute.items()}), step)
            for g, v in group_compute.items():
                scalars[f"probe/group/{g}"] = float(np.mean(v))

        logger.log_table(
            "probe/table",
            ["group", "text", "q", "enc_compute", "dec_compute", "flops", "saving"],
            rows, step,
        )
        scalars["probe/routing_frozen"] = float(routing_frozen)
        logger.log_scalars(scalars, step)
        logger.flush_figures()
        if was_training:
            model.train()
        return scalars


@torch.no_grad()
def alignment_diagnostics(out, batch) -> dict:
    """Scalar health checks for the aligner.

    These exist because alignment is the single thing that, when broken, makes
    every downstream number meaningless -- and because the alignment *images*
    are the first thing to disappear when figure export is unavailable. Watch
    these curves, not just the loss:

    align/entropy_ratio  1.0 means the soft attention is uniform, i.e. the
                         aligner is at chance and only the prior is working.
                         It should fall well below 0.5 and keep going.
    align/diagonality    1.0 means the argmax path tracks the diagonal. It
                         starts high because of the prior, so the number to
                         trust is entropy_ratio.
    align/hard_agreement probability mass the soft attention puts on the hard
                         (MAS) path. Rises as the two agree.
    """
    logs: dict = {}
    if out.attn_soft is None or out.attn_hard is None:
        return logs
    soft = out.attn_soft.float()                       # (B, T_mel, T_text)
    tlen = batch["token_lens"].to(soft.device)
    mlen = batch["mel_lens"].to(soft.device)
    ent, diag, agree = [], [], []
    for i in range(soft.size(0)):
        t, l = int(tlen[i]), int(mlen[i])
        if t < 2 or l < 2:
            continue
        p = soft[i, :l, :t]
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        e = -(p.clamp_min(1e-9).log() * p).sum(-1).mean() / math.log(t)
        ent.append(float(e))
        idx = p.argmax(-1).float()
        ref = torch.linspace(0, t - 1, l, device=soft.device)
        diag.append(float(1.0 - (idx - ref).abs().mean() / t))
        agree.append(float((p * out.attn_hard[i, :l, :t].float()).sum() / l))
    if ent:
        logs["align/entropy_ratio"] = float(np.mean(ent))
        logs["align/diagonality"] = float(np.mean(diag))
        logs["align/hard_agreement"] = float(np.mean(agree))
    return logs


def _safe_corr(x, y) -> float:
    """Pearson r, or nan when a series is constant (instead of a warning)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_by_word(values: np.ndarray, word_ids: list[int], n_words: int) -> np.ndarray:
    """Mean token value per word."""
    sums = np.zeros(max(n_words, 1))
    counts = np.zeros(max(n_words, 1))
    for v, w in zip(values, word_ids):
        if 0 <= w < n_words:
            sums[w] += float(v)
            counts[w] += 1.0
    return sums / np.maximum(counts, 1.0)


@torch.inference_mode()
def log_training_examples(model, batch, out, logger, step: int, vocoder=None,
                          sample_rate: int = 22050, n: int = 2) -> None:
    """Mel / alignment / complexity for a few items from the current batch."""
    n = min(n, out.mel.size(0))
    for i in range(n):
        L = int(batch["mel_lens"][i])
        T = int(batch["token_lens"][i])
        logger.log_figure(
            f"train/{i}/mel",
            plotting.plot_mel(out.mel_post[i, :, :L].float().cpu().numpy(),
                              title=f"pred: {batch['text'][i][:70]}",
                              target=batch["mel"][i, :, :L].float().cpu().numpy()),
            step,
        )
        if out.attn_soft is not None:
            toks = batch["token_strings"][i][:T] if batch["token_strings"] else None
            logger.log_figure(
                f"train/{i}/alignment_soft",
                plotting.plot_alignment(out.attn_soft[i, :L, :T].float().cpu().numpy(),
                                        "soft alignment", toks), step)
            logger.log_figure(
                f"train/{i}/alignment_hard",
                plotting.plot_alignment(out.attn_hard[i, :L, :T].float().cpu().numpy(),
                                        "hard alignment (MAS)", toks), step)
        c_tok = out.encoder_router.complexity[i, :T].float().cpu().numpy()
        toks = batch["token_strings"][i][:T] if batch["token_strings"] else [""] * T
        words = batch["words"][i] if batch.get("words") else None
        word_ids = batch["word_ids"][i][:T] if batch.get("word_ids") else None
        logger.log_figure(
            f"train/{i}/token_complexity",
            plotting.plot_token_complexity(list(toks), c_tok,
                                           title=batch["text"][i][:70],
                                           max_steps=model.encoder.max_steps,
                                           words=words, word_ids=word_ids), step)
        if words:
            logger.log_figure(
                f"train/{i}/word_complexity",
                plotting.plot_word_complexity(
                    words, aggregate_by_word(c_tok, word_ids or [], len(words))), step)
        # Three tags, deliberately distinct. "target_vocoded" is the ground
        # truth mel put through the SAME vocoder as the prediction: it is the
        # ceiling this vocoder can reach, and with Griffin-Lim it already
        # sounds rough. "target_original" is the untouched audio file. If
        # target_vocoded sounds bad but target_original is clean, the vocoder
        # is the limit, not the acoustic model.
        if vocoder is not None:
            try:
                logger.log_audio(f"train/{i}/audio_pred",
                                 vocoder.to_wav(out.mel_post[i, :, :L]), step, sample_rate)
                logger.log_audio(f"train/{i}/audio_target_vocoded",
                                 vocoder.to_wav(batch["mel"][i, :, :L]), step, sample_rate)
            except Exception:
                pass
        path = (batch.get("audio_path") or [""] * (i + 1))[i]
        if path and Path(path).exists():
            try:
                from ..data.audio import AudioConfig, load_wav

                wav = load_wav(path, AudioConfig(sample_rate=sample_rate))
                logger.log_audio(f"train/{i}/audio_target_original", wav, step, sample_rate)
            except Exception:
                pass

    depths = out.encoder_router.ponder[out.encoder_router.mask].float().cpu().numpy()
    if depths.size:
        logger.log_figure("train/token_depth_hist",
                          plotting.plot_depth_histogram(depths, model.encoder.max_steps), step)
        logger.log_histogram("train/token_depth", depths, step)
    fdepths = out.decoder_router.ponder[out.decoder_router.mask].float().cpu().numpy()
    if fdepths.size:
        logger.log_histogram("train/frame_depth", fdepths, step)
    logger.flush_figures()
