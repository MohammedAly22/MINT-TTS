"""Homograph probe: does the model *pronounce* ambiguous words differently?

The complexity heatmaps answer "where did the compute go". They do not answer
the question the hypothesis actually rests on: **did spending that compute
change the pronunciation?** A router can happily assign extra depth to
"record" and still say it identically in both contexts, which would be a
null result dressed up as a positive one.

So for each minimal pair -- the same spelling in two contexts that demand
different pronunciations -- this probe synthesises both sentences, cuts out the
mel frames belonging to the ambiguous word in each, and measures how far apart
they are.

The number in isolation means little, because *any* two renderings of a word
differ a bit (different neighbours, different prosody). So every pair also
measures a **control**: the words that appear in *both* sentences and are not
ambiguous. The signal is the ratio.

    homograph/divergence_ratio  >> 1   the model treats the ambiguous word
                                       differently across contexts, beyond the
                                       ordinary contextual variation
                                ~= 1   it renders it the same way both times:
                                       no disambiguation is happening

This is a necessary condition, not a sufficient one: differing is not the same
as differing *correctly*. Judging correctness needs the logged audio or a
phoneme recogniser. The probe writes both renditions to TensorBoard so they can
simply be listened to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..evaluation.metrics import mel_cepstral_distortion


@dataclass
class MinimalPair:
    word: str
    sentence_a: str
    sentence_b: str
    note: str = ""
    extras: dict = field(default_factory=dict)


DEFAULT_PAIRS = [
    MinimalPair("read", "I read a book yesterday.", "I will read a book tomorrow.",
                "past /red/ vs present /reed/ - decided by the temporal adverb"),
    MinimalPair("record", "Please record the album today.", "She bought the record today.",
                "verb stresses the second syllable, noun the first"),
    MinimalPair("lead", "He will lead the team.", "The pipe is made of lead.",
                "verb /leed/ vs metal /led/"),
    MinimalPair("live", "They live in Berlin.", "It was a live broadcast.",
                "verb /liv/ vs adjective /lyve/"),
    MinimalPair("present", "Please present your work.", "She opened the present.",
                "verb vs noun, stress shift"),
    MinimalPair("desert", "Do not desert your post.", "They crossed the desert.",
                "verb vs noun, stress shift"),
    MinimalPair("close", "Please close the door.", "The station is close.",
                "verb /clohz/ vs adjective /clohs/"),
    MinimalPair("wind", "Wind the clock carefully.", "The wind was very strong.",
                "verb /wynd/ vs noun /wind/"),
]


def load_pairs(cfg) -> list[MinimalPair]:
    items = cfg.log.get("homograph_pairs", None)
    if not items:
        return list(DEFAULT_PAIRS)
    out = []
    for it in items:
        out.append(MinimalPair(word=it["word"], sentence_a=it["a"], sentence_b=it["b"],
                               note=it.get("note", "")))
    return out


def word_frame_span(word_index: int, word_ids: list[int], durations) -> tuple[int, int]:
    """Mel-frame range covered by one word, from the predicted durations."""
    dur = durations.detach().cpu().to(torch.long)
    idx = [i for i, w in enumerate(word_ids) if w == word_index and i < dur.numel()]
    if not idx:
        return 0, 0
    starts = torch.cumsum(dur, 0) - dur
    return int(starts[idx[0]]), int(starts[idx[-1]] + dur[idx[-1]])


def _resample(mel: torch.Tensor, length: int = 16) -> torch.Tensor:
    if mel.shape[-1] == 0:
        return torch.zeros(mel.shape[0], length)
    return torch.nn.functional.interpolate(
        mel.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)


def span_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Length-invariant distance between two renderings of the same word."""
    if a.shape[-1] == 0 or b.shape[-1] == 0:
        return float("nan")
    return mel_cepstral_distortion(_resample(a), _resample(b))


class HomographProbe:
    def __init__(self, cfg, text_processor, device):
        self.cfg = cfg
        self.tp = text_processor
        self.device = device
        self.pairs = load_pairs(cfg)
        self.quality = float(cfg.log.get("homograph_quality", 0.9))
        self.log_audio = bool(cfg.log.get("homograph_audio", True))
        self.max_audio = int(cfg.log.get("homograph_max_audio", 4))

    @torch.inference_mode()
    def _synth(self, model, text: str, routing_frozen: bool):
        enc = self.tp.encode(text)
        tokens = torch.tensor(enc.ids, dtype=torch.long, device=self.device).unsqueeze(0)
        lens = torch.tensor([len(enc.ids)], dtype=torch.long, device=self.device)
        budget = torch.tensor([[self.quality, 1.0]], device=self.device)
        out = model(tokens, lens, budget=budget, hard=True, force_full_depth=routing_frozen)
        return enc, out

    @staticmethod
    def _word_compute(enc, out, word_index: int) -> float:
        c = out.encoder_router.complexity[0].float().cpu().numpy()
        vals = [c[i] for i, w in enumerate(enc.word_ids) if w == word_index and i < len(c)]
        return float(np.mean(vals)) if vals else float("nan")

    @torch.inference_mode()
    def run(self, model, logger, step: int, vocoder=None, routing_frozen: bool = False) -> dict:
        was_training = model.training
        model.eval()
        scalars: dict[str, float] = {}
        rows, div, ctrl, dcomp = [], [], [], []

        for pi, pair in enumerate(self.pairs):
            enc_a, out_a = self._synth(model, pair.sentence_a, routing_frozen)
            enc_b, out_b = self._synth(model, pair.sentence_b, routing_frozen)
            target = pair.word.lower()
            if target not in enc_a.words or target not in enc_b.words:
                continue
            ia, ib = enc_a.words.index(target), enc_b.words.index(target)

            mel_a = out_a.mel_post[0].float().cpu()
            mel_b = out_b.mel_post[0].float().cpu()
            sa = word_frame_span(ia, enc_a.word_ids, out_a.duration_rounded[0])
            sb = word_frame_span(ib, enc_b.word_ids, out_b.duration_rounded[0])
            d = span_distance(mel_a[:, sa[0]:sa[1]], mel_b[:, sb[0]:sb[1]])

            # control: unambiguous words present in BOTH sentences
            shared = [w for w in set(enc_a.words) & set(enc_b.words) if w != target]
            controls = []
            for w in shared:
                ca = word_frame_span(enc_a.words.index(w), enc_a.word_ids, out_a.duration_rounded[0])
                cb = word_frame_span(enc_b.words.index(w), enc_b.word_ids, out_b.duration_rounded[0])
                v = span_distance(mel_a[:, ca[0]:ca[1]], mel_b[:, cb[0]:cb[1]])
                if np.isfinite(v):
                    controls.append(v)
            control = float(np.mean(controls)) if controls else float("nan")

            comp_a = self._word_compute(enc_a, out_a, ia)
            comp_b = self._word_compute(enc_b, out_b, ib)
            others_a = [self._word_compute(enc_a, out_a, j)
                        for j in range(len(enc_a.words)) if j != ia]
            other = float(np.nanmean(others_a)) if others_a else float("nan")

            tag = f"homograph/{pair.word}"
            if np.isfinite(d):
                scalars[f"{tag}/divergence"] = d
                div.append(d)
            if np.isfinite(control):
                scalars[f"{tag}/control_divergence"] = control
                ctrl.append(control)
            if np.isfinite(comp_a) and np.isfinite(comp_b):
                scalars[f"{tag}/compute"] = 0.5 * (comp_a + comp_b)
                if np.isfinite(other):
                    dcomp.append(0.5 * (comp_a + comp_b) - other)

            if vocoder is not None and self.log_audio and pi < self.max_audio:
                for name, out_ in (("a", out_a), ("b", out_b)):
                    try:
                        L = int(out_.mel_mask[0].sum().item())
                        logger.log_audio(f"{tag}/audio_{name}",
                                         vocoder.to_wav(out_.mel_post[0, :, :L]),
                                         step, self.cfg.audio.sample_rate)
                    except Exception:
                        pass

            rows.append([pair.word, pair.sentence_a[:40], pair.sentence_b[:40],
                         f"{d:.2f}", f"{control:.2f}",
                         f"{(d / control) if control else float('nan'):.2f}",
                         f"{comp_a:.3f}", f"{comp_b:.3f}", f"{other:.3f}"])

        if div:
            scalars["homograph/divergence_mean"] = float(np.mean(div))
        if ctrl:
            scalars["homograph/control_divergence_mean"] = float(np.mean(ctrl))
        if div and ctrl and np.mean(ctrl) > 1e-6:
            # >1 means the ambiguous word varies MORE across contexts than
            # ordinary words do. That is the claim, as a single number.
            scalars["homograph/divergence_ratio"] = float(np.mean(div) / np.mean(ctrl))
        if dcomp:
            scalars["homograph/compute_advantage"] = float(np.mean(dcomp))

        logger.log_table(
            "homograph/table",
            ["word", "context A", "context B", "divergence", "control",
             "ratio", "compute A", "compute B", "compute others"],
            rows, step)
        logger.log_scalars(scalars, step)
        if was_training:
            model.train()
        return scalars
