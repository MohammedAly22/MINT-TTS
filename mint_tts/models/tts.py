"""AdaptiveTTS: a non-autoregressive acoustic model whose *depth* is decided
per token (linguistic stack) and per frame (acoustic stack).

    text -> encoder(adaptive depth) -> aligner/duration -> length regulator
         -> decoder(adaptive depth) -> mel -> fixed vocoder -> audio

Setting `routing: fixed` on both stacks turns the exact same code into the
dense baseline, which is what makes the Dense-vs-Adaptive comparison a
controlled experiment rather than two different codebases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..modules.adaptive import RouterOutput, build_stack
from ..modules.aligner import (
    AlignmentEncoder,
    monotonic_alignment_search,
    path_to_durations,
)
from ..modules.transformer import ConvStack, PositionalEncoding, lengths_to_mask
from ..modules.variance import (
    QuantizedEmbedding,
    VariancePredictor,
    average_by_duration,
    length_regulate,
)
from ..utils.flops import (
    FlopReport,
    convstack_flops,
    linear_flops,
    stack_flops,
    variance_predictor_flops,
)


@dataclass
class TTSOutput:
    mel: torch.Tensor                  # (B, n_mels, T_mel) pre-postnet
    mel_post: torch.Tensor             # (B, n_mels, T_mel)
    mel_mask: torch.Tensor             # (B, T_mel) bool
    log_duration_pred: torch.Tensor    # (B, T_text)
    duration_target: torch.Tensor | None
    duration_rounded: torch.Tensor
    encoder_router: RouterOutput
    decoder_router: RouterOutput
    pitch_pred: torch.Tensor | None = None
    energy_pred: torch.Tensor | None = None
    pitch_target: torch.Tensor | None = None
    energy_target: torch.Tensor | None = None
    attn_logprob: torch.Tensor | None = None
    attn_hard: torch.Tensor | None = None
    attn_soft: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None


class AdaptiveTTS(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        d = m.d_model
        self.d_model = d
        self.n_mels = cfg.audio.n_mels
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        nn.init.normal_(self.embedding.weight, 0.0, d ** -0.5)
        self.emb_scale = math.sqrt(d)

        self.encoder_prenet = ConvStack(
            d, d, d, n_layers=m.get("prenet_layers", 3),
            kernel_size=m.get("prenet_kernel", 5), dropout=m.get("dropout", 0.1),
            residual=True,
        )
        self.pos_enc = PositionalEncoding(d, max_len=m.get("max_positions", 8192))
        self.pos_dec = PositionalEncoding(d, max_len=m.get("max_positions", 8192))

        self.encoder = build_stack(m.encoder, d)
        self.decoder = build_stack(m.decoder, d)

        # speaker / emotion conditioning (single-speaker LJSpeech ignores these)
        self.speaker_emb = (
            nn.Embedding(m.n_speakers, d) if m.get("n_speakers", 0) > 1 else None
        )
        self.emotion_emb = (
            nn.Embedding(m.n_emotions, d) if m.get("n_emotions", 0) > 1 else None
        )

        v = m.variance
        self.duration_predictor = VariancePredictor(
            d, v.get("hidden", 256), v.get("layers", 2), v.get("kernel", 3), v.get("dropout", 0.5)
        )
        self.use_pitch = bool(v.get("use_pitch", True))
        self.use_energy = bool(v.get("use_energy", True))
        if self.use_pitch:
            self.pitch_predictor = VariancePredictor(
                d, v.get("hidden", 256), v.get("layers", 2), v.get("kernel", 3), v.get("dropout", 0.5)
            )
            self.pitch_embed = QuantizedEmbedding(d, v.get("n_bins", 256), *v.get("pitch_range", [0.0, 1.0]))
        if self.use_energy:
            self.energy_predictor = VariancePredictor(
                d, v.get("hidden", 256), v.get("layers", 2), v.get("kernel", 3), v.get("dropout", 0.5)
            )
            self.energy_embed = QuantizedEmbedding(d, v.get("n_bins", 256), *v.get("energy_range", [0.0, 1.0]))

        self.aligner = AlignmentEncoder(d, self.n_mels, d_attn=m.get("aligner_dim", 80),
                                        temperature=m.get("aligner_temperature", 0.0005))

        self.mel_linear = nn.Linear(d, self.n_mels)
        self.postnet = ConvStack(
            self.n_mels, m.get("postnet_hidden", 256), self.n_mels,
            n_layers=m.get("postnet_layers", 5), kernel_size=m.get("postnet_kernel", 5),
            dropout=m.get("dropout", 0.1), activation="tanh",
        )
        self.min_duration = int(m.get("min_duration", 1))

    # -- helpers ----------------------------------------------------------
    def _condition(self, x, speakers, emotions):
        if self.speaker_emb is not None and speakers is not None:
            x = x + self.speaker_emb(speakers).unsqueeze(1)
        if self.emotion_emb is not None and emotions is not None:
            x = x + self.emotion_emb(emotions).unsqueeze(1)
        return x

    def encode_text(self, tokens, text_mask, budget=None, hard=False, max_steps=None,
                    speakers=None, emotions=None):
        x = self.embedding(tokens) * self.emb_scale
        x = self.encoder_prenet(x, text_mask)
        x = self.pos_enc(x) * text_mask.unsqueeze(-1)
        x = self._condition(x, speakers, emotions)
        enc = self.encoder(x, text_mask, budget=budget, hard=hard, max_steps_override=max_steps)
        return enc

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        token_lens: torch.Tensor,
        mels: torch.Tensor | None = None,
        mel_lens: torch.Tensor | None = None,
        pitch: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        speakers: torch.Tensor | None = None,
        emotions: torch.Tensor | None = None,
        budget: torch.Tensor | None = None,
        attn_prior: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        hard: bool = False,
        duration_scale: float = 1.0,
        encoder_max_steps: int | None = None,
        decoder_max_steps: int | None = None,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
    ) -> TTSOutput:
        text_mask = lengths_to_mask(token_lens, tokens.size(1))
        enc = self.encode_text(tokens, text_mask, budget, hard, encoder_max_steps,
                               speakers, emotions)
        h = enc.output

        attn_logprob = attn_hard = attn_soft = None
        duration_target = durations
        training_mode = mels is not None and mel_lens is not None

        if training_mode and durations is None:
            attn_logprob, _ = self.aligner(h, mels, text_mask, attn_prior)
            attn_hard = monotonic_alignment_search(
                attn_logprob.squeeze(1).float(), token_lens, mel_lens
            ).to(h.dtype)
            attn_soft = attn_logprob.squeeze(1).exp()
            duration_target = path_to_durations(attn_hard)

        log_duration_pred = self.duration_predictor(h, text_mask)

        if training_mode:
            dur_used = duration_target
        else:
            dur = torch.expm1(log_duration_pred.clamp(max=15.0)).clamp_min(0)
            dur = (dur * duration_scale).round().clamp_min(self.min_duration)
            dur = dur.masked_fill(~text_mask, 0)
            # never emit an empty utterance
            dur[:, 0] = torch.where(dur.sum(1) < 1, torch.ones_like(dur[:, 0]), dur[:, 0])
            dur_used = dur
        duration_rounded = dur_used.round().to(torch.long)

        pitch_pred = energy_pred = pitch_target = energy_target = None
        if self.use_pitch:
            pitch_pred = self.pitch_predictor(h, text_mask)
            if training_mode and pitch is not None:
                pitch_target = average_by_duration(pitch, duration_rounded)
                h = h + self.pitch_embed(pitch_target)
            else:
                h = h + self.pitch_embed(pitch_pred * pitch_scale)
        if self.use_energy:
            energy_pred = self.energy_predictor(h, text_mask)
            if training_mode and energy is not None:
                energy_target = average_by_duration(energy, duration_rounded)
                h = h + self.energy_embed(energy_target)
            else:
                h = h + self.energy_embed(energy_pred * energy_scale)

        target_len = int(mels.size(-1)) if training_mode else None
        frames, mel_mask = length_regulate(h, duration_rounded, target_len)
        frames = self.pos_dec(frames) * mel_mask.unsqueeze(-1)
        frames = self._condition(frames, speakers, emotions)
        dec = self.decoder(frames, mel_mask, budget=budget, hard=hard,
                           max_steps_override=decoder_max_steps)

        mel = self.mel_linear(dec.output).transpose(1, 2)
        mel = mel * mel_mask.unsqueeze(1)
        mel_post = mel + self.postnet(mel.transpose(1, 2), mel_mask).transpose(1, 2)
        mel_post = mel_post * mel_mask.unsqueeze(1)

        return TTSOutput(
            mel=mel,
            mel_post=mel_post,
            mel_mask=mel_mask,
            log_duration_pred=log_duration_pred,
            duration_target=duration_target,
            duration_rounded=duration_rounded,
            encoder_router=enc,
            decoder_router=dec,
            pitch_pred=pitch_pred,
            energy_pred=energy_pred,
            pitch_target=pitch_target,
            energy_target=energy_target,
            attn_logprob=attn_logprob,
            attn_hard=attn_hard,
            attn_soft=attn_soft,
            text_mask=text_mask,
        )

    # -- inference --------------------------------------------------------
    @torch.inference_mode()
    def infer(
        self,
        tokens: torch.Tensor,
        token_lens: torch.Tensor,
        budget: torch.Tensor | None = None,
        hard: bool = True,
        **kwargs,
    ) -> TTSOutput:
        self.eval()
        return self.forward(tokens, token_lens, budget=budget, hard=hard, **kwargs)

    # -- accounting -------------------------------------------------------
    def flops(self, out: TTSOutput) -> FlopReport:
        """Exact FLOPs for the forward pass that produced `out` (batch total)."""
        rep = FlopReport()
        T_text = int(out.text_mask.sum().item())
        T_mel = int(out.mel_mask.sum().item())
        rep.add("embedding", 0.0)
        rep.add("encoder_prenet", convstack_flops(self.encoder_prenet, T_text))
        rep.add("encoder", stack_flops(self.encoder, out.encoder_router.token_steps,
                                       out.encoder_router.attn_kv_steps,
                                       out.encoder_router.kv_token_steps))
        rep.add("duration_predictor", variance_predictor_flops(self.duration_predictor, T_text))
        if self.use_pitch:
            rep.add("pitch_predictor", variance_predictor_flops(self.pitch_predictor, T_text))
        if self.use_energy:
            rep.add("energy_predictor", variance_predictor_flops(self.energy_predictor, T_text))
        rep.add("decoder", stack_flops(self.decoder, out.decoder_router.token_steps,
                                       out.decoder_router.attn_kv_steps,
                                       out.decoder_router.kv_token_steps))
        rep.add("mel_linear", linear_flops(self.d_model, self.n_mels, T_mel))
        rep.add("postnet", convstack_flops(self.postnet, T_mel))

        # The dense reference must exclude padding, otherwise a batch with
        # ragged lengths would show a "saving" that is really just padding.
        dense = rep.total - rep.parts["encoder"] - rep.parts["decoder"]
        dense += _dense_flops(self.encoder, out.text_mask)
        dense += _dense_flops(self.decoder, out.mel_mask)
        rep.dense_equivalent = dense
        return rep


def _dense_flops(stack, mask: torch.Tensor) -> float:
    """FLOPs this stack would cost at full depth on the *unpadded* sequences."""
    lens = mask.sum(1).to(torch.float64)
    steps = float(stack.max_steps)
    token_steps = float((lens * steps).sum())
    kv_pairs = float((lens * lens * steps).sum())
    return stack_flops(stack, token_steps, kv_pairs, token_steps)


def build_model(cfg, vocab_size: int) -> AdaptiveTTS:
    return AdaptiveTTS(cfg, vocab_size)
