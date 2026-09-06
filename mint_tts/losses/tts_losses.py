"""Reconstruction / alignment / variance losses."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..modules.aligner import BinLoss, ForwardSumLoss


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target: (B, C, T); mask: (B, T) bool."""
    m = mask.unsqueeze(1).to(pred.dtype)
    diff = (pred - target).abs() * m
    return diff.sum() / (m.sum() * pred.size(1)).clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target: (B, T); mask: (B, T) bool."""
    m = mask.to(pred.dtype)
    return (((pred - target) ** 2) * m).sum() / m.sum().clamp_min(1.0)


class TTSLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        w = cfg.loss
        self.w_mel = w.get("mel", 1.0)
        self.w_mel_post = w.get("mel_post", 1.0)
        self.w_duration = w.get("duration", 1.0)
        self.w_pitch = w.get("pitch", 0.1)
        self.w_energy = w.get("energy", 0.1)
        self.w_forwardsum = w.get("forwardsum", 2.0)
        self.w_bin = w.get("bin", 1.0)
        self.bin_warmup_steps = w.get("bin_warmup_steps", 6000)
        self.bin_start_step = w.get("bin_start_step", 6000)
        self.forwardsum = ForwardSumLoss()
        self.bin_loss = BinLoss()

    def forward(self, out, batch, step: int = 0) -> tuple[torch.Tensor, dict]:
        logs: dict[str, torch.Tensor] = {}
        mel_target = batch["mel"]
        mel_mask = out.mel_mask
        text_mask = out.text_mask

        l_mel = masked_l1(out.mel, mel_target, mel_mask)
        l_mel_post = masked_l1(out.mel_post, mel_target, mel_mask)
        total = self.w_mel * l_mel + self.w_mel_post * l_mel_post
        logs["loss/mel"] = l_mel.detach()
        logs["loss/mel_post"] = l_mel_post.detach()

        if out.duration_target is not None:
            log_dur_target = torch.log1p(out.duration_target.to(out.log_duration_pred.dtype))
            l_dur = masked_mse(out.log_duration_pred, log_dur_target, text_mask)
            total = total + self.w_duration * l_dur
            logs["loss/duration"] = l_dur.detach()

        if out.pitch_pred is not None and out.pitch_target is not None:
            l_pitch = masked_mse(out.pitch_pred, out.pitch_target, text_mask)
            total = total + self.w_pitch * l_pitch
            logs["loss/pitch"] = l_pitch.detach()
        if out.energy_pred is not None and out.energy_target is not None:
            l_energy = masked_mse(out.energy_pred, out.energy_target, text_mask)
            total = total + self.w_energy * l_energy
            logs["loss/energy"] = l_energy.detach()

        if out.attn_logprob is not None:
            l_fs = self.forwardsum(out.attn_logprob, batch["token_lens"], batch["mel_lens"])
            total = total + self.w_forwardsum * l_fs
            logs["loss/forwardsum"] = l_fs.detach()
            if step >= self.bin_start_step and self.w_bin > 0:
                ramp = min(1.0, (step - self.bin_start_step) / max(self.bin_warmup_steps, 1))
                l_bin = self.bin_loss(out.attn_hard, out.attn_logprob)
                total = total + self.w_bin * ramp * l_bin
                logs["loss/bin"] = l_bin.detach()

        logs["loss/recon_total"] = total.detach()
        return total, logs
