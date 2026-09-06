"""Variance adaptor: duration / pitch / energy prediction and the length
regulator that turns token-rate states into frame-rate states."""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import ConvStack


class VariancePredictor(nn.Module):
    """Conv stack -> scalar per token (log-duration, pitch or energy)."""

    def __init__(self, d_model: int, hidden: int = 256, n_layers: int = 2,
                 kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        self.body = ConvStack(
            d_model, hidden, hidden, n_layers=n_layers, kernel_size=kernel_size,
            dropout=dropout, activation="relu",
        )
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.body(x, mask)
        out = self.proj(h).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


class QuantizedEmbedding(nn.Module):
    """Bucketises a continuous variance value and embeds it (FastSpeech2)."""

    def __init__(self, d_model: int, n_bins: int = 256, vmin: float = 0.0, vmax: float = 1.0,
                 log_scale: bool = False):
        super().__init__()
        self.log_scale = log_scale
        if log_scale:
            vmin, vmax = float(vmin), float(vmax)
            bins = torch.exp(torch.linspace(torch.tensor(max(vmin, 1e-3)).log(),
                                            torch.tensor(max(vmax, 1e-2)).log(), n_bins - 1))
        else:
            bins = torch.linspace(vmin, vmax, n_bins - 1)
        self.register_buffer("bins", bins, persistent=True)
        self.emb = nn.Embedding(n_bins, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.bucketize(x.detach(), self.bins)
        return self.emb(idx)


def length_regulate(
    x: torch.Tensor, durations: torch.Tensor, max_len: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T, D) + (B, T) integer durations -> (B, L, D), frame mask (B, L).

    Fully vectorised: cumulative durations + searchsorted, no Python loop.
    """
    B, T, D = x.shape
    durations = durations.clamp_min(0).to(torch.long)
    cum = durations.cumsum(1)
    total = cum[:, -1]
    L = int(max_len if max_len is not None else max(int(total.max().item()), 1))
    ar = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    idx = torch.searchsorted(cum.contiguous(), ar.contiguous(), right=True).clamp(max=T - 1)
    out = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
    mask = ar < total.unsqueeze(1)
    return out * mask.unsqueeze(-1), mask


def average_by_duration(frame_values: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Frame-level values (B, L) -> token-level averages (B, T)."""
    B, T = durations.shape
    L = frame_values.shape[1]
    durations = durations.clamp_min(0).to(torch.long)
    cum = durations.cumsum(1)
    ar = torch.arange(L, device=frame_values.device).unsqueeze(0).expand(B, L)
    idx = torch.searchsorted(cum.contiguous(), ar.contiguous(), right=True).clamp(max=T - 1)
    valid = (ar < cum[:, -1:]).to(frame_values.dtype)
    sums = torch.zeros(B, T, device=frame_values.device, dtype=frame_values.dtype)
    sums.scatter_add_(1, idx, frame_values * valid)
    counts = torch.zeros_like(sums).scatter_add_(1, idx, valid)
    return sums / counts.clamp_min(1.0)
