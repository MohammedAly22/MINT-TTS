"""Reference-encoder speaker conditioning: the path to zero-shot cloning.

A lookup-table speaker embedding (``nn.Embedding(n_speakers, d)``) can only
ever produce voices it was trained on. Zero-shot cloning needs the voice to be
an *input*, computed from a reference clip, so an unseen speaker is just an
unseen input rather than an unseen id.

This module provides that input. A small convolutional + attentive-pooling
encoder maps a reference mel of any length to one fixed vector, which is then
added to the encoder and decoder states exactly where the lookup embedding
used to be. Nothing else in the model changes.

Training it on a single-speaker corpus
--------------------------------------
This is deliberately enabled even for the 100-hour single-speaker Egyptian run,
where there is only one voice to encode and no cloning is possible. The reason
is that it costs ~1.5M parameters and removes the migration entirely: the
conditioning pathway, the shapes, the checkpoint layout and the inference API
are all the multi-speaker ones from the first step. Adding a second corpus
later is then a data change, not an architecture change that invalidates every
checkpoint trained before it.

On a single-speaker corpus the reference is a random *other* utterance by the
same speaker, never the target utterance itself. Encoding the target would let
the model read the answer off its own reference -- the mel it is being asked to
predict would be available as an input -- and the reconstruction loss would
collapse without anything being learned about voice.

``speaker_dropout`` randomly replaces the reference vector with a learned
"unknown speaker" token. That keeps the model able to synthesise with no
reference at all, and stops it becoming wholly dependent on a conditioning
signal that a user may not supply.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentivePooling(nn.Module):
    """Length-invariant pooling with a learned per-frame weight.

    Mean pooling would let long silences dominate the voice vector; an
    attention weight lets the encoder concentrate on the voiced frames that
    actually carry speaker identity.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(dim, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, 1, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, C, T) -> (B, 2C) (weighted mean and std)."""
        w = self.score(x)                                   # (B, 1, T)
        if mask is not None:
            w = w.masked_fill(~mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(w.float(), dim=-1).to(x.dtype)
        mean = (x * w).sum(-1)
        # Clamped before the sqrt: a reference shorter than a frame or a fully
        # masked one otherwise yields a tiny negative variance and a NaN that
        # propagates into every conditioned state.
        var = ((x - mean.unsqueeze(-1)) ** 2 * w).sum(-1).clamp_min(1e-6)
        return torch.cat([mean, var.sqrt()], dim=-1)


class ReferenceEncoder(nn.Module):
    """Reference mel -> one speaker vector.

    Strided convolutions downsample in time (a 10-second reference does not
    need frame resolution to identify a voice), then attentive pooling removes
    the time axis entirely.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 384,
        hidden: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_dropout_token: bool = True,
    ):
        super().__init__()
        dims = [n_mels] + [hidden] * n_layers
        self.convs = nn.ModuleList([
            nn.Conv1d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1)
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.GroupNorm(8, dims[i + 1]) for i in range(n_layers)])
        self.stride = 2 ** n_layers
        self.pool = AttentivePooling(hidden)
        self.proj = nn.Linear(2 * hidden, d_model)
        self.drop = nn.Dropout(dropout)
        # Used when no reference is supplied, and as the dropout target.
        self.unknown = nn.Parameter(torch.zeros(d_model)) if use_dropout_token else None
        self.d_model = d_model

    def forward(self, mel: torch.Tensor, mel_lens: torch.Tensor | None = None) -> torch.Tensor:
        """mel: (B, n_mels, T) -> (B, d_model)."""
        h = mel
        for conv, norm in zip(self.convs, self.norms):
            h = self.drop(F.silu(norm(conv(h))))
        mask = None
        if mel_lens is not None:
            lens = torch.div(mel_lens - 1, self.stride, rounding_mode="floor") + 1
            lens = lens.clamp(1, h.size(-1))
            ar = torch.arange(h.size(-1), device=h.device)
            mask = ar.unsqueeze(0) < lens.unsqueeze(1)
        return self.proj(self.pool(h, mask))

    def unknown_vector(self, batch: int, device, dtype) -> torch.Tensor:
        if self.unknown is None:
            return torch.zeros(batch, self.d_model, device=device, dtype=dtype)
        return self.unknown.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)

    def apply_dropout(self, vec: torch.Tensor, p: float) -> torch.Tensor:
        """Replace a random subset of references with the unknown token."""
        if not self.training or p <= 0:
            return vec
        keep = (torch.rand(vec.size(0), 1, device=vec.device) >= p).to(vec.dtype)
        unk = self.unknown_vector(vec.size(0), vec.device, vec.dtype)
        return vec * keep + unk * (1.0 - keep)
