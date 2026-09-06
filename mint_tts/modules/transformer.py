"""Transformer building blocks used by both the dense baselines and the
adaptive stacks.

Two forward paths are provided per block:

* ``forward`` -- the standard dense path over all positions (training, and the
  dense baselines).
* ``forward_active`` -- computes the block **only for a subset of positions**,
  reading frozen keys/values of already-halted positions from a cache. This is
  what turns "expected depth" into an actual wall-clock/FLOP saving at
  inference time instead of a masked-out no-op.

Because ``forward_active`` gathers a non-contiguous subset of positions, the
position-wise sublayer must be token-independent: the adaptive stacks
therefore use a *linear* FFN. Local convolutional mixing still exists in the
model, but it lives in the (non-adaptive) pre/post-nets.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positions(length: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    denom = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * denom)
    cos = torch.cos(pos * denom)
    pe[:, 1::2] = cos[:, : pe[:, 1::2].shape[1]]
    return pe.to(dtype)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, learnable_scale: bool = True):
        super().__init__()
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model), persistent=False)
        self.alpha = nn.Parameter(torch.ones(1)) if learnable_scale else None
        self.d_model = d_model

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        T = x.size(1)
        if offset + T > self.pe.size(0):
            self.pe = sinusoidal_positions(offset + T + 512, self.d_model, x.device, x.dtype)
        pe = self.pe[offset: offset + T].to(x.dtype).unsqueeze(0)
        return x + (self.alpha * pe if self.alpha is not None else pe)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                kv: torch.Tensor | None = None) -> torch.Tensor:
        """`kv` lets keys/values come from a different (frozen) state than the
        queries -- the dense equivalent of the halted-position KV cache."""
        source = x if kv is None else kv
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(source))
        v = self._shape(self.v_proj(source))
        attn_mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        B, _, T, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(B, T, self.d_model))

    def attend(self, x_q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Queries from `x_q`; keys/values already projected, (B, T, D)."""
        q = self._shape(self.q_proj(x_q))
        kk, vv = self._shape(k), self._shape(v)
        attn_mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=attn_mask, dropout_p=0.0)
        B, _, A, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(B, A, self.d_model))

    def forward_cached(
        self,
        x_active: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        active_index: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ):
        """Queries for the active subset only; keys/values read from a cache.

        ``k_cache``/``v_cache`` are (B, T, D); the freshly projected active
        states are scattered into them at ``active_index`` (B, A). Valid only
        when the same projection weights are used at every step, i.e. shared
        blocks -- otherwise the cached keys belong to the wrong layer.
        """
        idx = active_index.unsqueeze(-1).expand(-1, -1, self.d_model)
        k_cache = k_cache.scatter(1, idx, self.k_proj(x_active))
        v_cache = v_cache.scatter(1, idx, self.v_proj(x_active))
        return self.attend(x_active, k_cache, v_cache, key_padding_mask), k_cache, v_cache


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class ConvFFN(nn.Module):
    """FastSpeech-style conv FFN. Dense path only (not token-independent)."""

    def __init__(self, d_model: int, ff_dim: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, ff_dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(ff_dim, d_model, kernel_size, padding=pad)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.conv2(self.drop(F.gelu(self.conv1(h))))
        return h.transpose(1, 2)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with a dense and a gather-based path."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        ffn_type: str = "linear",
        conv_kernel: int = 9,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn_type = ffn_type
        self.ffn = (
            PositionwiseFFN(d_model, ff_dim, dropout, activation)
            if ffn_type == "linear"
            else ConvFFN(d_model, ff_dim, conv_kernel, dropout)
        )
        self.drop = nn.Dropout(dropout)
        self.token_independent = ffn_type == "linear"

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                kv_input: torch.Tensor | None = None) -> torch.Tensor:
        kv = None if kv_input is None else self.norm1(kv_input)
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask, kv))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def forward_active(
        self,
        x_full: torch.Tensor,
        x_active: torch.Tensor,
        active_index: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        kv_source: torch.Tensor | None = None,
    ):
        """Apply the block to ``active_index`` positions only.

        Pass ``k_cache``/``v_cache`` for shared blocks (keys of halted
        positions stay valid across steps, so they cost nothing to reuse), or
        ``kv_source`` for independent blocks (this step owns different
        projection weights, so every readable position must be re-projected).

        Returns ``(x_full, x_active, k_cache, v_cache)``.
        """
        if not self.token_independent:
            raise RuntimeError("forward_active requires ffn_type='linear'")
        h = self.norm1(x_active)
        if kv_source is not None:
            hk = self.norm1(kv_source)
            k_cache, v_cache = self.attn.k_proj(hk), self.attn.v_proj(hk)
            attn_out = self.attn.attend(h, k_cache, v_cache, key_padding_mask)
        else:
            attn_out, k_cache, v_cache = self.attn.forward_cached(
                h, k_cache, v_cache, active_index, key_padding_mask
            )
        x_a = x_active + attn_out
        x_a = x_a + self.ffn(self.norm2(x_a))
        idx = active_index.unsqueeze(-1).expand(-1, -1, x_full.size(-1))
        x_full = x_full.scatter(1, idx, x_a)
        return x_full, x_a, k_cache, v_cache


class ConvStack(nn.Module):
    """Local mixing pre/post net: Conv1d + LayerNorm + activation + dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        n_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
        activation: str = "gelu",
        residual: bool = False,
    ):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        pad = (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            [nn.Conv1d(dims[i], dims[i + 1], kernel_size, padding=pad) for i in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu, "tanh": torch.tanh}[activation]
        self.residual = residual and in_dim == out_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C_out); mask: (B, T), True = keep."""
        res = x
        h = x.transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            if mask is not None:
                h = h * mask.unsqueeze(1)
            h = conv(h)
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h = self.drop(self.act(h))
        h = h.transpose(1, 2)
        if self.residual:
            h = h + res
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        return h


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """(B,) int lengths -> (B, T) bool mask, True where valid."""
    max_len = int(max_len if max_len is not None else lengths.max().item())
    ar = torch.arange(max_len, device=lengths.device)
    return ar.unsqueeze(0) < lengths.unsqueeze(1)
