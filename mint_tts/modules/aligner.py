"""Unsupervised text<->mel aligner (RAD-TTS / "One TTS Alignment To Rule Them
All" style) plus monotonic alignment search.

Using an internal aligner instead of Montreal Forced Aligner keeps the repo
self-contained (`git clone` -> `train`, no external toolchain), and the
alignment matrix it produces is exactly the diagnostic we want on TensorBoard:
if alignment collapses, every downstream complexity number is meaningless.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


class AlignmentEncoder(nn.Module):
    """Projects text and mel into a shared space and scores soft alignment.

    Scale matters more than it looks. The score is ``-scale * ||q - k||^2``,
    and the text side of this model is LayerNormed, so the squared distances
    land around 30 rather than the tens of thousands seen when the same
    formulation is fed raw embeddings. With a fixed temperature of 5e-4 the
    logits then span ~0.01, the attention comes out *exactly uniform*, and
    gradients into the projections are scaled by that same 5e-4 -- the aligner
    is effectively frozen and only the beta-binomial prior carries any signal.

    So the distance is normalised per channel and the scale is a learned
    parameter, which makes the initial sharpness independent of feature
    magnitude and lets the model sharpen the alignment as it trains.
    """

    def __init__(self, d_text: int, n_mels: int, d_attn: int = 80, temperature: float = 1.0):
        super().__init__()
        self.d_attn = d_attn
        # softplus keeps it positive; init so softplus(raw) == temperature
        init = math.log(math.expm1(max(float(temperature), 1e-3)))
        self.log_scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.key_proj = nn.Sequential(
            nn.Conv1d(d_text, d_text * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_text * 2, d_attn, kernel_size=1),
        )
        self.query_proj = nn.Sequential(
            nn.Conv1d(n_mels, n_mels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_mels * 2, n_mels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_mels, d_attn, kernel_size=1),
        )

    def forward(
        self,
        text: torch.Tensor,       # (B, T_text, D)
        mel: torch.Tensor,        # (B, n_mels, T_mel)
        text_mask: torch.Tensor,  # (B, T_text) bool
        attn_prior: torch.Tensor | None = None,  # (B, T_mel, T_text)
    ):
        keys = self.key_proj(text.transpose(1, 2))     # (B, d_attn, T_text)
        queries = self.query_proj(mel)                 # (B, d_attn, T_mel)
        # ||q - k||^2 expanded as |q|^2 + |k|^2 - 2 q.k so we never materialise
        # the (B, d_attn, T_mel, T_text) difference tensor.
        q = queries.transpose(1, 2)                    # (B, T_mel, d)
        k = keys.transpose(1, 2)                       # (B, T_text, d)
        dist = (
            q.pow(2).sum(-1, keepdim=True)
            + k.pow(2).sum(-1).unsqueeze(1)
            - 2.0 * torch.bmm(q, k.transpose(1, 2))
        ).clamp_min(0)                                 # (B, T_mel, T_text)
        # per-channel mean squared distance x a learned scale
        scale = F.softplus(self.log_scale)
        score = (-scale * dist / self.d_attn).unsqueeze(1)   # (B, 1, T_mel, T_text)

        if attn_prior is not None:
            score = F.log_softmax(score, dim=-1) + torch.log(attn_prior.unsqueeze(1) + 1e-8)

        score = score.masked_fill(~text_mask[:, None, None, :], NEG)
        logprob = F.log_softmax(score, dim=-1)
        return logprob, score


def beta_binomial_prior(text_len: int, mel_len: int, scaling: float = 1.0) -> torch.Tensor:
    """Diagonal-ish prior that dramatically speeds up alignment learning."""
    x = torch.arange(1, text_len + 1, dtype=torch.float64)
    m = torch.arange(1, mel_len + 1, dtype=torch.float64).unsqueeze(1)
    a = scaling * m
    b = scaling * (mel_len + 1 - m)
    # log Beta-binomial pmf, computed with lgamma for stability
    n = torch.full_like(x, float(text_len - 1))
    k = x - 1
    log_c = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    log_beta_num = torch.lgamma(k + a) + torch.lgamma(n - k + b) - torch.lgamma(n + a + b)
    log_beta_den = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    logp = log_c + log_beta_num - log_beta_den
    p = torch.exp(logp - logp.max(dim=1, keepdim=True).values)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return p.float()  # (T_mel, T_text)


def beta_binomial_prior_batch(
    text_lens: torch.Tensor, mel_lens: torch.Tensor, max_text: int, max_mel: int,
    scaling: float = 1.0,
) -> torch.Tensor:
    """The same prior for a whole padded batch, on whatever device the lengths live.

    Building it per utterance in the DataLoader cost 25-55 ms of float64
    `lgamma` per clip at this corpus's lengths -- seconds of CPU per batch --
    and then shipped a ~200 MB (B, T_mel, T_text) tensor through worker IPC,
    pinning and the host->device copy on every step. On the GPU it is a few
    elementwise kernels.

    Every term that is constant along a mel row (the Beta normaliser and
    lgamma(n + a + b)) cancels in the per-row normalisation, so only two large
    lgamma evaluations remain. float32 is accurate to ~1e-3 in log space here,
    far below anything the aligner can resolve.
    Returns (B, max_mel, max_text), zero on padding.
    """
    device = text_lens.device
    t = text_lens.to(torch.float32).view(-1, 1, 1)                  # (B,1,1)
    l = mel_lens.to(torch.float32).view(-1, 1, 1)
    k = torch.arange(max_text, device=device, dtype=torch.float32).view(1, 1, -1)
    m = torch.arange(1, max_mel + 1, device=device, dtype=torch.float32).view(1, -1, 1)
    n = t - 1
    a = scaling * m
    b = scaling * (l + 1 - m)
    valid_x = k < t                                                  # (B,1,Tt)
    valid_y = m <= l                                                 # (B,Ty,1)
    # clamp keeps lgamma finite on padding; those cells are masked below
    n_minus_k = (n - k).clamp_min(0)
    log_c = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n_minus_k + 1)
    logp = log_c + torch.lgamma(k + a) + torch.lgamma(n_minus_k + b.clamp_min(1e-3))
    logp = logp.masked_fill(~valid_x, float("-inf"))
    p = torch.softmax(logp, dim=-1)
    return p.masked_fill(~valid_y, 0.0)

class ForwardSumLoss(nn.Module):
    """CTC-based forward-sum over all monotonic alignments."""

    def __init__(self, blank_logprob: float = -1.0):
        super().__init__()
        self.blank_logprob = blank_logprob
        self.ctc = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, text_lens, mel_lens):
        # attn_logprob: (B, 1, T_mel, T_text)
        B, _, T_mel, T_text = attn_logprob.shape
        lp = F.pad(attn_logprob.squeeze(1), (1, 0), value=self.blank_logprob)
        lp = F.log_softmax(lp, dim=-1).permute(1, 0, 2)  # (T_mel, B, T_text+1)
        targets = torch.arange(1, T_text + 1, device=attn_logprob.device)
        targets = targets.unsqueeze(0).expand(B, -1)
        return self.ctc(lp, targets, mel_lens.to(torch.long), text_lens.to(torch.long))


class BinLoss(nn.Module):
    """Pushes the soft alignment towards the hard (MAS) alignment."""

    def forward(self, hard_attn, soft_logprob):
        # hard_attn: (B, T_mel, T_text); soft_logprob: (B, 1, T_mel, T_text)
        log_sum = (soft_logprob.squeeze(1) * hard_attn).sum()
        return -log_sum / hard_attn.sum().clamp_min(1.0)


try:  # numba is optional: without it MAS falls back to the torch loop below
    import numba as _numba
    import numpy as _np

    @_numba.njit(nogil=True, cache=True)
    def _mas_one(value, t_x, t_y, out):  # pragma: no cover - compiled
        """Viterbi for one utterance; writes per-token durations into `out`."""
        neg = -1e30
        cum = _np.full((t_y, t_x), neg, dtype=_np.float64)
        for y in range(t_y):
            # x can neither run ahead of y nor fall so far behind that the
            # last token becomes unreachable
            lo = max(0, t_x + y - t_y)
            hi = min(t_x, y + 1)
            for x in range(lo, hi):
                if y == 0:
                    prev = 0.0 if x == 0 else neg
                else:
                    stay = cum[y - 1, x]
                    move = cum[y - 1, x - 1] if x > 0 else neg
                    prev = move if move > stay else stay
                cum[y, x] = prev + value[y, x]
        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            out[index] += 1
            if index > 0 and (index == y or cum[y - 1, index - 1] > cum[y - 1, index]):
                index -= 1

    @_numba.njit(parallel=True, cache=True)
    def _mas_batch(values, t_xs, t_ys, out):  # pragma: no cover - compiled
        for b in _numba.prange(values.shape[0]):
            if t_xs[b] > 0 and t_ys[b] > 0:
                _mas_one(values[b], t_xs[b], t_ys[b], out[b])

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - depends on the environment
    HAVE_NUMBA = False


def durations_to_path(durations: torch.Tensor, max_mel: int, dtype=torch.float32) -> torch.Tensor:
    """(B, T_text) integer durations -> (B, T_mel, T_text) hard 0/1 path."""
    B, Tx = durations.shape
    cum = durations.to(torch.long).cumsum(1)
    ar = torch.arange(max_mel, device=durations.device).unsqueeze(0).expand(B, -1)
    idx = torch.searchsorted(cum.contiguous(), ar.contiguous(), right=True)
    valid = ar < cum[:, -1:]
    path = F.one_hot(idx.clamp(max=Tx - 1), num_classes=Tx).to(dtype)
    return path * valid.unsqueeze(-1).to(dtype)


@torch.no_grad()
def monotonic_alignment_search(
    neg_cent: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor,
    backend: str = "auto",
) -> torch.Tensor:
    """Viterbi over monotonic, surjective alignments.

    neg_cent: (B, T_mel, T_text) log-likelihood of aligning mel frame y to
    text token x. Returns a hard 0/1 path of the same shape.

    With numba installed this runs compiled on the CPU, parallel over the
    batch: the torch fallback issues ~8 tiny kernels per mel frame in both
    passes -- ~30k launches per step at 1800 frames -- which made the search
    alone a large fraction of every training step.
    """
    if backend == "numba" or (backend == "auto" and HAVE_NUMBA):
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not installed")
        values = neg_cent.detach().float().cpu().numpy()
        t_xs = text_lens.detach().cpu().numpy().astype(_np.int64)
        t_ys = mel_lens.detach().cpu().numpy().astype(_np.int64)
        dur = _np.zeros((values.shape[0], values.shape[2]), dtype=_np.int64)
        _mas_batch(values, t_xs, t_ys, dur)
        dur_t = torch.from_numpy(dur).to(neg_cent.device, non_blocking=True)
        return durations_to_path(dur_t, neg_cent.shape[1], neg_cent.dtype)
    return _mas_torch(neg_cent, text_lens, mel_lens)


def _mas_torch(neg_cent: torch.Tensor, text_lens: torch.Tensor,
               mel_lens: torch.Tensor) -> torch.Tensor:
    """Vectorised over batch and text; loops in Python over mel frames."""
    B, Ty, Tx = neg_cent.shape
    device, dtype = neg_cent.device, neg_cent.dtype
    x_range = torch.arange(Tx, device=device)
    valid_x = x_range.unsqueeze(0) < text_lens.unsqueeze(1)

    cum = torch.full((B, Tx), NEG, device=device, dtype=dtype)
    directions = torch.zeros(B, Ty, Tx, dtype=torch.bool, device=device)

    for y in range(Ty):
        if y == 0:
            new = torch.full_like(cum, NEG)
            new[:, 0] = neg_cent[:, 0, 0]
        else:
            shifted = torch.cat(
                [torch.full((B, 1), NEG, device=device, dtype=dtype), cum[:, :-1]], dim=1
            )
            directions[:, y, :] = shifted > cum
            new = torch.maximum(cum, shifted) + neg_cent[:, y, :]
        cum = torch.where(valid_x, new, torch.full_like(new, NEG))

    path = torch.zeros(B, Ty, Tx, device=device, dtype=dtype)
    index = (text_lens - 1).clamp_min(0).to(torch.long)
    for y in range(Ty - 1, -1, -1):
        active = (y < mel_lens).to(dtype)
        onehot = F.one_hot(index, num_classes=Tx).to(dtype)
        path[:, y, :] = onehot * active.unsqueeze(1)
        came_from_prev = directions[:, y, :].gather(1, index.unsqueeze(1)).squeeze(1)
        step_back = came_from_prev & (active > 0)
        index = torch.where(step_back, index - 1, index).clamp_min(0)
    return path


def path_to_durations(path: torch.Tensor) -> torch.Tensor:
    """(B, T_mel, T_text) hard path -> (B, T_text) integer durations."""
    return path.sum(1)
