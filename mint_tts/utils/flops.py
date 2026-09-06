"""Analytic FLOP accounting.

We count FLOPs analytically rather than by profiling because the claim we are
testing is about *arithmetic actually performed*, and because it has to be
comparable across CPU, a GTX 1660 Ti and an A100 without kernel-level noise.
Convention: one multiply-accumulate = 2 FLOPs.

The adaptive stacks report the exact number of executed token-steps and
scored query/key pairs, so `stack_flops` is exact, not an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


def linear_flops(in_dim: int, out_dim: int, n_positions: int) -> float:
    return 2.0 * in_dim * out_dim * n_positions


def conv1d_flops(in_ch: int, out_ch: int, kernel: int, length: int) -> float:
    return 2.0 * in_ch * out_ch * kernel * length


def convstack_flops(stack, length: int) -> float:
    total = 0.0
    for conv in stack.convs:
        total += conv1d_flops(conv.in_channels, conv.out_channels, conv.kernel_size[0], length)
    return total


def variance_predictor_flops(pred, length: int) -> float:
    return convstack_flops(pred.body, length) + linear_flops(
        pred.proj.in_features, pred.proj.out_features, length
    )


def stack_flops(stack, token_steps: float, attn_kv_steps: float,
                kv_token_steps: float | None = None) -> float:
    """Exact FLOPs for an AdaptiveStack given its executed work.

    token_steps    -- executed token-steps: q/out projections and the FFN
    kv_token_steps -- positions whose keys/values were projected. Equal to
                      token_steps for shared blocks (a halted position keeps
                      valid cached keys); for independent blocks every readable
                      position must be re-projected at every step.
    attn_kv_steps  -- sum over steps of (#queries x #keys) actually scored
    """
    d, f = stack.d_model, stack.ff_dim
    if kv_token_steps is None:
        kv_token_steps = token_steps
    per_token = 4.0 * d * d + 4.0 * d * f          # q/out projections + FFN
    per_kv = 4.0 * d * d                           # k/v projections
    per_qk_pair = 4.0 * d                          # scores + weighted value sum
    router = 0.0
    if stack.router is not None:
        lin = [m for m in stack.router.net if isinstance(m, nn.Linear)]
        router = sum(linear_flops(m.in_features, m.out_features, 1) for m in lin)
    return (token_steps * (per_token + router) + kv_token_steps * per_kv
            + attn_kv_steps * per_qk_pair)


def dense_stack_flops(stack, seq_len: int, batch: int = 1, steps: int | None = None) -> float:
    """FLOPs the same stack would cost if every token took every step."""
    steps = steps if steps is not None else stack.max_steps
    token_steps = float(batch * seq_len * steps)
    return stack_flops(stack, token_steps, token_steps * seq_len, token_steps)


@dataclass
class FlopReport:
    total: float = 0.0
    parts: dict = field(default_factory=dict)
    dense_equivalent: float = 0.0

    def add(self, name: str, value: float) -> None:
        self.parts[name] = self.parts.get(name, 0.0) + float(value)
        self.total += float(value)

    @property
    def saving(self) -> float:
        """Fraction of the dense-equivalent FLOPs that were skipped."""
        if self.dense_equivalent <= 0:
            return 0.0
        return 1.0 - self.total / self.dense_equivalent

    def as_dict(self) -> dict:
        out = {"flops_total": self.total, "flops_dense_equivalent": self.dense_equivalent,
               "flops_saving": self.saving}
        out.update({f"flops/{k}": v for k, v in self.parts.items()})
        return out


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def parameter_table(model: nn.Module, top_level: bool = True) -> dict[str, int]:
    out = {}
    for name, child in model.named_children():
        out[name] = count_parameters(child, trainable_only=False)
    out["TOTAL"] = count_parameters(model, trainable_only=False)
    return out


def human(n: float) -> str:
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(n) < 1000.0:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}E"
