"""The compute-allocation objective.

The whole hypothesis lives here. Without pressure on computation, a router
always learns the degenerate policy "use every step" -- it is the cheapest way
to minimise reconstruction error. The penalty below makes the model answer a
different question:

    minimise computation  subject to  quality staying above the requested level

Concretely, per batch element we sample a quality budget q ~ U[q_lo, q_hi] and
(optionally) a hardware budget h. The compute penalty is scaled by (1 - q), so
one checkpoint learns the whole quality/compute trade-off curve C*(x, q, h)
instead of a single operating point.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_budget(
    batch_size: int,
    device,
    q_range=(0.0, 1.0),
    h_range=(0.0, 1.0),
    use_hardware: bool = True,
    fixed_q: float | None = None,
    fixed_h: float | None = None,
) -> torch.Tensor:
    """Returns (B, 2) budget vectors [q, h]."""
    if fixed_q is None:
        q = torch.rand(batch_size, 1, device=device) * (q_range[1] - q_range[0]) + q_range[0]
    else:
        q = torch.full((batch_size, 1), float(fixed_q), device=device)
    if not use_hardware:
        h = torch.ones(batch_size, 1, device=device)
    elif fixed_h is None:
        h = torch.rand(batch_size, 1, device=device) * (h_range[1] - h_range[0]) + h_range[0]
    else:
        h = torch.full((batch_size, 1), float(fixed_h), device=device)
    return torch.cat([q, h], dim=1)


class ComputeLoss(nn.Module):
    """lambda(q) * normalised-ponder  +  hardware-cap hinge  (+ optional target)."""

    def __init__(self, cfg):
        super().__init__()
        c = cfg.loss.compute
        self.enabled = bool(c.get("enabled", True))
        self.lambda_max = float(c.get("lambda_max", 0.05))
        self.warmup_steps = int(c.get("warmup_steps", 5000))
        self.ramp_steps = int(c.get("ramp_steps", 5000))
        self.q_exponent = float(c.get("q_exponent", 1.0))
        self.lambda_floor = float(c.get("lambda_floor", 0.02))
        self.encoder_weight = float(c.get("encoder_weight", 1.0))
        self.decoder_weight = float(c.get("decoder_weight", 1.0))
        self.hardware_weight = float(c.get("hardware_weight", 0.5))
        self.use_hardware_cap = bool(c.get("use_hardware_cap", True))
        self.target_compute = c.get("target_compute", None)
        self.target_weight = float(c.get("target_weight", 0.0))
        self.diversity_weight = float(c.get("diversity_weight", 0.0))
        self.c_star_weight = float(c.get("c_star_weight", 0.0))

    def lambda_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0
        ramp = min(1.0, (step - self.warmup_steps) / max(self.ramp_steps, 1))
        return self.lambda_max * ramp

    @staticmethod
    def _norm_ponder(router) -> torch.Tensor:
        """Per-utterance normalised ACT ponder cost.

        This is the *differentiable* compute surrogate, so it is deliberately
        not clamped: ponder is n_updates + remainder and can slightly exceed
        max_steps. The reported compute fraction (`RouterOutput.complexity`)
        uses the executed step count instead, and is the number to quote.
        """
        n = max(router.halting_probs.size(1), 1)
        return router.per_utterance_ponder() / n

    def forward(self, out, budget: torch.Tensor | None, step: int,
                c_star: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        device = out.mel.device
        logs: dict[str, torch.Tensor] = {}
        zero = torch.zeros((), device=device)
        if not self.enabled:
            return zero, logs

        c_enc = self._norm_ponder(out.encoder_router)
        c_dec = self._norm_ponder(out.decoder_router)
        w_sum = self.encoder_weight + self.decoder_weight
        c = (self.encoder_weight * c_enc + self.decoder_weight * c_dec) / max(w_sum, 1e-6)

        logs["compute/encoder_norm"] = c_enc.mean().detach()
        logs["compute/decoder_norm"] = c_dec.mean().detach()
        logs["compute/combined_norm"] = c.mean().detach()

        lam = self.lambda_at(step)
        if lam <= 0.0:
            logs["compute/lambda"] = torch.tensor(0.0, device=device)
            return zero, logs

        if budget is not None:
            q = budget[:, 0]
            h = budget[:, 1]
        else:
            q = torch.zeros_like(c)
            h = torch.ones_like(c)

        scale = ((1.0 - q).clamp_min(0.0) ** self.q_exponent) + self.lambda_floor
        penalty = (lam * scale * c).mean()

        if self.use_hardware_cap:
            over = F.relu(c - h)
            penalty = penalty + self.hardware_weight * (over ** 2).mean()
            logs["compute/hardware_violation"] = over.mean().detach()

        if self.target_compute is not None and self.target_weight > 0:
            penalty = penalty + self.target_weight * ((c.mean() - float(self.target_compute)) ** 2)

        if self.diversity_weight > 0:
            # discourage a constant policy: reward spread of per-token depth
            enc_tok = out.encoder_router.ponder
            m = out.encoder_router.mask.float()
            mean = (enc_tok * m).sum() / m.sum().clamp_min(1)
            var = (((enc_tok - mean) ** 2) * m).sum() / m.sum().clamp_min(1)
            penalty = penalty - self.diversity_weight * var
            logs["compute/token_depth_var"] = var.detach()

        if c_star is not None and self.c_star_weight > 0:
            valid = torch.isfinite(c_star)
            if bool(valid.any()):
                l_cs = ((c[valid] - c_star[valid]) ** 2).mean()
                penalty = penalty + self.c_star_weight * l_cs
                logs["compute/c_star_distill"] = l_cs.detach()

        logs["compute/lambda"] = torch.tensor(lam, device=device)
        logs["compute/penalty"] = penalty.detach()
        return penalty, logs
