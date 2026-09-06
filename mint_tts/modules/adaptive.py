"""Adaptive-depth computation: the core of the resource-allocation formulation.

The stack below answers the question *"how many times must this representation
be transformed before it is good enough?"* rather than the usual MoE question
*"which expert should this token use?"*.

Three routing modes share one implementation so that baselines and adaptive
models differ only by a config flag (which keeps the comparison honest):

``fixed``     every token takes ``max_steps`` -- the dense baseline.
``sentence``  one halting decision for the whole utterance (Experiment 1).
``token``     per-token halting via ACT (Experiments 2-5).

Blocks may be ``shared`` (one block re-applied, Universal-Transformer style,
so depth is free in parameters) or ``independent`` (a distinct block per step,
so depth costs parameters). Comparing the two is Experiment 3.

Budget conditioning
-------------------
The router additionally sees a budget vector ``(q, h)``:

* ``q`` -- the desired quality in [0, 1], so a user can ask for
  "the cheapest computation that still reaches quality q".
* ``h`` -- a hardware/compute-availability scalar in [0, 1], so the same
  checkpoint can run shallower on a CPU than on an A100.

This is what makes the learned quantity ``C*(x, q, h)`` rather than just
``C*(x)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .transformer import TransformerBlock

ROUTING_MODES = ("fixed", "sentence", "token")


@dataclass
class RouterOutput:
    """Everything the trainer, the loggers and the FLOP counter need."""

    output: torch.Tensor                 # (B, T, D)
    ponder: torch.Tensor                 # (B, T) expected depth, continuous
    n_updates: torch.Tensor              # (B, T) executed steps
    remainders: torch.Tensor             # (B, T)
    halting_probs: torch.Tensor          # (B, N, T) per-step halting probability
    step_active: torch.Tensor            # (B, N, T) 1 where the step was executed
    mask: torch.Tensor                   # (B, T) bool, True = real token
    token_steps: float = 0.0             # sum of executed token-steps (for FLOPs)
    kv_token_steps: float = 0.0          # positions whose keys/values were projected
    attn_kv_steps: float = 0.0           # sum of query*key pairs actually scored
    hard: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def max_steps(self) -> int:
        return max(self.halting_probs.size(1), 1)

    @property
    def complexity(self) -> torch.Tensor:
        """Fraction of the maximum depth actually executed, per position.

        This is `n_updates`, not `ponder`: it is the quantity the FLOP counter
        is built from, so the heatmaps show real compute rather than the ACT
        surrogate (whose remainder term can exceed the step count).
        """
        return (self.n_updates / self.max_steps).clamp(0, 1)

    def mean_depth(self) -> torch.Tensor:
        """Mean executed steps per real position."""
        m = self.mask.float()
        return (self.n_updates * m).sum() / m.sum().clamp_min(1.0)

    def mean_ponder(self) -> torch.Tensor:
        """Mean ACT ponder cost -- the differentiable objective, not the count."""
        m = self.mask.float()
        return (self.ponder * m).sum() / m.sum().clamp_min(1.0)

    def per_utterance_ponder(self) -> torch.Tensor:
        m = self.mask.float()
        return (self.ponder * m).sum(1) / m.sum(1).clamp_min(1.0)

    def per_utterance_depth(self) -> torch.Tensor:
        m = self.mask.float()
        return (self.n_updates * m).sum(1) / m.sum(1).clamp_min(1.0)


class BudgetEncoder(nn.Module):
    """Maps the budget vector (q, h) to a halting-logit bias and FiLM params."""

    def __init__(self, d_model: int, n_budget: int = 2, hidden: int = 64, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.n_budget = n_budget
        if not enabled:
            return
        self.net = nn.Sequential(
            nn.Linear(n_budget, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.to_bias = nn.Linear(hidden, 1)
        self.to_film = nn.Linear(hidden, 2 * d_model)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, budget: torch.Tensor | None, batch: int, device, dtype):
        """Returns (bias (B,1,1), gamma (B,1,D), beta (B,1,D))."""
        if not self.enabled or budget is None:
            return None, None, None
        h = self.net(budget.to(dtype))
        bias = self.to_bias(h).unsqueeze(1)                      # (B, 1, 1)
        gamma, beta = self.to_film(h).unsqueeze(1).chunk(2, -1)  # (B, 1, D)
        return bias, gamma, beta


class ComplexityRouter(nn.Module):
    """Predicts a per-token halting probability from the current state."""

    def __init__(self, d_model: int, hidden: int = 0, init_bias: float = -1.0, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or d_model // 2
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, x: torch.Tensor, film=None) -> torch.Tensor:
        if film is not None and film[0] is not None:
            gamma, beta = film
            x = x * (1.0 + gamma) + beta
        return self.net(x).squeeze(-1)  # (B, T) logits


class AdaptiveStack(nn.Module):
    """A depth-adaptive transformer stack with ACT halting."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        max_steps: int = 8,
        min_steps: int = 1,
        routing: str = "token",
        share_weights: bool = True,
        dropout: float = 0.1,
        ffn_type: str = "linear",
        act_epsilon: float = 0.01,
        router_hidden: int = 0,
        router_init_bias: float = -1.0,
        step_embedding: bool = True,
        budget_conditioning: bool = True,
        n_budget: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        if routing not in ROUTING_MODES:
            raise ValueError(f"routing must be one of {ROUTING_MODES}, got {routing}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.max_steps = int(max_steps)
        self.min_steps = max(1, min(int(min_steps), int(max_steps)))
        self.routing = routing
        self.share_weights = bool(share_weights)
        self.act_epsilon = act_epsilon
        self.threshold = 1.0 - act_epsilon

        n_blocks = 1 if share_weights else self.max_steps
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_dim, dropout, ffn_type, activation=activation)
                for _ in range(n_blocks)
            ]
        )
        self.step_emb = nn.Embedding(self.max_steps, d_model) if step_embedding else None
        if self.step_emb is not None:
            nn.init.normal_(self.step_emb.weight, std=0.02)
        self.router = (
            ComplexityRouter(d_model, router_hidden, router_init_bias)
            if routing != "fixed"
            else None
        )
        self.budget = BudgetEncoder(d_model, n_budget, enabled=budget_conditioning and routing != "fixed")
        self.final_norm = nn.LayerNorm(d_model)

    # -- helpers ----------------------------------------------------------
    def block(self, step: int) -> TransformerBlock:
        return self.blocks[0] if self.share_weights else self.blocks[step]

    def _step_bias(self, step: int, device, dtype):
        if self.step_emb is None:
            return None
        idx = torch.tensor([step], device=device)
        return self.step_emb(idx).to(dtype).unsqueeze(0)  # (1, 1, D)

    def flops_per_token_step(self, seq_len: int) -> float:
        """Analytic FLOPs (multiply-accumulate x2) for one token, one step."""
        d, f = self.d_model, self.ff_dim
        proj = 4 * 2 * d * d               # q, k, v, out projections
        attn = 2 * 2 * d * seq_len         # scores + weighted sum over keys
        ffn = 2 * 2 * d * f
        return float(proj + attn + ffn)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        budget: torch.Tensor | None = None,
        hard: bool = False,
        max_steps_override: int | None = None,
    ) -> RouterOutput:
        """x: (B, T, D); mask: (B, T) bool (True = real token)."""
        if hard and self.routing != "fixed":
            return self._forward_hard(x, mask, budget, max_steps_override)
        return self._forward_soft(x, mask, budget, max_steps_override)

    # -- soft (differentiable) path ---------------------------------------
    def _forward_soft(self, x, mask, budget, max_steps_override):
        B, T, D = x.shape
        device, dtype = x.device, x.dtype
        N = int(max_steps_override or self.max_steps)
        N = max(1, min(N, self.max_steps))
        m = mask.to(dtype)
        bias, gamma, beta = self.budget(budget, B, device, dtype)
        film = (gamma, beta) if gamma is not None else None

        state = x
        kv_source = torch.zeros_like(x)
        halting = torch.zeros(B, T, device=device, dtype=dtype)
        remainders = torch.zeros_like(halting)
        n_updates = torch.zeros_like(halting)
        accum = torch.zeros_like(x)
        halting_probs, step_active = [], []
        token_steps = torch.zeros((), device=device, dtype=dtype)
        kv_token_steps = torch.zeros((), device=device, dtype=dtype)
        attn_kv_steps = torch.zeros((), device=device, dtype=dtype)
        n_keys = m.sum(1)  # real keys per sample, excluding padding

        for step in range(N):
            sb = self._step_bias(step, device, dtype)
            forced = step < self.min_steps

            if self.routing == "fixed" or forced:
                p = torch.ones(B, T, device=device, dtype=dtype)
            else:
                logits = self.router(state, film)
                if bias is not None:
                    logits = logits + bias.squeeze(-1)
                if self.routing == "sentence":
                    pooled = (logits * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
                    logits = pooled.expand(-1, T)
                p = torch.sigmoid(logits)

            if self.routing == "fixed":
                still_running = m
                new_halted = torch.zeros_like(m)
                update_weights = m
                n_updates = n_updates + m
            elif forced:
                still_running = m * (halting < self.threshold).to(dtype)
                new_halted = torch.zeros_like(m)
                update_weights = still_running
                n_updates = n_updates + still_running
            else:
                alive = m * (halting < self.threshold).to(dtype)
                # The final step forces a halt. Without this, a token that
                # never crosses the threshold has ponder == N exactly, which is
                # a *constant*: the router would receive no gradient at all and
                # could never be pushed towards halting earlier.
                last = step == N - 1
                will_exceed = ((halting + p * alive) > self.threshold) | last
                new_halted = will_exceed.to(dtype) * alive
                still_running = (~will_exceed).to(dtype) * alive
                halting = halting + p * still_running
                remainders = remainders + new_halted * (1.0 - halting)
                halting = halting + new_halted * remainders
                n_updates = n_updates + still_running + new_halted
                update_weights = p * still_running + new_halted * remainders

            halting_probs.append(p * m)
            active = (still_running + new_halted).clamp(0, 1)
            step_active.append(active)
            token_steps = token_steps + active.sum()
            # Count query x key pairs against the REAL sequence length. Using
            # the padded length here would report a "saving" that is only the
            # padding the model never attended to.
            attn_kv_steps = attn_kv_steps + (active.sum(1) * n_keys).sum()

            inp = state + sb if sb is not None else state
            # Freeze halted positions, both as residual state and as attention
            # keys/values. This is what makes the soft path numerically
            # identical to the gathered inference path, where a halted
            # position's keys/values simply stay in the cache untouched.
            a = active.unsqueeze(-1)
            kv_source = inp * a + kv_source * (1.0 - a)
            # Shared blocks reuse one projection at every step, so a halted
            # position's cached keys stay valid and cost nothing again.
            # Independent blocks own different weights per step, so every
            # readable position must be re-projected.
            kv_token_steps = kv_token_steps + (active.sum() if self.share_weights else m.sum())
            new_state = self.block(step)(inp, mask, kv_input=kv_source)
            state = (new_state * a + state * (1.0 - a)) * m.unsqueeze(-1)
            w = update_weights.unsqueeze(-1)
            accum = state * w + accum * (1.0 - w)

            # Early exit costs a device sync, so only take it outside training.
            if self.routing != "fixed" and not self.training and not bool(active.any()):
                for _ in range(step + 1, N):  # pad logs so shapes stay (B, N, T)
                    halting_probs.append(torch.zeros_like(p))
                    step_active.append(torch.zeros_like(p))
                break

        if self.routing == "fixed":
            remainders = torch.zeros_like(halting)
            ponder = n_updates
        else:
            ponder = n_updates + remainders
        out = self.final_norm(accum) * m.unsqueeze(-1)
        return RouterOutput(
            output=out,
            ponder=ponder * m,
            n_updates=n_updates * m,
            remainders=remainders * m,
            halting_probs=torch.stack(halting_probs, 1),
            step_active=torch.stack(step_active, 1),
            mask=mask,
            token_steps=float(token_steps.detach()),
            kv_token_steps=float(kv_token_steps.detach()),
            attn_kv_steps=float(attn_kv_steps.detach()),
            hard=False,
        )

    # -- hard (gathered) path: real FLOP savings --------------------------
    @torch.no_grad()
    def _forward_hard(self, x, mask, budget, max_steps_override):
        B, T, D = x.shape
        device, dtype = x.device, x.dtype
        N = int(max_steps_override or self.max_steps)
        N = max(1, min(N, self.max_steps))
        m = mask.to(dtype)
        bias, gamma, beta = self.budget(budget, B, device, dtype)
        film = (gamma, beta) if gamma is not None else None

        # one scratch slot at index T absorbs writes from padded active slots
        pad_slot = 1
        state = torch.cat([x, torch.zeros(B, pad_slot, D, device=device, dtype=dtype)], 1)
        accum = torch.zeros_like(state)
        k_cache = torch.zeros_like(state)
        v_cache = torch.zeros_like(state)
        kv_source = torch.zeros_like(state)
        key_mask = torch.cat([mask, torch.zeros(B, pad_slot, dtype=torch.bool, device=device)], 1)

        halting = torch.zeros(B, T, device=device, dtype=dtype)
        remainders = torch.zeros_like(halting)
        n_updates = torch.zeros_like(halting)
        halting_probs, step_active = [], []
        alive = mask.clone()
        token_steps = 0.0
        kv_token_steps = 0.0
        attn_kv_steps = 0.0
        n_real = float(mask.sum().item())
        n_keys = mask.sum(1).to(dtype)  # real keys per sample

        for step in range(N):
            if not bool(alive.any()):
                zeros = torch.zeros(B, T, device=device, dtype=dtype)
                for _ in range(step, N):
                    halting_probs.append(zeros)
                    step_active.append(zeros)
                break

            A = int(alive.sum(1).max().item())
            order = torch.argsort((~alive).to(torch.int8), dim=1, stable=True)[:, :A]
            slot_valid = alive.gather(1, order)
            active_index = torch.where(slot_valid, order, torch.full_like(order, T))

            sb = self._step_bias(step, device, dtype)
            block = self.block(step)
            idx = active_index.unsqueeze(-1).expand(-1, -1, D)
            x_a_in = state.gather(1, idx)

            forced = step < self.min_steps
            if forced:
                p_a = torch.ones(B, A, device=device, dtype=dtype)
            else:
                logits = self.router(x_a_in, film)
                if bias is not None:
                    logits = logits + bias.squeeze(-1)
                if self.routing == "sentence":
                    sv = slot_valid.to(dtype)
                    pooled = (logits * sv).sum(1, keepdim=True) / sv.sum(1, keepdim=True).clamp_min(1)
                    logits = pooled.expand(-1, A)
                p_a = torch.sigmoid(logits)

            p_full = torch.zeros(B, T + pad_slot, device=device, dtype=dtype)
            p_full = p_full.scatter(1, active_index, p_a * slot_valid.to(dtype))
            p = p_full[:, :T]

            alive_f = alive.to(dtype)
            if forced:
                still_running = alive_f
                new_halted = torch.zeros_like(alive_f)
            else:
                last = step == N - 1
                will_exceed = ((halting + p * alive_f) > self.threshold) | last
                new_halted = will_exceed.to(dtype) * alive_f
                still_running = (~will_exceed).to(dtype) * alive_f
                halting = halting + p * still_running
                remainders = remainders + new_halted * (1.0 - halting)
                halting = halting + new_halted * remainders
            n_updates = n_updates + still_running + new_halted
            update_weights = (
                still_running if forced else p * still_running + new_halted * remainders
            )

            halting_probs.append(p)
            active_now = (still_running + new_halted).clamp(0, 1)
            step_active.append(active_now)
            n_active = float(active_now.sum().item())
            token_steps += n_active
            kv_token_steps += n_active if self.share_weights else n_real
            attn_kv_steps += float((active_now.sum(1) * n_keys).sum().item())

            x_a = x_a_in if sb is None else x_a_in + sb
            kv_source = kv_source.scatter(1, idx, x_a)
            if self.share_weights:
                state, x_a_out, k_cache, v_cache = block.forward_active(
                    state, x_a, active_index, key_mask, k_cache, v_cache)
            else:
                state, x_a_out, k_cache, v_cache = block.forward_active(
                    state, x_a, active_index, key_mask, kv_source=kv_source)
            w = torch.cat([update_weights, torch.zeros(B, pad_slot, device=device, dtype=dtype)], 1)
            w = w.unsqueeze(-1)
            accum = state * w + accum * (1.0 - w)

            alive = (still_running > 0).to(torch.bool)

        ponder = n_updates + remainders
        out = self.final_norm(accum[:, :T]) * m.unsqueeze(-1)
        return RouterOutput(
            output=out,
            ponder=ponder * m,
            n_updates=n_updates * m,
            remainders=remainders * m,
            halting_probs=torch.stack(halting_probs, 1),
            step_active=torch.stack(step_active, 1),
            mask=mask,
            token_steps=token_steps,
            kv_token_steps=kv_token_steps,
            attn_kv_steps=attn_kv_steps,
            hard=True,
        )


def build_stack(cfg, d_model: int, routing: str | None = None) -> AdaptiveStack:
    """Build an AdaptiveStack from a config section."""
    return AdaptiveStack(
        d_model=d_model,
        n_heads=cfg.n_heads,
        ff_dim=cfg.ff_dim,
        max_steps=cfg.max_steps,
        min_steps=cfg.get("min_steps", 1),
        routing=routing or cfg.routing,
        share_weights=cfg.get("share_weights", True),
        dropout=cfg.get("dropout", 0.1),
        ffn_type=cfg.get("ffn_type", "linear"),
        act_epsilon=cfg.get("act_epsilon", 0.01),
        router_hidden=cfg.get("router_hidden", 0),
        router_init_bias=cfg.get("router_init_bias", -1.0),
        step_embedding=cfg.get("step_embedding", True),
        budget_conditioning=cfg.get("budget_conditioning", True),
        n_budget=cfg.get("n_budget", 2),
        activation=cfg.get("activation", "gelu"),
    )
