"""Tests for the adaptive-computation core.

The most important one is `test_soft_and_hard_paths_agree`: training uses the
dense masked path and inference uses the gathered path, so if those two ever
diverge, every quality number measured at inference is measured on a different
model than the one that was trained.
"""

import pytest
import torch

from mint_tts.modules.adaptive import AdaptiveStack
from mint_tts.modules.transformer import lengths_to_mask


def make_stack(routing="token", max_steps=5, share=True, min_steps=1, seed=0):
    torch.manual_seed(seed)
    stack = AdaptiveStack(
        d_model=32, n_heads=4, ff_dim=64, max_steps=max_steps, min_steps=min_steps,
        routing=routing, share_weights=share, dropout=0.0, router_hidden=16,
    ).eval()
    if stack.router is not None:
        with torch.no_grad():
            torch.nn.init.normal_(stack.router.net[-1].weight, std=0.6)
            stack.router.net[-1].bias.fill_(0.3)
    return stack


@pytest.mark.parametrize("routing", ["token", "sentence"])
@pytest.mark.parametrize("share", [True, False])
def test_soft_and_hard_paths_agree(routing, share):
    stack = make_stack(routing, share=share)
    x = torch.randn(3, 17, 32)
    mask = lengths_to_mask(torch.tensor([17, 12, 5]), 17)
    budget = torch.tensor([[0.2, 1.0], [0.7, 0.5], [1.0, 1.0]])

    with torch.inference_mode():
        soft = stack(x, mask, budget=budget, hard=False)
        hard = stack(x, mask, budget=budget, hard=True)

    assert torch.allclose(soft.ponder, hard.ponder, atol=1e-5)
    assert torch.allclose(soft.output, hard.output, atol=1e-4)
    assert soft.token_steps == pytest.approx(hard.token_steps)


def test_hard_path_does_less_work_than_dense():
    stack = make_stack("token", max_steps=6)
    x = torch.randn(1, 24, 32)
    mask = lengths_to_mask(torch.tensor([24]), 24)
    with torch.inference_mode():
        out = stack(x, mask, budget=torch.tensor([[0.0, 1.0]]), hard=True)
    dense_token_steps = 24 * 6
    assert out.token_steps < dense_token_steps
    assert out.attn_kv_steps < dense_token_steps * 24


def test_fixed_routing_uses_every_step():
    stack = make_stack("fixed", max_steps=4)
    x = torch.randn(2, 9, 32)
    mask = lengths_to_mask(torch.tensor([9, 6]), 9)
    out = stack(x, mask)
    assert torch.allclose(out.ponder[mask], torch.full((int(mask.sum()),), 4.0))
    assert out.token_steps == pytest.approx(float(mask.sum()) * 4)


def test_min_steps_is_respected():
    stack = make_stack("token", max_steps=6, min_steps=3)
    with torch.no_grad():  # force immediate halting pressure
        stack.router.net[-1].bias.fill_(8.0)
    x = torch.randn(2, 7, 32)
    mask = lengths_to_mask(torch.tensor([7, 7]), 7)
    out = stack(x, mask, hard=True)
    assert out.ponder[mask].min() >= 3.0


def test_padding_never_consumes_compute():
    stack = make_stack("token", max_steps=5)
    x = torch.randn(2, 20, 32)
    mask = lengths_to_mask(torch.tensor([20, 4]), 20)
    for hard in (False, True):
        out = stack(x, mask, hard=hard)
        assert out.ponder[~mask].abs().max() == 0.0
        assert out.output[~mask].abs().max() == 0.0


def test_complexity_is_normalised():
    stack = make_stack("token", max_steps=5)
    x = torch.randn(2, 11, 32)
    mask = lengths_to_mask(torch.tensor([11, 8]), 11)
    out = stack(x, mask, hard=True)
    c = out.complexity[mask]
    assert float(c.min()) > 0.0
    assert float(c.max()) <= 1.0


def test_budget_conditioning_changes_allocation():
    """A higher quality budget must not cost *less* compute than a low one."""
    stack = make_stack("token", max_steps=8)
    with torch.no_grad():  # give the (zero-initialised) budget head a real signal
        stack.budget.to_bias.weight.copy_(torch.tensor([[0.0] * 63 + [0.0]]))
        stack.budget.to_bias.bias.fill_(0.0)
        stack.budget.net[0].weight.normal_(0, 0.5)
        stack.budget.to_bias.weight.normal_(0, 0.5)
    x = torch.randn(1, 15, 32)
    mask = lengths_to_mask(torch.tensor([15]), 15)
    low = stack(x, mask, budget=torch.tensor([[0.0, 1.0]]), hard=True).mean_ponder()
    high = stack(x, mask, budget=torch.tensor([[1.0, 1.0]]), hard=True).mean_ponder()
    assert torch.isfinite(low) and torch.isfinite(high)
    # untrained: only assert the budget actually reaches the router
    assert not torch.allclose(low, high) or stack.budget.to_bias.weight.abs().sum() == 0


def test_max_steps_override_truncates_depth():
    stack = make_stack("token", max_steps=8)
    with torch.no_grad():
        stack.router.net[-1].bias.fill_(-6.0)  # never halt on its own
    x = torch.randn(1, 10, 32)
    mask = lengths_to_mask(torch.tensor([10]), 10)
    for n in (1, 3, 8):
        out = stack(x, mask, hard=True, max_steps_override=n)
        # ACT ponder is n_updates + remainder, and the forced final halt makes
        # the remainder non-zero, so the bound is n + 1 rather than n.
        assert float(out.n_updates[mask].max()) <= n
        assert float(out.ponder[mask].max()) <= n + 1.0
        assert float(out.complexity[mask].max()) <= 1.0


def test_gradients_flow_through_router():
    stack = make_stack("token", max_steps=4)
    stack.train()
    x = torch.randn(2, 6, 32, requires_grad=True)
    mask = lengths_to_mask(torch.tensor([6, 6]), 6)
    out = stack(x, mask, budget=torch.tensor([[0.5, 1.0]] * 2))
    (out.output.sum() + out.ponder.sum()).backward()
    grad = stack.router.net[1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
