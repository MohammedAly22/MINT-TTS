"""Pieces of the two-stage (dense -> routed) training scheme."""

from __future__ import annotations

import torch

from mint_tts.utils.common import EMA, save_checkpoint, warm_start_weights


def test_full_depth_path_honours_depth_override(model):
    """Random-depth training truncates the whole encoder to one depth."""
    enc = model.encoder
    x = torch.randn(2, 7, model.d_model)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    full = enc(x, mask, force_full_depth=True)
    short = enc(x, mask, force_full_depth=True, max_steps_override=3)
    assert full.halting_probs.size(1) == enc.max_steps
    assert short.halting_probs.size(1) == 3
    assert float(short.token_steps) == float(mask.sum()) * 3
    assert not torch.allclose(full.output, short.output)


def test_warm_start_prefers_ema_weights(tmp_path):
    net = torch.nn.Linear(3, 3)
    ema = EMA(net, decay=0.5)
    with torch.no_grad():
        net.weight.add_(1.0)
    ema.update(net)                       # shadow now differs from the live weights
    save_checkpoint(tmp_path / "c.pt", net, ema=ema, step=5)
    state = warm_start_weights(tmp_path / "c.pt")
    assert torch.allclose(state["weight"], ema.shadow["weight"])
    assert not torch.allclose(state["weight"], net.weight)
    raw = warm_start_weights(tmp_path / "c.pt", prefer_ema=False)
    assert torch.allclose(raw["weight"], net.weight)
