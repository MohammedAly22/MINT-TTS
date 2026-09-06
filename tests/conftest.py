"""Shared fixtures.

Tests build their config from `configs/base.yaml` plus tiny overrides rather
than from a dedicated test config, so they exercise the same defaults a real
run uses. Nothing here touches the disk or the network.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mint_tts.config import load_config  # noqa: E402
from mint_tts.models.tts import build_model  # noqa: E402
from mint_tts.text.tokenizer import TextProcessor  # noqa: E402

TINY_OVERRIDES = [
    "model.d_model=64",
    "model.postnet_hidden=64",
    "model.aligner_dim=32",
    "model.encoder.max_steps=4",
    "model.encoder.n_heads=2",
    "model.encoder.ff_dim=128",
    "model.encoder.router_hidden=32",
    "model.decoder.routing=token",
    "model.decoder.max_steps=3",
    "model.decoder.n_heads=2",
    "model.decoder.ff_dim=128",
    "model.decoder.router_hidden=32",
    "model.variance.hidden=64",
    "loss.compute.warmup_steps=2",
    "loss.compute.ramp_steps=2",
    "train.device=cpu",
    "train.amp=false",
]


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "base.yaml", TINY_OVERRIDES)


@pytest.fixture(scope="session")
def tp():
    """Character frontend: no espeak/nltk needed, so tests stay hermetic."""
    return TextProcessor("char")


@pytest.fixture()
def model(cfg, tp):
    torch.manual_seed(0)
    m = build_model(cfg, max(tp.vocab_size, 64)).eval()
    # Push the router off its initial "always run to the end" policy so the
    # tests actually exercise early halting.
    with torch.no_grad():
        for stack in (m.encoder, m.decoder):
            if stack.router is not None:
                torch.nn.init.normal_(stack.router.net[-1].weight, std=0.5)
                stack.router.net[-1].bias.fill_(0.4)
    return m


@pytest.fixture()
def batch(tp):
    texts = ["The record is broken by the record broker.", "Hello, how are you doing?"]
    encs = [tp.encode(t) for t in texts]
    T = max(len(e.ids) for e in encs)
    tokens = torch.zeros(len(encs), T, dtype=torch.long)
    lens = torch.tensor([len(e.ids) for e in encs])
    for i, e in enumerate(encs):
        tokens[i, : len(e.ids)] = torch.tensor(e.ids)
    return {"tokens": tokens, "token_lens": lens, "encoded": encs}
