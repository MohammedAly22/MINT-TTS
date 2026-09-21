"""The fast paths must compute exactly what the slow ones did."""

from __future__ import annotations

import pytest
import torch

from mint_tts.config import Config
from mint_tts.models.vocoder import check_vocoder_matches
from mint_tts.modules.aligner import (
    HAVE_NUMBA,
    _mas_torch,
    beta_binomial_prior,
    beta_binomial_prior_batch,
    durations_to_path,
    monotonic_alignment_search,
)
from mint_tts.utils.common import EMA, find_latest_checkpoint, save_checkpoint


def test_batched_prior_matches_per_utterance_prior():
    text_lens = torch.tensor([37, 12, 5])
    mel_lens = torch.tensor([150, 60, 9])
    batch = beta_binomial_prior_batch(text_lens, mel_lens, 40, 160)
    assert batch.shape == (3, 160, 40)
    for i in range(3):
        t, m = int(text_lens[i]), int(mel_lens[i])
        ref = beta_binomial_prior(t, m)
        assert torch.allclose(batch[i, :m, :t], ref, atol=1e-4)
        assert batch[i, m:].abs().sum() == 0      # padded frames
        assert batch[i, :, t:].abs().sum() == 0   # padded tokens


def _random_logprobs(seed: int):
    g = torch.Generator().manual_seed(seed)
    B, Tx = 5, int(torch.randint(4, 30, (1,), generator=g))
    Ty = Tx * 4
    tl = torch.randint(2, Tx + 1, (B,), generator=g)
    tl[0] = Tx
    ml = torch.clamp(tl * torch.randint(2, 5, (B,), generator=g), max=Ty)
    ml[0] = Ty
    mask = torch.arange(Tx)[None] < tl[:, None]
    x = torch.randn(B, Ty, Tx, generator=g).masked_fill(~mask[:, None], -1e9)
    return torch.log_softmax(x, -1), tl, ml, mask


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("seed", range(8))
def test_numba_mas_equals_torch_mas(seed):
    lp, tl, ml, mask = _random_logprobs(seed)
    ref = _mas_torch(lp, tl, ml)
    got = monotonic_alignment_search(lp, tl, ml, backend="numba")
    assert torch.equal(ref, got)
    dur = got.sum(1)
    assert torch.equal(dur.sum(1).long(), ml)     # every frame assigned once
    assert (dur[mask] >= 1).all()                 # every real token gets a frame


def test_durations_to_path_round_trip():
    dur = torch.tensor([[2, 1, 3, 0], [1, 1, 0, 0]])
    path = durations_to_path(dur, 7)
    assert torch.equal(path.sum(1).long(), dur)
    assert path[1, 2:].sum() == 0


def test_find_latest_checkpoint_uses_stored_step(tmp_path):
    model = torch.nn.Linear(2, 2)
    save_checkpoint(tmp_path / "best.pt", model, step=2000)
    save_checkpoint(tmp_path / "last.pt", model, step=4000)
    save_checkpoint(tmp_path / "step_3000.pt", model, step=3000)
    assert find_latest_checkpoint(tmp_path).name == "last.pt"
    assert not list(tmp_path.glob("*.tmp"))       # atomic save cleaned up
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_ema_foreach_update_matches_formula():
    model = torch.nn.Linear(3, 3)
    ema = EMA(model, decay=0.9)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for k, v in model.state_dict().items():
        assert torch.allclose(ema.shadow[k], 0.9 * before[k] + 0.1 * v)


def test_vocoder_mel_mismatch_is_refused():
    h = {"sampling_rate": 24000, "num_mels": 100, "hop_size": 256, "n_fft": 1024,
         "win_size": 1024, "fmax": None}
    good = Config({"sample_rate": 24000, "n_mels": 100, "hop_length": 256, "n_fft": 1024,
                   "win_length": 1024, "fmax": 12000.0})
    check_vocoder_matches(h, good, "bigvgan")
    bad = Config({**good.to_dict(), "n_mels": 80})
    with pytest.raises(ValueError, match="num_mels"):
        check_vocoder_matches(h, bad, "bigvgan")
