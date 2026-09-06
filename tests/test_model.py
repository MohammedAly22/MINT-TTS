"""Model-level tests: shapes, alignment, duration accounting, FLOPs."""

import pytest
import torch

from mint_tts.losses.compute import ComputeLoss, sample_budget
from mint_tts.losses.tts_losses import TTSLoss
from mint_tts.modules.aligner import beta_binomial_prior, monotonic_alignment_search
from mint_tts.modules.variance import average_by_duration, length_regulate


def make_training_batch(tp, n_mels=80, seed=0):
    torch.manual_seed(seed)
    texts = ["The record is broken by the record broker.", "Hello, how are you doing?"]
    encs = [tp.encode(t) for t in texts]
    T = max(len(e.ids) for e in encs)
    lens = torch.tensor([len(e.ids) for e in encs])
    tokens = torch.zeros(2, T, dtype=torch.long)
    for i, e in enumerate(encs):
        tokens[i, : len(e.ids)] = torch.tensor(e.ids)
    mel_lens = torch.tensor([96, 64])
    L = int(mel_lens.max())
    mel = torch.randn(2, n_mels, L) * 0.5 - 5.0
    prior = torch.zeros(2, L, T)
    for i in range(2):
        prior[i, : mel_lens[i], : lens[i]] = beta_binomial_prior(int(lens[i]), int(mel_lens[i]))
    return {
        "tokens": tokens, "token_lens": lens, "mel": mel, "mel_lens": mel_lens,
        "pitch": torch.randn(2, L), "energy": torch.randn(2, L), "attn_prior": prior,
        "text": texts, "token_strings": [e.tokens for e in encs],
    }


def test_training_forward_shapes(model, tp):
    b = make_training_batch(tp)
    out = model(b["tokens"], b["token_lens"], mels=b["mel"], mel_lens=b["mel_lens"],
                pitch=b["pitch"], energy=b["energy"], attn_prior=b["attn_prior"],
                budget=torch.rand(2, 2))
    assert out.mel.shape == b["mel"].shape
    assert out.mel_post.shape == b["mel"].shape
    assert out.attn_hard.shape == (2, b["mel"].shape[-1], b["tokens"].shape[1])


def test_durations_cover_every_frame(model, tp):
    """MAS must produce a surjective path: durations sum to the mel length."""
    b = make_training_batch(tp)
    out = model(b["tokens"], b["token_lens"], mels=b["mel"], mel_lens=b["mel_lens"],
                attn_prior=b["attn_prior"])
    assert torch.allclose(out.duration_target.sum(1), b["mel_lens"].float())


def test_inference_produces_audio_length_from_durations(model, tp):
    enc = tp.encode("Hello, how are you doing?")
    tokens = torch.tensor(enc.ids).unsqueeze(0)
    lens = torch.tensor([len(enc.ids)])
    out = model(tokens, lens, budget=torch.tensor([[0.9, 1.0]]), hard=True)
    assert int(out.mel_mask.sum()) == int(out.duration_rounded.sum())
    assert int(out.mel_mask.sum()) >= len(enc.ids)  # min_duration floor


def test_flops_report_is_consistent(model, tp):
    enc = tp.encode("The record is broken by the record broker.")
    tokens = torch.tensor(enc.ids).unsqueeze(0)
    lens = torch.tensor([len(enc.ids)])
    out = model(tokens, lens, budget=torch.tensor([[0.1, 1.0]]), hard=True)
    rep = model.flops(out)
    assert rep.total > 0
    assert rep.dense_equivalent >= rep.total
    assert 0.0 <= rep.saving < 1.0
    assert pytest.approx(rep.total, rel=1e-6) == sum(rep.parts.values())


def test_losses_are_finite(cfg, model, tp):
    b = make_training_batch(tp)
    budget = sample_budget(2, torch.device("cpu"))
    out = model(b["tokens"], b["token_lens"], mels=b["mel"], mel_lens=b["mel_lens"],
                pitch=b["pitch"], energy=b["energy"], attn_prior=b["attn_prior"],
                budget=budget)
    loss_fn, comp_fn = TTSLoss(cfg), ComputeLoss(cfg)
    recon, logs = loss_fn(out, b, step=10_000)
    comp, clogs = comp_fn(out, budget, step=10_000)
    assert torch.isfinite(recon) and torch.isfinite(comp)
    assert all(torch.isfinite(torch.as_tensor(v)) for v in {**logs, **clogs}.values())
    (recon + comp).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_compute_penalty_is_zero_during_warmup(cfg, model, tp):
    b = make_training_batch(tp)
    budget = sample_budget(2, torch.device("cpu"))
    out = model(b["tokens"], b["token_lens"], mels=b["mel"], mel_lens=b["mel_lens"],
                attn_prior=b["attn_prior"], budget=budget)
    comp, _ = ComputeLoss(cfg)(out, budget, step=0)
    assert float(comp) == 0.0


def test_length_regulator_matches_repeat_interleave():
    x = torch.randn(2, 5, 7)
    dur = torch.tensor([[2, 0, 3, 1, 1], [1, 1, 1, 1, 1]])
    out, mask = length_regulate(x, dur)
    ref = torch.repeat_interleave(x[0], dur[0], dim=0)
    assert torch.allclose(out[0, : ref.shape[0]], ref)
    assert mask.sum(1).tolist() == dur.sum(1).tolist()


def test_average_by_duration_inverts_expansion():
    values = torch.tensor([[1.0, 1.0, 3.0, 3.0, 3.0, 5.0]])
    dur = torch.tensor([[2, 3, 1]])
    assert torch.allclose(average_by_duration(values, dur), torch.tensor([[1.0, 3.0, 5.0]]))


def test_mas_path_is_monotonic():
    torch.manual_seed(0)
    neg = torch.log(torch.rand(2, 40, 9) + 1e-3)
    path = monotonic_alignment_search(neg, torch.tensor([9, 6]), torch.tensor([40, 25]))
    for i, (tl, ml) in enumerate([(9, 40), (6, 25)]):
        idx = path[i, :ml].argmax(-1)
        assert torch.all(idx[1:] - idx[:-1] >= 0)      # never goes backwards
        assert torch.all(idx[1:] - idx[:-1] <= 1)      # never skips a token
        assert int(idx[0]) == 0 and int(idx[-1]) == tl - 1
        assert path[i, ml:].sum() == 0
