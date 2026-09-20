"""Tests for the Arabic frontend, the difficulty signal and the semantic path.

These exist because every one of them corresponds to a failure that would be
invisible at training time: Arabic silently normalising to the empty string,
a word map off by one so every semantic vector lands on the wrong word, or a
semantic path that never escapes its zero initialisation.
"""

from __future__ import annotations

import torch

from mint_tts.config import Config
from mint_tts.text.arabic import (
    ARABIC_WORD_RE,
    ArabicNormalizer,
    normalise_alef,
    strip_diacritics,
    strip_tatweel,
)
from mint_tts.text.homographs_ar import (
    ARABIC_PAIRS,
    HOMOGRAPH_INDEX,
    difficulty_profile,
    is_clitic_ambiguous,
    sentence_difficulty,
    sentence_has_feminine_cue,
)
from mint_tts.text.tokenizer import build_text_processor, is_arabic_frontend

# Test strings as escapes, so the file stays readable in any editor and
# cannot be mangled by a tool that mishandles bidirectional text.
FLAG = "انا رسمت علم مصر"
SCIENCE = ("انا بحب العلم جدا "
           "و نفسي ابقا عالم")
HARD = ("عمرك فكرتي الراجل "
        "بتاع غزل البنات هسيبك "
        "تجاوبي")
EASY = "عامل ايه النهاردة؟"
ALAM = "علم"
OMRAK = "عمرك"


def ar_cfg(**text_over) -> Config:
    text = {
        "input_type": "ar_char", "lowercase": True, "add_bos_eos": True,
        "add_word_boundary": True, "add_punctuation": True,
        "allow_vocab_growth": True, "arabic": {},
    }
    text.update(text_over)
    return Config({"text": text})


# -- normalisation ---------------------------------------------------------
def test_arabic_survives_normalisation():
    """The original bug: the English pipeline folded Arabic to an empty string."""
    out = ArabicNormalizer()(FLAG)
    assert out, "Arabic text must not normalise away to nothing"
    assert ARABIC_WORD_RE.findall(out) == [
        "انا", "رسمت", ALAM, "مصر"]


def test_diacritics_are_stripped():
    """A partially diacritised corpus would otherwise leak the answer."""
    assert strip_diacritics("عَلَم") == ALAM
    assert ArabicNormalizer()("عَلَم") == ALAM


def test_tatweel_and_alef_folding():
    assert strip_tatweel("ليـــ") == "لي"
    assert normalise_alef("آدم") == "ادم"


def test_arabic_indic_digits_are_verbalised():
    out = ArabicNormalizer()("عندي ٢٥ كتاب")
    assert not any(ch.isdigit() for ch in out), "digits must be spoken, not left as digits"


def test_arabic_question_mark_maps_to_ascii():
    assert "?" in ArabicNormalizer()(EASY)


def test_code_switched_latin_survives():
    out = ArabicNormalizer()("اللاب laptop")
    assert "laptop" in out, "code-switched English must reach the model"


def test_emoji_are_removed():
    out = ArabicNormalizer()(FLAG + " " + chr(0x1F600))
    assert chr(0x1F600) not in out


def test_ta_marbuta_preserved_by_default():
    """It carries feminine morphology; folding it would destroy evidence."""
    assert "ة" in ArabicNormalizer()("مدرسة")
    assert "ة" not in ArabicNormalizer(normalise_ta_marbuta=True)("مدرسة")


def test_trace_reports_every_step():
    steps = ArabicNormalizer().trace(FLAG)
    assert steps[0][0] == "input" and len(steps) > 3


# -- tokenisation ----------------------------------------------------------
def test_tokenizer_builds_a_consistent_word_map():
    """If this drifts, every semantic vector lands on the wrong word."""
    tp = build_text_processor(ar_cfg())
    enc = tp.encode(HARD)
    assert len(enc.ids) == len(enc.tokens) == len(enc.word_ids)
    assert max(enc.word_ids) < len(enc.words)
    assert OMRAK in enc.words


def test_every_character_maps_to_its_own_word():
    tp = build_text_processor(ar_cfg())
    enc = tp.encode(FLAG)
    for wi, word in enumerate(enc.words):
        chars = [t for t, w in zip(enc.tokens, enc.word_ids)
                 if w == wi and len(t) == 1 and t.isalpha()]
        assert "".join(chars) == word


def test_arabic_frontend_selects_the_arabic_normalizer():
    assert is_arabic_frontend("ar_char") and not is_arabic_frontend("char")
    tp = build_text_processor(ar_cfg())
    assert isinstance(tp.normalizer, ArabicNormalizer)


def test_english_frontend_is_unaffected():
    """The Arabic work must not change the existing English path."""
    cfg = Config({"text": {"input_type": "char", "lowercase": True}})
    tp = build_text_processor(cfg)
    enc = tp.encode("The record is broken.")
    assert enc.words == ["the", "record", "is", "broken"]


# -- difficulty ------------------------------------------------------------
def test_known_homograph_is_maximally_difficult():
    assert difficulty_profile([ALAM]) == [1.0]
    assert ALAM in HOMOGRAPH_INDEX


def test_easy_sentence_scores_zero():
    """`How are you today` must be cheap, or compute is not tracking difficulty."""
    words = ARABIC_WORD_RE.findall(ArabicNormalizer()(EASY))
    assert sentence_difficulty(words) == 0.0


def test_hard_sentence_flags_the_clitics():
    words = ARABIC_WORD_RE.findall(ArabicNormalizer()(HARD))
    profile = dict(zip(words, difficulty_profile(words)))
    assert profile[OMRAK] > 0.5
    assert profile["هسيبك"] > 0.5
    assert sentence_difficulty(words) > 0.0


def test_feminine_cue_is_detected_at_a_distance():
    """The cue sits several words from the clitic -- that is the whole point."""
    words = ARABIC_WORD_RE.findall(ArabicNormalizer()(HARD))
    assert sentence_has_feminine_cue(words)


def test_clitic_detection_rejects_short_words():
    assert not is_clitic_ambiguous("ملك")
    assert is_clitic_ambiguous(OMRAK)


def test_every_pair_contains_its_target_word():
    """A pair whose word is absent is silently skipped by the probe."""
    norm = ArabicNormalizer()
    for pair in ARABIC_PAIRS:
        for sentence in (pair.a, pair.b):
            words = ARABIC_WORD_RE.findall(norm(sentence))
            joined = " ".join(words)
            assert pair.word in words or pair.word in joined, (
                f"{pair.word!r} missing from {sentence!r}")


# -- the semantic path -----------------------------------------------------
def _tiny_model(semantic=True, reference=False):
    from mint_tts.config import load_config
    from mint_tts.models.tts import build_model

    cfg = load_config("configs/base.yaml")
    cfg.model.d_model = 64
    for st in (cfg.model.encoder, cfg.model.decoder):
        st.n_heads, st.ff_dim, st.max_steps = 4, 128, 3
    cfg.model.semantic = {"enabled": semantic, "hidden_size": 32,
                          "proj_hidden": 64, "to_router": True}
    cfg.model.reference_encoder = {"enabled": reference}
    return build_model(cfg, vocab_size=40)


def _batch(B=2, T=10, W=3, F=32, H=32):
    return dict(
        tokens=torch.randint(1, 40, (B, T)),
        token_lens=torch.tensor([T] * B),
        mels=torch.randn(B, 80, F),
        mel_lens=torch.tensor([F] * B),
        budget=torch.rand(B, 2),
        semantic=torch.randn(B, W, H),
        word_index=torch.randint(0, W, (B, T)),
    )


def test_semantic_path_is_a_noop_at_initialisation():
    """Zero-init means enabling semantics cannot perturb a fresh model.

    That is what keeps the comparison against the no-semantics control
    honest: any difference must be learned, not an initialisation artefact.
    """
    model = _tiny_model(semantic=True).eval()
    b = _batch()
    with torch.inference_mode():
        a = model(b["tokens"], b["token_lens"], budget=b["budget"],
                  semantic=b["semantic"], word_index=b["word_index"])
        z = model(b["tokens"], b["token_lens"], budget=b["budget"],
                  semantic=torch.zeros_like(b["semantic"]), word_index=b["word_index"])
    assert torch.allclose(a.mel_post, z.mel_post, atol=1e-6)


def test_semantic_path_becomes_active_after_training():
    """A permanently dead path would defeat the entire design silently."""
    model = _tiny_model(semantic=True)
    b = _batch()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(3):
        out = model(b["tokens"], b["token_lens"], mels=b["mels"], mel_lens=b["mel_lens"],
                    budget=b["budget"], semantic=b["semantic"], word_index=b["word_index"])
        out.mel_post.pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    assert float(model.semantic_adapter.to_add.weight.norm()) > 0
    assert float(out.semantic_delta_norm) > 0


def test_model_runs_without_semantics():
    """The English configs pass no semantics at all; that must still work."""
    model = _tiny_model(semantic=False).eval()
    b = _batch()
    with torch.inference_mode():
        out = model(b["tokens"], b["token_lens"], budget=b["budget"])
    assert out.mel_post.shape[0] == 2 and out.semantic_delta_norm is None


def test_hard_routing_path_accepts_semantics():
    """The gathered inference path gathers context too; shapes must line up."""
    model = _tiny_model(semantic=True).eval()
    b = _batch()
    with torch.inference_mode():
        out = model(b["tokens"], b["token_lens"], budget=b["budget"], hard=True,
                    semantic=b["semantic"], word_index=b["word_index"])
    assert out.mel_post.shape[0] == 2


def test_reference_encoder_produces_a_speaker_vector():
    model = _tiny_model(semantic=False, reference=True).eval()
    b = _batch()
    with torch.inference_mode():
        out = model(b["tokens"], b["token_lens"], budget=b["budget"],
                    reference_mel=b["mels"], reference_lens=b["mel_lens"])
    assert out.speaker_vector.shape == (2, 64)


def test_reference_encoder_falls_back_to_the_unknown_token():
    """Synthesis with no reference must still work."""
    model = _tiny_model(semantic=False, reference=True).eval()
    b = _batch()
    with torch.inference_mode():
        out = model(b["tokens"], b["token_lens"], budget=b["budget"])
    assert out.speaker_vector is not None and out.speaker_vector.shape == (2, 64)


def test_reference_vector_differs_between_voices():
    model = _tiny_model(semantic=False, reference=True).eval()
    b = _batch()
    other = torch.randn_like(b["mels"]) * 3.0
    with torch.inference_mode():
        v1 = model.encode_speaker(b["mels"], b["mel_lens"])
        v2 = model.encode_speaker(other, b["mel_lens"])
    assert not torch.allclose(v1, v2, atol=1e-4)


# -- difficulty-aware compute loss ----------------------------------------
def test_difficulty_relief_discounts_hard_tokens():
    """An ambiguous token must cost less per step than an ordinary one."""
    from mint_tts.config import load_config
    from mint_tts.losses.compute import ComputeLoss

    cfg = load_config("configs/base.yaml")
    cfg.loss.compute.difficulty_relief = 0.75
    cfg.loss.compute.warmup_steps = 0
    loss = ComputeLoss(cfg)

    model = _tiny_model(semantic=False)
    b = _batch()
    out = model(b["tokens"], b["token_lens"], mels=b["mels"], mel_lens=b["mel_lens"],
                budget=b["budget"])
    T = out.encoder_router.mask.shape[1]
    easy = torch.zeros(2, T)
    hard = torch.ones(2, T)
    p_easy, _ = loss(out, b["budget"], step=10_000, difficulty=easy)
    p_hard, _ = loss(out, b["budget"], step=10_000, difficulty=hard)
    assert float(p_hard) < float(p_easy), (
        "a fully ambiguous utterance must be penalised less than a trivial one")


def test_difficulty_contrast_is_logged():
    from mint_tts.config import load_config
    from mint_tts.losses.compute import ComputeLoss

    cfg = load_config("configs/base.yaml")
    cfg.loss.compute.difficulty_relief = 0.75
    cfg.loss.compute.warmup_steps = 0
    loss = ComputeLoss(cfg)
    model = _tiny_model(semantic=False)
    b = _batch()
    out = model(b["tokens"], b["token_lens"], mels=b["mels"], mel_lens=b["mel_lens"],
                budget=b["budget"])
    T = out.encoder_router.mask.shape[1]
    d = torch.zeros(2, T)
    d[:, :3] = 1.0
    _, logs = loss(out, b["budget"], step=10_000, difficulty=d)
    assert "compute/difficulty_contrast" in logs
    assert "compute/hard_token_depth" in logs


def test_compute_loss_unchanged_without_difficulty():
    """Existing English runs pass no difficulty and must behave as before."""
    from mint_tts.config import load_config
    from mint_tts.losses.compute import ComputeLoss

    cfg = load_config("configs/base.yaml")
    cfg.loss.compute.warmup_steps = 0
    loss = ComputeLoss(cfg)
    model = _tiny_model(semantic=False)
    b = _batch()
    out = model(b["tokens"], b["token_lens"], mels=b["mels"], mel_lens=b["mel_lens"],
                budget=b["budget"])
    penalty, logs = loss(out, b["budget"], step=10_000)
    assert torch.isfinite(penalty)
    assert "compute/difficulty_contrast" not in logs


# -- configs ---------------------------------------------------------------
def test_egyptian_configs_load_and_build():
    from mint_tts.config import load_config
    from mint_tts.models.tts import build_model

    for name in ("egyptian_homograph", "egyptian_nosemantic",
                 "egyptian_dense", "egyptian_homograph_small"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.text.input_type == "ar_char"
        assert build_model(cfg, vocab_size=100) is not None


def test_egyptian_config_uses_the_arabic_probe_sets():
    from mint_tts.config import load_config
    from mint_tts.training.homograph import load_pairs
    from mint_tts.training.monitors import load_probe_sentences

    cfg = load_config("configs/egyptian_homograph.yaml")
    pairs = load_pairs(cfg)
    assert pairs and any(p.word == OMRAK for p in pairs)
    sentences = load_probe_sentences(cfg)
    groups = {s.group for s in sentences}
    # All four control groups must be present, or the result cannot be
    # distinguished from "the router learned sentence length".
    assert "easy" in groups and "long_easy" in groups and "tongue_twister" in groups
    assert any(g.startswith("homograph") for g in groups)
