"""Tests for the Arabic frontend, learned ambiguity, and the semantic path.

Each of these corresponds to a failure that would otherwise be invisible at
training time: Arabic normalising to the empty string, MSA number words
mismatching Egyptian audio, a word map off by one so every semantic vector
lands on the wrong word, a semantic path that never leaves its zero
initialisation, or an ambiguity measure that flags noise.

There is deliberately no test asserting that any *particular* word is a
homograph. Which words are ambiguous is measured from the corpus, so the
tests check the measurement instead -- on synthetic data where the ground
truth is known by construction.
"""

from __future__ import annotations

import numpy as np
import torch

from mint_tts.config import Config
from mint_tts.text.ambiguity import (
    AmbiguityMiner,
    AmbiguityTable,
    ContextualAmbiguity,
)
from mint_tts.text.arabic import (
    ARABIC_WORD_RE,
    ArabicNormalizer,
    normalise_alef,
    strip_diacritics,
    strip_tatweel,
)
from mint_tts.text.homographs_ar import (
    clitic_gender_ambiguity,
    cue_distance,
    is_feminine_verb,
    sentence_has_feminine_cue,
    structural_difficulty,
)
from mint_tts.text.numbers_ar import decimal_to_words, number_to_words
from mint_tts.text.tokenizer import build_text_processor, is_arabic_frontend

# Escapes, so the file is readable in any editor and cannot be mangled by a
# tool that mishandles bidirectional text.
FLAG = "انا رسمت علم مصر"
HARD = ("عمرك فكرتي الراجل "
        "بتاع غزل البنات هسيبك "
        "تجاوبي")
EASY = "عامل ايه النهاردة؟"
ALAM = "علم"
OMRAK = "عمرك"
FAKKARTI = "فكرتي"


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
    assert strip_diacritics("عَلَم") == ALAM
    assert ArabicNormalizer()("عَلَم") == ALAM


def test_tatweel_and_alef_folding():
    assert strip_tatweel("ليـــ") == "لي"
    assert normalise_alef("آدم") == "ادم"


def test_question_mark_is_preserved_as_a_token():
    """Arabic '?' must survive to the model: it carries the intonation.

    It does NOT appear in a word list, because a word regex matches words --
    that is why this asserts on the tokens rather than on ARABIC_WORD_RE.
    """
    assert "?" in ArabicNormalizer()(EASY)
    enc = build_text_processor(ar_cfg()).encode(EASY)
    assert "?" in enc.tokens


def test_code_switched_latin_survives():
    out = ArabicNormalizer()("اللاب laptop")
    assert "laptop" in out


def test_emoji_are_removed():
    assert chr(0x1F600) not in ArabicNormalizer()(FLAG + " " + chr(0x1F600))


def test_ta_marbuta_preserved_by_default():
    """It carries feminine morphology; folding it would destroy evidence."""
    assert "ة" in ArabicNormalizer()("مدرسة")
    assert "ة" not in ArabicNormalizer(normalise_ta_marbuta=True)("مدرسة")


# -- Egyptian numbers ------------------------------------------------------
def test_numbers_are_egyptian_not_msa():
    """num2words emits MSA nominative, which this corpus never says.

    Writing `3ishruun` into a transcript whose audio says `3ishriin` trains
    the aligner on a text/audio mismatch.
    """
    # 20 -> 3ishriin (oblique), NOT 3ishruun (MSA nominative)
    assert number_to_words(20) == "عشرين"
    # 2 -> itnein, not ithnaan
    assert number_to_words(2) == "اتنين"
    # 3 -> talaata (Egyptian /t/), not thalaatha (MSA interdental)
    assert number_to_words(3) == "تلاتة"
    # 100 -> miyya, not mi'a
    assert number_to_words(100) == "مية"


def test_number_agreement_for_scale_words():
    """Arabic's three-way agreement: 1 singular, 2 dual, 3-10 plural, 11+ singular."""
    alf = "الف"
    assert number_to_words(1000) == alf
    assert number_to_words(2000) == "الفين"          # dual
    assert number_to_words(3000).endswith("الاف")          # plural
    assert number_to_words(11000).endswith(alf)                                # back to singular


def test_unit_precedes_tens():
    """Egyptian says 'five and twenty', not 'twenty five'."""
    words = number_to_words(25).split()
    assert words[0] == "خمسة"          # khamsa first
    assert words[-1] == "عشرين"   # 3ishriin last


def test_normalizer_verbalises_arabic_indic_digits():
    out = ArabicNormalizer()("عندي ٢٥ كتاب")
    assert not any(ch.isdigit() for ch in out)
    assert "عشرين" in out      # 3ishriin, the Egyptian form
    assert "عشرون" not in out  # not the MSA 3ishruun


def test_decimals_read_digit_by_digit():
    out = decimal_to_words("12.5")
    assert "فاصلة" in out       # faasla
    assert out.endswith("خمسة")      # khamsa


def test_zero_and_negative():
    assert number_to_words(0) == "صفر"
    assert number_to_words(-5).startswith("ناقص")


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
    assert isinstance(build_text_processor(ar_cfg()).normalizer, ArabicNormalizer)


def test_english_frontend_is_unaffected():
    """The Arabic work must not change the existing English path."""
    tp = build_text_processor(Config({"text": {"input_type": "char", "lowercase": True}}))
    assert tp.encode("The record is broken.").words == ["the", "record", "is", "broken"]


# -- structural (orthographic) signal, not a word list ---------------------
def test_clitic_detection_is_structural():
    """A property of the script: any long word ending in kaf hides a vowel."""
    assert clitic_gender_ambiguity(OMRAK)
    assert clitic_gender_ambiguity("هسيبك")
    assert not clitic_gender_ambiguity("ملك")   # root-final kaf
    assert not clitic_gender_ambiguity("ك")               # too short


def test_feminine_verb_detection():
    assert is_feminine_verb(FAKKARTI)
    assert not is_feminine_verb("كتب")


def test_feminine_cue_is_found_at_a_distance():
    """The cue sits several words from the clitic -- that is the whole point."""
    words = ARABIC_WORD_RE.findall(ArabicNormalizer()(HARD))
    assert sentence_has_feminine_cue(words)
    assert cue_distance(words) >= 1


def test_structural_prior_flags_only_clitics():
    words = ARABIC_WORD_RE.findall(ArabicNormalizer()(HARD))
    prof = dict(zip(words, structural_difficulty(words)))
    assert prof[OMRAK] > 0
    assert prof[FAKKARTI] == 0.0


def test_easy_sentence_has_no_structural_difficulty():
    words = ARABIC_WORD_RE.findall(ArabicNormalizer()(EASY))
    assert sum(structural_difficulty(words)) == 0.0


# -- learned ambiguity: the replacement for the word list ------------------
def _synthetic_corpus(seed: int = 0, dim: int = 24):
    """Occurrences with known ground truth.

    homograph  two acoustic clusters, PREDICTED by context   -> must score high
    noisy      two acoustic clusters, context unrelated       -> must score ~0
    stable     one pronunciation, varied context              -> must score 0
    """
    rng = np.random.default_rng(seed)
    miner = AmbiguityMiner(min_count=6)
    ac0, ac1 = rng.standard_normal(dim), rng.standard_normal(dim)
    cx0, cx1 = rng.standard_normal(dim), rng.standard_normal(dim)
    for i in range(30):
        k = i % 2
        miner.add("homograph", (ac1 if k else ac0) + 0.15 * rng.standard_normal(dim),
                  (cx1 if k else cx0) + 0.15 * rng.standard_normal(dim), "s")
        miner.add("noisy", (ac1 if k else ac0) + 0.15 * rng.standard_normal(dim),
                  rng.standard_normal(dim), "s")
    base = rng.standard_normal(dim)
    for _ in range(30):
        miner.add("stable", base + 0.05 * rng.standard_normal(dim),
                  rng.standard_normal(dim), "s")
    for _ in range(3):
        miner.add("rare", rng.standard_normal(dim), rng.standard_normal(dim), "s")
    return miner.finalise()


def test_miner_finds_the_context_predictable_homograph():
    stats = _synthetic_corpus()
    assert stats["homograph"].ambiguity > 0.5


def test_miner_rejects_unpredictable_variation():
    """Acoustic variation that context cannot explain is noise, not ambiguity.

    This is the measurement that makes the whole approach work: without it,
    any word appearing in varied prosodic positions would be flagged.
    """
    stats = _synthetic_corpus()
    assert stats["noisy"].ambiguity < stats["homograph"].ambiguity / 2
    assert stats["noisy"].context_predictivity < 0.3


def test_miner_rejects_stable_words():
    stats = _synthetic_corpus()
    assert stats["stable"].ambiguity < 0.1


def test_miner_skips_rare_words():
    """Too few occurrences to measure anything: better silent than wrong."""
    assert "rare" not in _synthetic_corpus()


def test_ambiguity_table_roundtrip(tmp_path):
    stats = _synthetic_corpus()
    table = AmbiguityTable.from_stats(stats)
    path = table.save(tmp_path / "ambiguity.json", stats)
    reloaded = AmbiguityTable.load(path)
    assert reloaded.score("homograph") == table.score("homograph")
    assert reloaded.top(1)[0][0] == "homograph"


def test_unseen_word_scores_zero_not_an_error():
    """A word never mined is ordinary, not a crash: the score only relaxes
    a penalty, so being wrong about a rare word costs efficiency, not
    correctness."""
    assert AmbiguityTable.from_stats(_synthetic_corpus()).score("كلمة") == 0.0


def _bootstrap_stats(dim=768, seed=1):
    """Text-only bootstrap on vectors of realistic LM dimensionality."""
    rng = np.random.default_rng(seed)
    boot = ContextualAmbiguity(min_count=6)
    a, b = rng.standard_normal(dim), rng.standard_normal(dim)
    for i in range(16):
        boot.add("two_senses", (a if i % 2 else b) + 0.5 * rng.standard_normal(dim))
        boot.add("one_sense", a + 0.5 * rng.standard_normal(dim))
        boot.add("pure_noise", rng.standard_normal(dim))
    return boot.finalise()


def test_text_only_bootstrap_runs_without_audio():
    """The pre-alignment fallback must work from LM vectors alone."""
    stats = _bootstrap_stats()
    assert stats["two_senses"].ambiguity > stats["one_sense"].ambiguity


def test_bootstrap_rejects_structureless_vectors():
    """The failure this guards against is subtle and was actually hit.

    In 768 dimensions random points are nearly orthogonal, so after centring
    2-means reports a separation of ~1.999 on a SINGLE structureless Gaussian
    blob -- at every sample size tested. Scoring on separation alone therefore
    marked every word maximally ambiguous. The score is now calibrated against
    a dimension-shuffled null, so a blob must come out near zero.
    """
    stats = _bootstrap_stats()
    assert stats["pure_noise"].ambiguity < 0.2
    assert stats["one_sense"].ambiguity < 0.2


def test_two_means_separation_alone_is_not_evidence():
    """Document the artefact directly, so nobody reintroduces it.

    If this ever fails -- i.e. 2-means stops splitting structureless
    high-dimensional data -- the null calibration could be simplified.
    """
    from mint_tts.text.ambiguity import _centre, _two_means

    rng = np.random.default_rng(0)
    _, separation = _two_means(_centre(rng.standard_normal((20, 768))), rng=rng)
    assert separation > 1.5, (
        "a structureless blob still yields a large 2-means separation; "
        "raw separation must not be used as an ambiguity score")


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
    """Zero-init means enabling semantics cannot perturb a fresh model, which
    is what keeps the comparison against the no-semantics control honest."""
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
    model = _tiny_model(semantic=False, reference=True).eval()
    b = _batch()
    with torch.inference_mode():
        out = model(b["tokens"], b["token_lens"], budget=b["budget"])
    assert out.speaker_vector is not None and out.speaker_vector.shape == (2, 64)


def test_reference_vector_differs_between_voices():
    model = _tiny_model(semantic=False, reference=True).eval()
    b = _batch()
    with torch.inference_mode():
        v1 = model.encode_speaker(b["mels"], b["mel_lens"])
        v2 = model.encode_speaker(torch.randn_like(b["mels"]) * 3.0, b["mel_lens"])
    assert not torch.allclose(v1, v2, atol=1e-4)


# -- difficulty-aware compute loss ----------------------------------------
def _loss_and_out(relief=0.75):
    from mint_tts.config import load_config
    from mint_tts.losses.compute import ComputeLoss

    cfg = load_config("configs/base.yaml")
    cfg.loss.compute.difficulty_relief = relief
    cfg.loss.compute.warmup_steps = 0
    model = _tiny_model(semantic=False)
    b = _batch()
    out = model(b["tokens"], b["token_lens"], mels=b["mels"], mel_lens=b["mel_lens"],
                budget=b["budget"])
    return ComputeLoss(cfg), out, b


def test_difficulty_relief_discounts_hard_tokens():
    """An ambiguous token must cost less per step than an ordinary one."""
    loss, out, b = _loss_and_out()
    T = out.encoder_router.mask.shape[1]
    p_easy, _ = loss(out, b["budget"], 10_000, difficulty=torch.zeros(2, T))
    p_hard, _ = loss(out, b["budget"], 10_000, difficulty=torch.ones(2, T))
    assert float(p_hard) < float(p_easy)


def test_difficulty_contrast_is_logged():
    loss, out, b = _loss_and_out()
    T = out.encoder_router.mask.shape[1]
    d = torch.zeros(2, T)
    d[:, :3] = 1.0
    _, logs = loss(out, b["budget"], 10_000, difficulty=d)
    assert "compute/difficulty_contrast" in logs
    assert "compute/hard_token_depth" in logs


def test_compute_loss_unchanged_without_difficulty():
    """Existing English runs pass no difficulty and must behave as before."""
    loss, out, b = _loss_and_out(relief=0.0)
    penalty, logs = loss(out, b["budget"], 10_000)
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


def test_probe_sets_are_empty_without_mining(tmp_path):
    """Probes are mined, so they must be ABSENT rather than invented when the
    mining pass has not run. A silently substituted hand-written set is what
    this design is meant to avoid."""
    from mint_tts.config import load_config
    from mint_tts.training.homograph import load_pairs
    from mint_tts.training.monitors import load_probe_sentences

    cfg = load_config("configs/egyptian_homograph.yaml")
    cfg.data.preprocessed_dir = str(tmp_path)   # no ambiguity.json here
    assert load_probe_sentences(cfg) == []
    assert load_pairs(cfg) == []


def test_probe_sets_are_built_from_mined_scores(tmp_path):
    """Given a mined table and a corpus index, probes come from real rows."""
    import json

    from mint_tts.config import load_config
    from mint_tts.training.monitors import load_probe_sentences

    rows = []
    for i in range(6):
        words = [ALAM, "مصر", f"w{i}", "كتاب"]
        rows.append({"uid": f"u{i}", "words": words,
                     "clean_text": " ".join(words), "word_ids": [0, 1, 2, 3],
                     "tokens": [1, 2, 3, 4], "n_frames": 50, "n_tokens": 4})
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    AmbiguityTable({ALAM: 0.9, "مصر": 0.0}).save(tmp_path / "ambiguity.json")

    cfg = load_config("configs/egyptian_homograph.yaml")
    cfg.data.preprocessed_dir = str(tmp_path)
    sentences = load_probe_sentences(cfg)
    assert sentences, "probe set should be built from the mined table"
    assert any(ALAM in s.ambiguous_words for s in sentences)


def test_no_homograph_word_list_remains():
    """Guard against the old design creeping back.

    The point of `text/ambiguity.py` is that ambiguity is measured. If a
    hand-written list of homographs reappears, this fails.
    """
    import mint_tts.text.homographs_ar as mod

    for banned in ("HOMOGRAPHS", "HOMOGRAPH_INDEX", "ARABIC_PAIRS",
                   "TONGUE_TWISTERS", "difficulty_profile"):
        assert not hasattr(mod, banned), (
            f"{banned} is back: ambiguity must be mined, not declared")
