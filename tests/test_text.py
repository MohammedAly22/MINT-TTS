"""Text frontend: normalisation, phonemisation, tokenisation, symbol tables."""

import pytest

from mint_tts.text.normalizer import TextNormalizer
from mint_tts.text.phonemes import build_phonemizer
from mint_tts.text.symbols import PUNCTUATION, SPECIALS
from mint_tts.text.tokenizer import SymbolTable, TextProcessor


@pytest.fixture(scope="module")
def norm():
    return TextNormalizer()


# -- normalisation -----------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("I paid $5.", "five dollars"),
    ("I paid $1.", "one dollar"),
    ("It costs $1,250.75.", "one thousand two hundred fifty dollars and seventy five cents"),
    ("Down 10% today.", "ten percent"),
    ("It is 25 kg.", "twenty five kilograms"),
    ("Room 4B please.", "four bee"),
    ("The 1st time.", "first"),
    ("Born in 1987.", "nineteen eighty seven"),
    ("Born in 2005.", "two thousand five"),
])
def test_number_like_expansions(norm, raw, expected):
    assert expected in norm(raw)


def test_no_digits_survive(norm):
    out = norm("On 3/15/2024 at 3:30 pm, 42 people paid $7.50 for 2 items.")
    assert not any(ch.isdigit() for ch in out), out


def test_email_and_url(norm):
    out = norm("Mail a.smith@example.com or see https://www.example.com/docs.")
    assert " at " in out and " dot " in out
    assert "@" not in out and "/" not in out and "http" not in out


def test_phone_number_is_spoken_digit_by_digit(norm):
    out = norm("Call +1 (555) 123-4567 now.")
    assert "five five five" in out
    assert "four five six seven" in out


def test_titles_and_streets_are_disambiguated(norm):
    assert "doctor smith" in norm("Dr. Smith arrived.")
    assert "oak drive" in norm("Turn onto Oak Dr. north.")
    assert "oak street" in norm("We live at 15 Oak St., nearby.")
    assert "saint mary" in norm("We passed St. Mary's church.")


def test_acronyms_spelled_but_words_kept(norm):
    out = norm("The FBI and NASA disagreed.")
    assert "ef bee eye" in out
    assert "nasa" in out


def test_british_and_is_dropped(norm):
    assert "thousand" in norm("2005 items")     # 'thousand' must survive
    assert " and " not in norm("There were 2005 items")


def test_normalizer_trace_covers_every_step(norm):
    trace = norm.trace("Dr. Smith paid $5 on 3/15/2024.")
    names = [n for n, _ in trace]
    assert names[0] == "input" and "currency" in names and names[-1] == "ascii"


def test_skip_steps_disables_a_rule():
    keep_caps = TextNormalizer(skip=("acronyms",), lowercase=False)
    assert "FBI" in keep_caps("The FBI arrived.")


# -- phonemizers -------------------------------------------------------------
def test_char_phonemizer_is_identity():
    p = build_phonemizer("char")
    assert p.phonemize_words(["cat", "dog"]) == [["c", "a", "t"], ["d", "o", "g"]]


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_phonemizer("no_such_backend")


@pytest.mark.parametrize("backend", ["char", "ipa", "arpabet"])
def test_one_phone_list_per_word(backend):
    p = build_phonemizer(backend)
    words = ["the", "record", "is", "broken"]
    out = p.phonemize_words(words, "the record is broken")
    assert len(out) == len(words)
    assert all(isinstance(chunk, list) and chunk for chunk in out)


# -- tokenizer ---------------------------------------------------------------
@pytest.mark.parametrize("backend", ["char", "ipa", "arpabet"])
def test_word_map_is_consistent(backend):
    tp = TextProcessor(backend)
    enc = tp.encode("The record is broken by the record broker.")
    assert enc.words == ["the", "record", "is", "broken", "by", "the", "record", "broker"]
    assert len(enc.ids) == len(enc.tokens) == len(enc.word_ids)
    assert max(enc.word_ids) == len(enc.words) - 1
    assert min(enc.word_ids) == 0


def test_both_homograph_instances_get_separate_word_slots():
    tp = TextProcessor("char")
    enc = tp.encode("The record is broken by the record broker.")
    first = [i for i, w in enumerate(enc.word_ids) if w == 1]
    second = [i for i, w in enumerate(enc.word_ids) if w == 6]
    assert first and second and set(first).isdisjoint(second)


def test_tokens_round_trip_through_ids():
    tp = TextProcessor("char")
    enc = tp.encode("hello world")
    assert tp.decode(enc.ids) == enc.tokens


def test_frozen_vocabulary_maps_unseen_symbols_to_unk():
    tp = TextProcessor("char")
    tp.encode("abc")
    size = tp.vocab_size
    tp.freeze()
    enc = tp.encode("xyz zzz qqq")
    assert tp.vocab_size == size          # no growth
    assert tp.symbol_table.symbols[enc.ids[1]] in set(tp.symbols)


def test_symbol_table_persists(tmp_path):
    tp = TextProcessor("char")
    tp.encode("the quick brown fox")
    path = tp.save_symbols(tmp_path / "symbols.json")
    reloaded = SymbolTable.load(path)
    assert reloaded.symbols == tp.symbols
    assert reloaded.pad_id == tp.pad_id


def test_specials_and_punctuation_come_first():
    tp = TextProcessor("char")
    assert tp.symbols[: len(SPECIALS)] == SPECIALS
    assert set(PUNCTUATION).issubset(set(tp.symbols))


def test_bos_eos_and_word_boundaries_can_be_disabled():
    tp = TextProcessor("char", add_bos_eos=False, add_word_boundary=False,
                       add_punctuation=False)
    enc = tp.encode("hi there!")
    assert enc.tokens == list("hithere")
