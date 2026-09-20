"""Egyptian Arabic homographs, and how difficulty is measured on a sentence.

Two things live here.

1. A lexicon of undiacritised Egyptian spellings that map to more than one
   pronunciation. This is the ground truth the difficulty signal and the
   probes are built from.
2. ``difficulty_profile``, which turns a word list into a per-word score used
   by the compute loss to *discount the penalty* on genuinely ambiguous words,
   so the router can afford to think about them.

Why Arabic homographs are a harder case than English ones
---------------------------------------------------------
English homographs are mostly lexical accidents: `record`/`record` are two
different words that happen to share a spelling, and there are a few hundred
of them. Undiacritised Arabic is different in kind, because the short vowels
are *systematically* absent from the orthography. Three distinct sources of
ambiguity stack up:

**Lexical.** The same consonant skeleton spells unrelated words. `علم` is
flag, science, knowledge, or "he taught", depending only on the vowels.

**Morphological.** The same skeleton spells different inflections of one root.
`كتب` is "he wrote", "books", or "it was written". Active and passive voice
are frequently distinguished by vowels alone, so `ضرب` can be "he hit" or "he
was hit" -- a difference that changes who did what.

**Clitic / agreement.** A possessive or object suffix agrees with the
addressee's gender, and that agreement is carried entirely by a short vowel.
`عمرك` is `3omrak` to a man and `3omrik` to a woman; `هسيبك` is `hasiibak` or
`hasiibik`. The spelling is identical. The evidence is elsewhere in the
sentence -- in the example this project was built around, the feminine verb
`فكرتي` and `تجاوبي` several words away are the only things that say the
addressee is a woman.

That last class is why routing *per token* and attending *across the whole
sentence* is the right shape of solution: the disambiguating evidence is
non-local, and a fixed-depth encoder has a fixed number of hops to find it.

Coverage, honestly
------------------
This lexicon is hand-built and therefore partial. It is used for two things
only -- a training-time difficulty prior and an evaluation probe set -- and
neither requires completeness: an unlisted homograph simply gets the default
difficulty and is not probed. It is emphatically *not* a pronunciation
dictionary, and nothing in the model consults it at inference time. The model
must generalise from context, not from this list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# The lexicon.
#
# Each entry: undiacritised spelling -> the readings it can take. `readings`
# are written with tashkeel purely for human readability; nothing consumes
# them programmatically.
# --------------------------------------------------------------------------


@dataclass
class Homograph:
    spelling: str
    readings: list[str]
    kind: str           # lexical | morphological | clitic
    gloss: str = ""
    note: str = ""


HOMOGRAPHS: list[Homograph] = [
    # -- lexical -----------------------------------------------------------
    Homograph("علم", ["عَلَم", "عِلْم", "عَلَّم"],
              "lexical", "flag / science / he taught"),
    Homograph("كتب", ["كَتَب", "كُتُب", "كُتِب"],
              "morphological", "he wrote / books / it was written"),
    Homograph("ذهب", ["ذَهَب", "ذَهَب"],
              "lexical", "gold / he went"),
    Homograph("بعد", ["بَعْد", "بَعَّد", "بُعْد"],
              "lexical", "after / he moved away / distance"),
    Homograph("سلم", ["سَلِّم", "سُلَّم", "سَلام"],
              "lexical", "he greeted / ladder / peace"),
    Homograph("قلب", ["قَلب", "قَلَب"],
              "lexical", "heart / he flipped"),
    Homograph("حسب", ["حَسَب", "حِسَاب"],
              "lexical", "he calculated / account"),
    Homograph("شرب", ["شِرب", "شَرَب"],
              "lexical", "drink (n) / he drank"),
    Homograph("مصر", ["ماصر", "مصر"],
              "lexical", "Egypt / he insisted"),
    Homograph("دهب", ["دَهَب", "داهب"],
              "lexical", "gold / going"),
    Homograph("عقد", ["عَقد", "عُقد", "عَقَد"],
              "lexical", "contract / necklace / he held"),
    Homograph("جمل", ["جَمَل", "جُملة"],
              "lexical", "camel / sentence"),
    Homograph("سن", ["سِن", "سَنّ"],
              "lexical", "age / tooth / he sharpened"),
    Homograph("ملك", ["مَلِك", "مُلك", "مَلَك"],
              "lexical", "king / property / angel"),
    Homograph("بحر", ["بَحر", "بَحّار"],
              "lexical", "sea / sailor"),
    Homograph("سبع", ["سَبع", "سِبع"],
              "lexical", "seven / lion"),
    Homograph("شعر", ["شَعر", "شِعر", "شَعَر"],
              "lexical", "hair / poetry / he felt"),
    Homograph("ورد", ["وَرد", "وَرَد"],
              "lexical", "roses / it arrived"),
    Homograph("طبع", ["طَبع", "طَبَع", "طَبعًا"],
              "lexical", "nature / he printed / of course"),
    Homograph("كرة", ["كُرة", "كَرّة"],
              "lexical", "ball / a time/instance"),
    Homograph("سر", ["سِر", "سَرّ", "سُرّ"],
              "lexical", "secret / he pleased / joy"),
    Homograph("حب", ["حُبّ", "حَبّ", "حِبّ"],
              "lexical", "love / grains / he loved"),
    Homograph("دور", ["دَور", "دُور", "دَوَّر"],
              "lexical", "role/turn / floor / he turned"),
    Homograph("عين", ["عَين", "عَيَّن"],
              "lexical", "eye / he appointed"),
    Homograph("نور", ["نُور", "نَوَّر"],
              "lexical", "light / he lit"),
    Homograph("عربي", ["عَرَبي", "عَرَبِيّ"],
              "lexical", "Arabic / a cart-driver"),

    # -- morphological: active vs passive, the vowel is the only difference
    Homograph("ضرب", ["ضَرَب", "ضُرِب", "ضَرب"],
              "morphological", "he hit / he was hit / a hit"),
    Homograph("قتل", ["قَتَل", "قُتِل", "قَتل"],
              "morphological", "he killed / he was killed / killing"),
    Homograph("اتقال", ["اتقال", "اتقَال"],
              "morphological", "it was said"),
    Homograph("درس", ["دَرَس", "دَرِّس", "دَرس"],
              "morphological", "he studied / he taught / a lesson"),
    Homograph("فهم", ["فِهِم", "فَهِّم", "فَهم"],
              "morphological", "he understood / he explained / understanding"),
    Homograph("عرف", ["عِرِف", "عَرَّف", "عُرف"],
              "morphological", "he knew / he introduced / custom"),
    Homograph("كسر", ["كَسَر", "كُسِر", "كَسر"],
              "morphological", "he broke / it was broken / a break"),

    # -- clitic: 2nd-person agreement carried by a short vowel -------------
    # These are the cases the project was retargeted for. The spelling is
    # identical; the addressee's gender decides the vowel, and the evidence
    # is elsewhere in the sentence.
    Homograph("عمرك", ["عُمرَك", "عُمرِك"],
              "clitic", "your age (m/f)", "3omrak vs 3omrik"),
    Homograph("هسيبك", ["هسيبَك", "هسيبِك"],
              "clitic", "I will leave you (m/f)", "hasiibak vs hasiibik"),
    Homograph("معاك", ["معاك", "معاكِ"],
              "clitic", "with you (m/f)"),
    Homograph("ليك", ["ليكَ", "ليكِ"],
              "clitic", "for you (m/f)"),
    Homograph("بيتك", ["بَيتَك", "بَيتِك"],
              "clitic", "your house (m/f)"),
    Homograph("اسمك", ["اسمَك", "اسمِك"],
              "clitic", "your name (m/f)"),
    Homograph("شكرالك", ["شكرالك"], "clitic", "thanks to you"),
    Homograph("قلبك", ["قلبَك", "قلبِك"],
              "clitic", "your heart (m/f)"),
    Homograph("رأيك", ["رأيَك", "رأيِك"],
              "clitic", "your opinion (m/f)"),
    Homograph("عندك", ["عندَك", "عندِك"],
              "clitic", "you have (m/f)"),
    Homograph("منك", ["منكَ", "منكِ"],
              "clitic", "from you (m/f)"),
    Homograph("وراك", ["وراكَ", "وراكِ"],
              "clitic", "behind you (m/f)"),
    Homograph("كلامك", ["كلامَك", "كلامِك"],
              "clitic", "your words (m/f)"),
    Homograph("شغلك", ["شغلَك", "شغلِك"],
              "clitic", "your work (m/f)"),
]

# Fast lookup: spelling -> number of distinct readings.
HOMOGRAPH_INDEX: dict[str, Homograph] = {h.spelling: h for h in HOMOGRAPHS}

# A 2nd-person object/possessive clitic written with a bare kaf is ambiguous
# for gender whenever the verb/adjective agreement is not adjacent. Matching
# the suffix generalises far beyond the explicit list above, which is what
# makes the difficulty signal useful on a 62k-word vocabulary.
_KAF = "ك"
_CLITIC_MIN_LEN = 4

# Feminine-addressee cues: if any of these appear, a bare-kaf clitic in the
# same sentence is very likely feminine. Used only to mark the sentence as
# *ambiguous and resolvable*, never to pick the pronunciation.
FEMININE_CUES = {
    "انتي", "إنتي",          # enti
    "يا بنت", "بنت",
    "هانم", "يا ست",
}
# Verb forms ending in -i are 2fs in Egyptian ("fakkarti", "tegawbi").
_FEM_VERB_SUFFIXES = ("تي", "ي")


def is_clitic_ambiguous(word: str) -> bool:
    """Does this word end in a 2nd-person kaf clitic whose vowel is unwritten?"""
    return (
        len(word) >= _CLITIC_MIN_LEN
        and word.endswith(_KAF)
        and word not in {"ملك", "سمك"}   # not clitics
    )


def sentence_has_feminine_cue(words: list[str]) -> bool:
    """Is there evidence elsewhere in the sentence that the addressee is female?

    This is the *reason* a clitic is resolvable at all, and its distance from
    the clitic is why the model needs to attend across the sentence.
    """
    for w in words:
        if w in FEMININE_CUES:
            return True
        # 2fs verb: ends in -ti/-i and is long enough not to be a noun
        if len(w) >= 5 and w.endswith("تي"):
            return True
    return False


def word_difficulty(word: str, words: list[str] | None = None) -> float:
    """A per-word difficulty score in [0, 1].

    0.0   ordinary word, one plausible reading
    0.5   ends in an ambiguous clitic, but nothing in the sentence resolves it
    0.7   ends in an ambiguous clitic AND the sentence carries the cue that
          resolves it -- the case the model can actually learn from
    1.0   a listed homograph with two or more distinct readings

    The scale is deliberately coarse. It is a *prior* that tells the compute
    penalty where to relax, not a target the router is trained to reproduce;
    making it finer would be false precision over a hand-built lexicon.
    """
    if word in HOMOGRAPH_INDEX:
        h = HOMOGRAPH_INDEX[word]
        return 1.0 if len(set(h.readings)) > 1 else 0.6
    if is_clitic_ambiguous(word):
        if words and sentence_has_feminine_cue(words):
            return 0.7
        return 0.5
    return 0.0


def difficulty_profile(words: list[str]) -> list[float]:
    """Per-word difficulty for a whole utterance."""
    return [word_difficulty(w, words) for w in words]


def sentence_difficulty(words: list[str]) -> float:
    """One number for the utterance: the mean of its word difficulties.

    Used to report whether the corpus contains hard sentences at all, and by
    `scripts/homograph_coverage.py` to say whether it can teach this.
    """
    if not words:
        return 0.0
    return sum(difficulty_profile(words)) / len(words)


# --------------------------------------------------------------------------
# minimal pairs for the probe
# --------------------------------------------------------------------------
@dataclass
class ArabicPair:
    word: str
    a: str
    b: str
    kind: str = "lexical"
    note: str = ""
    extras: dict = field(default_factory=dict)


# Each pair is the SAME spelling in two contexts that force different
# pronunciations. The probe measures whether the model renders them
# differently; the audio is logged so a speaker can judge whether it renders
# them *correctly*, which no automatic metric here can settle.
ARABIC_PAIRS = [
    ArabicPair(
        "علم",
        "انا رسمت علم مصر",
        "انا بحب العلم جدا و نفسي ابقا عالم لما اكبر",
        "lexical", "3alam (flag) vs 3elm (science)"),
    ArabicPair(
        "كتب",
        "هو كتب الجواب امبارح",
        "اشتريت تلات كتب من المكتبة",
        "morphological", "katab (he wrote) vs kotob (books)"),
    ArabicPair(
        "ذهب",
        "الخاتم ده معمول من ذهب",
        "هو ذهب للمدرسة بدري",
        "lexical", "dahab (gold) vs zahab (he went)"),
    ArabicPair(
        "قلب",
        "قلبي بيوجعني اوي",
        "هو قلب الطربيزة علي الارض",
        "lexical", "alb (heart) vs alab (he flipped)"),
    ArabicPair(
        "شعر",
        "شعرها طويل و ناعم",
        "انا بحب الشعر العربي القديم",
        "lexical", "sha3r (hair) vs she3r (poetry)"),
    ArabicPair(
        "عمرك",
        "يا عم عمرك شفت حاجة زي كده يا راجل",
        "عمرك فكرتي الراجل بتاع غزل البنات بينفخ الكيس ازاي",
        "clitic", "3omrak (to a man) vs 3omrik (to a woman) -- the cue is 'fakkarti', four words later"),
    ArabicPair(
        "هسيبك",
        "هسيبك ترتاح شوية يا عم احمد",
        "هسيبك تجاوبي و تخمني يا فاطمة",
        "clitic", "hasiibak vs hasiibik -- resolved by the 2fs verbs that follow"),
    ArabicPair(
        "ضرب",
        "هو ضرب الولد بسبب الشغل",
        "الولد ضرب من غير ذنب",
        "morphological", "darab (active) vs dorib (passive)"),
    ArabicPair(
        "سن",
        "سنه كبير علي الشغلانة دي",
        "الدكتور خلع لي السن اللي بيوجعني",
        "lexical", "senn (age) vs senn (tooth)"),
    ArabicPair(
        "دور",
        "جالي دور كبير في الفيلم",
        "انا ساكن في الدور التالت",
        "lexical", "dor (role) vs dowr (floor)"),
]


# --------------------------------------------------------------------------
# tongue twisters: hard PHONETICS, easy semantics
# --------------------------------------------------------------------------
# These separate two hypotheses that would otherwise be confounded. If the
# router allocates by *ambiguity*, these should stay cheap -- there is nothing
# to disambiguate. If it allocates by "articulatory difficulty" or by
# character-level surprise, they should light up. Either result is
# informative, and without them a positive homograph result could just be
# "the router spends more on unusual letter sequences".
TONGUE_TWISTERS = [
    "خمسة شرايط شراطتهم شرايط مشروطة",
    "شرشف السرير الاحمر مشرشر",
    "قلبي قلبك و قلبك قلبي",
    "تلات تلاجات متلجة تحت التلاجة",
    "سبع سبعان في سبع ساعات",
]

# Long but trivial: the length control. If these cost as much as the
# homograph sentences, compute is tracking LENGTH, not difficulty.
LONG_EASY = [
    "انا رحت السوق و اشتريت عيش و لبن و جبنة و زيتون و رجعت البيت",
    "كل يوم الصبح باشرب شاي سخن و باقرا الجرنان قبل ما اروح الشغل",
]

# Trivially easy: the speed control. These should be the cheapest utterances
# in the probe set, and the ones the model answers fastest.
EASY = [
    "عامل ايه النهاردة؟",
    "صباح الخير يا فندم",
    "ازيك عامل ايه؟",
]
