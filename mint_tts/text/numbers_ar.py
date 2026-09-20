"""Egyptian Arabic number verbalisation.

`num2words(lang="ar")` produces Modern Standard Arabic in the **nominative**
case, which an Egyptian speaker never says out loud::

    num2words   25 -> khamsa wa 3ishruun    (MSA nominative)
    Egyptian    25 -> khamsa wa 3ishriin    (oblique, used for everything)

That difference is not cosmetic here. The corpus is 100% Egyptian speech, so
writing the MSA form into the transcript means the text says one thing while
the audio says another. The aligner then has to map a grapheme sequence onto
a pronunciation that does not match it, and the model learns that mismatch.
Numbers are common enough in the source material (popular-science and history
programmes) that this is worth getting right rather than approximating.

The systematic differences from MSA:

    case        MSA inflects for case; Egyptian uses the oblique form always,
                so 20/30/.../90 end in -iin, never -uun
    dual        MSA `ithnaan`/`alfaan` -> Egyptian `itnein`/`alfein`
    interdental MSA /th/ -> Egyptian /t/: thalaatha -> talaata
                MSA /dh/ -> Egyptian /z/ or /d/
    hundred     MSA `mi'a` -> Egyptian `miyya`
    teens       MSA `thalaathata 3ashar` -> Egyptian `talattashar`

Gender agreement is deliberately **not** modelled. Arabic numbers agree with
the counted noun (and in the 3-10 range the agreement is inverted), which
needs the noun's gender -- information this pipeline does not have and cannot
reliably guess. The masculine/default forms are used, which is what a speaker
reading an isolated figure produces. Where the corpus disagrees, the model
sees a small consistent discrepancy rather than a large random one.
"""

from __future__ import annotations

# 0-10. These are the "counting" forms, as used when reading a bare number.
ONES = [
    "صفر",          # sifr       0
    "واحد",    # waahid     1
    "اتنين",  # itnein  2
    "تلاتة",  # talaata 3
    "اربعة",  # arba3a  4
    "خمسة",        # khamsa  5
    "ستة",              # sitta   6
    "سبعة",        # sab3a   7
    "تمانية",  # tamanya 8
    "تسعة",        # tis3a   9
    "عشرة",        # 3ashara 10
]

# 11-19. Egyptian contracts these heavily: 13 is `talattashar`, one word,
# not the MSA three-word `thalaathata 3ashara`.
TEENS = {
    11: "حداشر",                    # hidaashar
    12: "اتناشر",              # itnaashar
    13: "تلاتاشر",        # talattashar
    14: "اربعتاشر",  # arba3tashar
    15: "خمستاشر",        # khamastashar
    16: "ستاشر",                    # sittashar
    17: "سبعتاشر",        # saba3tashar
    18: "تمنتاشر",        # tamantashar
    19: "تسعتاشر",        # tisa3tashar
}

# 20, 30, ... 90 -- all oblique (-iin), which is the only form Egyptian uses.
TENS = {
    2: "عشرين",              # 3ishriin  20
    3: "تلاتين",        # talatiin  30
    4: "اربعين",        # arb3iin   40
    5: "خمسين",              # khamsiin  50
    6: "ستين",                    # sittiin   60
    7: "سبعين",              # sab3iin   70
    8: "تمانين",        # tamaniin  80
    9: "تسعين",              # tis3iin   90
}

# 100-900. 100 and 200 are their own words; 300-900 compound onto -miyya.
HUNDREDS = {
    1: "مية",                                  # miyya     100
    2: "ميتين",                      # miteen    200
    3: "تلتمية",                # tultumiyya 300
    4: "ربعمية",                # rub3umiyya 400
    5: "خمسمية",                # khumsumiyya 500
    6: "ستمية",                      # suttumiyya 600
    7: "سبعمية",                # sub3umiyya 700
    8: "تمنمية",                # tumnumiyya 800
    9: "تسعمية",                # tus3umiyya 900
}

WA = "و"                                      # wa -- "and"
ALF = "الف"                         # alf      1000
ALFEEN = "الفين"          # alfein   2000 (dual)
ALAAF = "الاف"                 # alaaf    thousands (3-10)
MALYOON = "مليون"         # malyoon
MALYOONEEN = "مليونين"    # malyoneen (dual)
MALAYEEN = "ملايين"  # malayeen  (3-10)
MELYAR = "مليار"          # milyar
MELYAREEN = "مليارين"
MELYARAT = "مليارات"
NAQS = "ناقص"                  # naaqis -- "minus"
FASLA = "فاصلة"           # faasla -- "point"


def _under_thousand(n: int) -> list[str]:
    """1-999 as a list of words."""
    parts: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts.append(HUNDREDS[hundreds])
    if rest:
        if parts:
            parts.append(WA)
        if rest <= 10:
            parts.append(ONES[rest])
        elif rest < 20:
            parts.append(TEENS[rest])
        else:
            tens, ones = divmod(rest, 10)
            if ones:
                # Egyptian puts the unit FIRST: "khamsa wa 3ishriin" (5 and 20).
                parts.extend([ONES[ones], WA, TENS[tens]])
            else:
                parts.append(TENS[tens])
    return parts


def _scale_words(count: int, singular: str, dual: str, plural: str) -> list[str]:
    """Apply Arabic's three-way number agreement to a scale word.

    1 -> the bare singular ("alf"), 2 -> the dual ("alfein"), 3-10 -> the
    plural ("talat alaaf"), 11+ -> the singular again ("khamastashar alf").
    Getting this wrong is immediately audible to a speaker.
    """
    if count == 1:
        return [singular]
    if count == 2:
        return [dual]
    if 3 <= count <= 10:
        return [*_under_thousand(count), plural]
    return [*_under_thousand(count), singular]


def number_to_words(value: int) -> str:
    """Verbalise an integer in Egyptian Arabic."""
    if value < 0:
        return NAQS + " " + number_to_words(-value)
    if value == 0:
        return ONES[0]

    parts: list[str] = []
    for divisor, (sg, du, pl) in (
        (1_000_000_000, (MELYAR, MELYAREEN, MELYARAT)),
        (1_000_000, (MALYOON, MALYOONEEN, MALAYEEN)),
        (1_000, (ALF, ALFEEN, ALAAF)),
    ):
        count, value = divmod(value, divisor)
        if count:
            if parts:
                parts.append(WA)
            parts.extend(_scale_words(count, sg, du, pl))
    if value:
        if parts:
            parts.append(WA)
        parts.extend(_under_thousand(value))
    return " ".join(parts)


def decimal_to_words(raw: str) -> str:
    """Verbalise a decimal: the digits after the point are read one by one."""
    whole, _, frac = raw.partition(".")
    out = number_to_words(int(whole or 0))
    if frac:
        digits = " ".join(ONES[int(d)] for d in frac if d.isdigit())
        if digits:
            out = f"{out} {FASLA} {digits}"
    return out


def year_to_words(value: int) -> str:
    """Years are read as ordinary numbers in Egyptian Arabic.

    Unlike English ("nineteen eighty-four"), Arabic reads 1984 as the full
    cardinal "alf wa tus3umiyya wa arba3a wa tamaniin", so no special casing
    is needed -- this exists to make that explicit rather than implicit.
    """
    return number_to_words(value)


def digits_to_words(raw: str) -> str:
    """Read a digit string one digit at a time (phone numbers, codes)."""
    return " ".join(ONES[int(d)] for d in raw if d.isdigit())
