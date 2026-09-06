"""Text normalisation for TTS.

Turns written text into something that can be spoken, before any G2P runs:

    "Dr. Smith paid $1,250.75 on 3/15/2024. Call +1 (555) 123-4567 or
     email a.smith@mit.edu about 15 Oak St., Apt. 4B."
    ->
    "doctor smith paid one thousand two hundred fifty dollars and seventy five
     cents on march fifteenth twenty twenty four. call plus one five five five,
     one two three, four five six seven or email a dot smith at m i t dot e d u
     about fifteen oak street, apartment four b."

Rule order matters a great deal -- currency must be handled before bare
numbers, dates before ordinals, phone numbers before digit groups -- so the
pipeline is an explicit ordered list rather than a dict of regexes.

`num2words` does the heavy lifting for number verbalisation; a small built-in
fallback keeps the module importable without it.
"""

from __future__ import annotations

import re
import unicodedata

try:
    from num2words import num2words as _n2w

    def _to_words(value, to: str = "cardinal") -> str:
        return _n2w(value, lang="en", to=to)

    HAVE_NUM2WORDS = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_NUM2WORDS = False
    _ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    def _to_words(value, to: str = "cardinal") -> str:
        return " ".join(_ONES[int(d)] for d in str(abs(int(value))) if d.isdigit())


# --------------------------------------------------------------------------
# lookup tables
# --------------------------------------------------------------------------
TITLES = {
    "mr": "mister", "mrs": "misses", "ms": "miss", "dr": "doctor", "prof": "professor",
    "rev": "reverend", "hon": "honorable", "pres": "president", "gov": "governor",
    "sen": "senator", "rep": "representative", "sgt": "sergeant", "capt": "captain",
    "lt": "lieutenant", "col": "colonel", "gen": "general", "maj": "major",
    "cpl": "corporal", "adm": "admiral", "jr": "junior", "sr": "senior",
    "st": "saint",  # only when followed by a name; see _expand_st
}

STREET_SUFFIXES = {
    "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "rd": "road", "ln": "lane", "dr": "drive", "ct": "court", "pl": "place",
    "sq": "square", "ter": "terrace", "pkwy": "parkway", "hwy": "highway",
    "cir": "circle", "trl": "trail", "way": "way", "expy": "expressway",
}
UNIT_DESIGNATORS = {
    "apt": "apartment", "ste": "suite", "bldg": "building", "fl": "floor",
    "rm": "room", "dept": "department", "no": "number", "po": "p o",
}
GENERAL_ABBREVIATIONS = {
    "etc": "et cetera", "vs": "versus", "v": "versus", "approx": "approximately",
    "est": "established", "min": "minutes", "max": "maximum", "misc": "miscellaneous",
    "inc": "incorporated", "ltd": "limited", "co": "company", "corp": "corporation",
    "univ": "university", "assn": "association", "dept": "department",
    "govt": "government", "intl": "international", "mt": "mount", "ft": "fort",
    "e.g": "for example", "i.e": "that is", "aka": "also known as",
}
MONTHS = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "may": "may", "jun": "june", "jul": "july", "aug": "august",
    "sep": "september", "sept": "september", "oct": "october",
    "nov": "november", "dec": "december",
}
MONTH_NUMBERS = ["january", "february", "march", "april", "may", "june", "july",
                 "august", "september", "october", "november", "december"]
CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", "sen", "sen"),
    "₹": ("rupee", "rupees", "paisa", "paise"),
}
UNITS = {
    "km": "kilometers", "cm": "centimeters", "mm": "millimeters", "m": "meters",
    "kg": "kilograms", "mg": "milligrams", "lb": "pounds", "lbs": "pounds",
    "oz": "ounces", "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz",
    "ghz": "gigahertz", "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes",
    "tb": "terabytes", "mph": "miles per hour", "kph": "kilometers per hour",
}
SYMBOLS = {
    "&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "%": " percent ",
    "#": " number ", "~": " approximately ", "/": " slash ", "\\": " backslash ",
    "*": " star ", "_": " underscore ", "^": " to the power of ",
}
LETTER_NAMES = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef",
    "g": "gee", "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "el",
    "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "ar",
    "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "double you",
    "x": "ex", "y": "why", "z": "zee",
}
# Acronyms that are pronounced as words, not spelled out.
SPOKEN_ACRONYMS = {
    "nasa", "nato", "unesco", "unicef", "opec", "aids", "laser", "radar",
    "scuba", "gif", "jpeg", "png", "ram", "rom", "wifi", "covid", "asap",
}

# Applied first: safe to do before any rule runs.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": ", ", "…": ".", " ": " ", "−": "-",
}
# Applied last, after the currency and temperature rules have consumed their
# symbols; whatever non-ASCII is still here gets spoken or dropped.
_RESIDUAL_MAP = {
    "½": " one half ", "¼": " one quarter ", "¾": " three quarters ",
    "°": " degrees ", "µ": " micro ", "©": " copyright ",
    "®": " registered ", "™": " trademark ",
    "$": " dollars ", "£": " pounds ", "€": " euros ",
    "¥": " yen ", "₹": " rupees ",
}


def _digits(text: str) -> str:
    return " ".join("oh" if d == "0" else _to_words(int(d)) for d in text if d.isdigit())


_AND_RE = re.compile(r"\band\b")
_MULTISPACE_RE = re.compile(r"\s{2,}")


def _drop_and(text: str) -> str:
    """num2words gives the British "one thousand and five"; TTS corpora
    (LJSpeech included) overwhelmingly say "one thousand five"."""
    return _MULTISPACE_RE.sub(" ", _AND_RE.sub(" ", text)).strip()


def _cardinal(n: int) -> str:
    return _drop_and(_to_words(n).replace("-", " ").replace(",", ""))


def _ordinal(n: int) -> str:
    if HAVE_NUM2WORDS:
        return _drop_and(_to_words(n, to="ordinal").replace("-", " ").replace(",", ""))
    return _cardinal(n) + "th"


def _year(n: int) -> str:
    """1987 -> nineteen eighty seven; 2005 -> two thousand five."""
    if 1100 <= n <= 1999 or 2010 <= n <= 2099:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return f"{_cardinal(hi)} hundred"
        if lo < 10:
            return f"{_cardinal(hi)} oh {_cardinal(lo)}"
        return f"{_cardinal(hi)} {_cardinal(lo)}"
    return _cardinal(n)


# --------------------------------------------------------------------------
# individual rules
# --------------------------------------------------------------------------
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE_RE = re.compile(
    r"(?<![\w.])(\+?\d{1,3}[\s.-]?)?(\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?![\w.])"
)
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap])\.?\s?m\.?\b", re.IGNORECASE)
_TIME24_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_TEXT_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"([$£€¥₹])\s?(\d[\d,]*)(?:\.(\d{1,2}))?\s*(million|billion|trillion|k|m|b)?",
                          re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")
_TEMP_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*°\s*([CF])\b")
_UNIT_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s?(" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")\b",
                      re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_COMMA_NUM_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+)\b")
_DECIMAL_RE = re.compile(r"\b(\d+)\.(\d+)\b")
_YEAR_RE = re.compile(r"\b(1[1-9]\d{2}|20\d{2})s?\b")
_INT_RE = re.compile(r"\b\d+\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{5,}\b")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")
_ADDRESS_RE = re.compile(
    r"\b(" + "|".join(sorted(STREET_SUFFIXES, key=len, reverse=True)) + r")\.(?=\s|,|$)",
    re.IGNORECASE,
)
_UNIT_DESIG_RE = re.compile(
    r"\b(" + "|".join(sorted(UNIT_DESIGNATORS, key=len, reverse=True)) + r")\.\s*",
    re.IGNORECASE,
)
_ALNUM_RE = re.compile(r"\b(\d+)([A-Za-z]{1,2})\b")
_ST_NAME_RE = re.compile(r"\bSt\.\s+(?=[A-Z])")
# "Dr. Smith" is a title; "15 Oak Dr." is a street. Case and what follows tell
# them apart, so the title rule requires a capitalised word after it and runs
# first; whatever is left falls through to the street-suffix rule.
# The abbreviation itself is matched case-insensitively, but the lookahead for
# a following capital must stay case-sensitive -- a global IGNORECASE would
# make [A-Z] match lowercase too, and "Oak Dr. north" would become "doctor".
_TITLE_CAP_RE = re.compile(
    r"(?i:\b(" + "|".join(sorted(TITLES, key=len, reverse=True)) + r")\.\s+)(?=[A-Z])")
_TITLE_RE = re.compile(r"\b(" + "|".join(sorted(TITLES, key=len, reverse=True)) + r")\.", re.IGNORECASE)
_GENERAL_ABBR_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(GENERAL_ABBREVIATIONS, key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


def expand_urls(text: str) -> str:
    def repl(m):
        url = m.group(0).rstrip(".,;:!?")
        url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        url = url.replace("www.", "w w w dot ")
        url = url.replace(".", " dot ").replace("/", " slash ").replace("-", " dash ")
        url = url.replace("_", " underscore ")
        return " " + url + " "
    return _URL_RE.sub(repl, text)


def expand_emails(text: str) -> str:
    def repl(m):
        local, domain = m.group(0).split("@", 1)
        local = local.replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
        parts = domain.split(".")
        spoken = []
        for i, part in enumerate(parts):
            spoken.append(_spell_if_initialism(part))
            if i < len(parts) - 1:
                spoken.append("dot")
        return f" {local} at {' '.join(spoken)} "
    return _EMAIL_RE.sub(repl, text)


def _spell_if_initialism(token: str) -> str:
    """Short, vowel-free or well-known TLD-ish chunks get spelled out."""
    low = token.lower()
    if low in SPOKEN_ACRONYMS:
        return low
    if len(token) <= 4 and not re.search(r"[aeiou]", low):
        return " ".join(LETTER_NAMES.get(c, c) for c in low)
    if low in {"com", "org", "net", "edu", "gov", "io", "ai", "co", "uk"}:
        return low if low in {"com", "org", "net"} else " ".join(LETTER_NAMES.get(c, c) for c in low)
    return low


def expand_phone_numbers(text: str) -> str:
    def repl(m):
        raw = m.group(0)
        groups = re.findall(r"\d+", raw)
        spoken = []
        if raw.strip().startswith("+"):
            spoken.append("plus")
        spoken.append(", ".join(_digits(g) for g in groups))
        return " " + " ".join(spoken) + " "
    return _PHONE_RE.sub(repl, text)


def expand_times(text: str) -> str:
    def repl12(m):
        h, mm, ss, ap = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4).lower()
        out = _cardinal(h)
        if mm == 0:
            out += " o'clock" if not ss else ""
        elif mm < 10:
            out += f" oh {_cardinal(mm)}"
        else:
            out += f" {_cardinal(mm)}"
        if ss:
            out += f" and {_cardinal(int(ss))} seconds"
        return f" {out} {ap} m "

    def repl24(m):
        h, mm, ss = int(m.group(1)), int(m.group(2)), m.group(3)
        out = f"{_cardinal(h)} " + (f"oh {_cardinal(mm)}" if 0 < mm < 10
                                    else ("hundred" if mm == 0 else _cardinal(mm)))
        if ss:
            out += f" and {_cardinal(int(ss))} seconds"
        return f" {out} "

    return _TIME24_RE.sub(repl24, _TIME_RE.sub(repl12, text))


def expand_dates(text: str) -> str:
    def repl_text(m):
        month = MONTHS[m.group(1).lower()[:4].rstrip(".")] if m.group(1).lower()[:4] in MONTHS \
            else MONTHS.get(m.group(1).lower()[:3], m.group(1).lower())
        day = _ordinal(int(m.group(2)))
        year = f" {_year(int(m.group(3)))}" if m.group(3) else ""
        return f" {month} {day}{year} "

    def repl_slash(m):
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1 <= mo <= 12:
            return m.group(0)
        y = y + 2000 if y < 100 and y < 50 else (y + 1900 if y < 100 else y)
        return f" {MONTH_NUMBERS[mo - 1]} {_ordinal(d)} {_year(y)} "

    def repl_iso(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1 <= mo <= 12:
            return m.group(0)
        return f" {MONTH_NUMBERS[mo - 1]} {_ordinal(d)} {_year(y)} "

    text = _DATE_ISO_RE.sub(repl_iso, text)
    text = _DATE_SLASH_RE.sub(repl_slash, text)
    return _DATE_TEXT_RE.sub(repl_text, text)


def expand_currency(text: str) -> str:
    def repl(m):
        sym, whole, frac, scale = m.group(1), m.group(2).replace(",", ""), m.group(3), m.group(4)
        major_s, major_p, minor_s, minor_p = CURRENCIES[sym]
        if scale:
            scale_word = {"k": "thousand", "m": "million", "b": "billion"}.get(scale.lower(), scale.lower())
            amount = f"{_cardinal(int(whole))}" + (f" point {_digits(frac)}" if frac else "")
            return f" {amount} {scale_word} {major_p} "
        n = int(whole)
        out = f"{_cardinal(n)} {major_s if n == 1 else major_p}"
        if frac:
            c = int(frac.ljust(2, '0'))
            if c:
                out += f" and {_cardinal(c)} {minor_s if c == 1 else minor_p}"
        return f" {out} "
    return _CURRENCY_RE.sub(repl, text)


def expand_measures(text: str) -> str:
    text = _PERCENT_RE.sub(lambda m: f" {_number_phrase(m.group(1))} percent ", text)
    text = _TEMP_RE.sub(
        lambda m: f" {_number_phrase(m.group(1))} degrees "
                  f"{'celsius' if m.group(2).upper() == 'C' else 'fahrenheit'} ", text)
    text = _UNIT_RE.sub(lambda m: f" {_number_phrase(m.group(1))} {UNITS[m.group(2).lower()]} ", text)
    return text


def _number_phrase(raw: str) -> str:
    raw = raw.replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        return f"{_cardinal(int(whole or 0))} point {_digits(frac)}"
    return _cardinal(int(raw))


def expand_abbreviations(text: str) -> str:
    text = _ST_NAME_RE.sub("saint ", text)
    text = _TITLE_CAP_RE.sub(lambda m: TITLES[m.group(1).lower()] + " ", text)
    text = _ADDRESS_RE.sub(lambda m: STREET_SUFFIXES[m.group(1).lower()] + " ", text)
    text = _UNIT_DESIG_RE.sub(lambda m: UNIT_DESIGNATORS[m.group(1).lower()] + " ", text)
    text = _TITLE_RE.sub(lambda m: TITLES[m.group(1).lower()] + " ", text)
    text = _GENERAL_ABBR_RE.sub(lambda m: GENERAL_ABBREVIATIONS[m.group(1).lower()] + " ", text)
    return text


def expand_acronyms(text: str) -> str:
    """Spell out capitalised initialisms (FBI -> ef bee eye), keep NASA as a word."""
    def repl(m):
        word = m.group(1)
        low = word.lower()
        if low in SPOKEN_ACRONYMS:
            return low
        if re.search(r"[AEIOU]", word) and len(word) >= 4:
            return word  # probably a real word in caps
        return " " + " ".join(LETTER_NAMES.get(c.lower(), c.lower()) for c in word) + " "
    return _ACRONYM_RE.sub(repl, text)


def expand_numbers(text: str) -> str:
    text = _ORDINAL_RE.sub(lambda m: " " + _ordinal(int(m.group(1))) + " ", text)
    text = _COMMA_NUM_RE.sub(lambda m: m.group(1).replace(",", ""), text)
    text = _DECIMAL_RE.sub(lambda m: f" {_cardinal(int(m.group(1)))} point {_digits(m.group(2))} ", text)
    text = _LONG_DIGITS_RE.sub(lambda m: " " + _digits(m.group(0)) + " ", text)
    text = _ALNUM_RE.sub(
        lambda m: " " + _cardinal(int(m.group(1))) + " "
        + " ".join(LETTER_NAMES.get(c.lower(), c.lower()) for c in m.group(2)) + " ", text)
    text = _YEAR_RE.sub(lambda m: " " + _year(int(m.group(0).rstrip('s')))
                        + ("s" if m.group(0).endswith("s") else "") + " ", text)
    text = _INT_RE.sub(lambda m: " " + _cardinal(int(m.group(0))) + " ", text)
    return text


def expand_symbols(text: str) -> str:
    for sym, word in SYMBOLS.items():
        text = text.replace(sym, word)
    return text


def normalise_punctuation(text: str) -> str:
    """Curly quotes, dashes and vulgar fractions -> plain ASCII equivalents.

    Currency and degree symbols are deliberately left alone here: the currency
    and temperature rules still need them.
    """
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return text


def fold_to_ascii(text: str) -> str:
    """Final pass: anything non-ASCII left over is dropped."""
    for src, dst in _RESIDUAL_MAP.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def collapse_whitespace(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?]){2,}", r"\1", text)
    return text.strip()


class TextNormalizer:
    """Ordered normalisation pipeline.

    Steps run in a fixed order because they overlap: `$1,250.75` must be seen
    by the currency rule before the decimal rule ever gets to it.
    """

    STEPS = [
        # Quote/dash cleanup runs first, but the ASCII fold runs *last*: folding
        # early would delete the currency and degree symbols before the rules
        # that depend on them ever see the text.
        ("punctuation", normalise_punctuation),
        ("urls", expand_urls),
        ("emails", expand_emails),
        ("phones", expand_phone_numbers),
        ("dates", expand_dates),
        ("times", expand_times),
        ("currency", expand_currency),
        ("measures", expand_measures),
        ("abbreviations", expand_abbreviations),
        ("acronyms", expand_acronyms),
        ("numbers", expand_numbers),
        ("symbols", expand_symbols),
        ("ascii", fold_to_ascii),
    ]

    def __init__(self, lowercase: bool = True, keep_punctuation: str = "!'(),-.:;?\"",
                 skip: tuple[str, ...] = ()):
        self.lowercase = lowercase
        self.keep_punctuation = keep_punctuation
        self.skip = set(skip)
        allowed = re.escape(keep_punctuation)
        self._strip_re = re.compile(rf"[^a-zA-Z0-9\s{allowed}]")

    def normalize(self, text: str) -> str:
        for name, fn in self.STEPS:
            if name not in self.skip:
                text = fn(text)
        if self.lowercase:
            text = text.lower()
        text = self._strip_re.sub(" ", text)
        return collapse_whitespace(text)

    __call__ = normalize

    def trace(self, text: str) -> list[tuple[str, str]]:
        """Return the text after each step -- for debugging a bad expansion."""
        out = [("input", text)]
        for name, fn in self.STEPS:
            if name in self.skip:
                continue
            text = fn(text)
            out.append((name, collapse_whitespace(text)))
        return out


_DEFAULT = TextNormalizer()


def normalize_text(text: str, normalizer: TextNormalizer | None = None) -> str:
    return (normalizer or _DEFAULT).normalize(text)
