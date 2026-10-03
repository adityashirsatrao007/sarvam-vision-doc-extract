"""Unicode and Indian-context normalisation helpers (stdlib only).

Handles:

* NFKC normalisation (full-width punctuation, non-breaking spaces, ...),
* Devanagari digits -> ASCII,
* money: ``₹1,234.50`` -> ``1234.5``, Indian lakh/crore grouping
  ``₹1,03,368.00`` -> ``103368.0``, ``₹1.5 लाख`` -> ``150000.0``,
* dates: ``DD/MM/YYYY``, ``DD-MM-YYYY``, ``YYYY-MM-DD`` and Hindi month names
  (``14 फ़रवरी 2026``) -> ISO ``YYYY-MM-DD``,
* 10-digit Indian mobile numbers (``+91 98450 12345`` -> ``9845012345``),
* email addresses.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any

__all__ = [
    "nfkc",
    "collapse_ws",
    "ascii_digits",
    "normalize_amount",
    "normalize_date",
    "normalize_phone",
    "normalize_email",
    "normalize_person_name",
    "format_inr",
    "DEVANAGARI_DIGITS",
]

DEVANAGARI_DIGITS = "०१२३४५६७८९"
DEVANAGARI_DIGIT_MAP = dict(zip(DEVANAGARI_DIGITS, "0123456789"))

_ZWNJ = "\u200c"
_ZWJ = "\u200d"
_NUKTA = "\u093c"

# Strings people actually write in front of a number in Indian documents.
# Both Hindi spellings of "rupees" occur (रुपये / रुपए), plus the short forms
# used on bills; anything left over is a non-numeric value, not a number.
_CURRENCY_TOKENS = (
    "\u20b9",  # ₹
    "rs.",
    "rs",
    "inr",
    "रुपये",
    "रुपया",
    "रुपय",
    "रुपए",
    "रूपये",
    "रूपए",
    "रु.",
    "रू",
)

# ``1,23,456.78`` (lakh/crore grouping) and ``1234.56`` both appear in the wild.
_INDIAN_GROUPING = re.compile(r"^-?\d{1,3}(?:,\d{2})*(?:,\d{3})(?:\.\d+)?$")
_WESTERN_GROUPING = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_PLAIN_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")

_MULTIPLIERS: dict[str, int] = {
    "hazaar": 1_000,
    "hazar": 1_000,
    "हज़ार": 1_000,
    "हजार": 1_000,
    "thousand": 1_000,
    "k": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "लाख": 100_000,
    "लाखों": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "करोड़": 10_000_000,
    "करोड": 10_000_000,
}
_MULTIPLIER_RE = re.compile(
    r"(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*(?P<mult>"
    + "|".join(re.escape(k) for k in sorted(_MULTIPLIERS, key=len, reverse=True))
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)

_EN_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Hindi month spellings vary in printed/OCR'd documents, but only the
# *spelling* variants belong here: `_month_key()` already drops nukta and
# ZWNJ/ZWJ, so फ़रवरी and फरवरी are one key and need one row. What is left to
# enumerate is genuine orthography — anusvara vs. conjunct (सितंबर / सितम्बर),
# missing vowel signs (फ़रवरि), and regional forms (अगष्ट).
_HI_MONTHS_RAW: dict[str, int] = {
    "जनवरी": 1,
    "फ़रवरी": 2,
    "फरवरी": 2,
    "फ़रवरि": 2,
    "मार्च": 3,
    "अप्रैल": 4,
    "मई": 5,
    "जून": 6,
    "जुलाई": 7,
    "अगस्त": 8,
    "अगष्ट": 8,
    "सितम्बर": 9,
    "सितंबर": 9,
    "अक्टूबर": 10,
    "अकतूबर": 10,
    "नवम्बर": 11,
    "नवंबर": 11,
    "दिसम्बर": 12,
    "दिसंबर": 12,
}


def _strip_joiners(text: str) -> str:
    """Drop ZWNJ / ZWJ / nukta so spelling variants collapse to one key."""
    for ch in (_ZWNJ, _ZWJ, _NUKTA):
        text = text.replace(ch, "")
    return text


def nfkc(text: str) -> str:
    """Unicode NFKC normalisation (full-width -> ASCII, NBSP -> space, ...)."""
    return unicodedata.normalize("NFKC", str(text))


def _month_key(name: str) -> str:
    return collapse_ws(_strip_joiners(nfkc(name)).lower())


def collapse_ws(text: str) -> str:
    """Trim the ends and collapse every whitespace run (tab, newline, NBSP) to one space."""
    return re.sub(r"\s+", " ", str(text)).strip()


#: month-name (normalised) -> 1..12, covering English and Hindi.
MONTHS: dict[str, int] = {}
for _name, _num in {**_EN_MONTHS, **_HI_MONTHS_RAW}.items():
    MONTHS[_month_key(_name)] = _num

_MONTH_ALT = "|".join(
    re.escape(key) for key in sorted(MONTHS, key=len, reverse=True)
)

_DAY_FIRST_RE = re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?!\d)")
_ISO_FIRST_RE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_DAY_MONTH_YEAR_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:-|/|\.|,)?\s*(" + _MONTH_ALT + r")\s*,?\s*(\d{4})",
    re.IGNORECASE,
)
_MONTH_DAY_YEAR_RE = re.compile(
    "(" + _MONTH_ALT + r")\s*(?:-|/|\.|,)?\s*(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def ascii_digits(text: str) -> str:
    """Fold every digit form in ``text`` to ASCII: ``१२३`` and ``１２３`` -> ``123``.

    NFKC first, digit map second, because NFKC covers the digit forms this map
    cannot (full-width ``１２３`` -> ``123``, NBSP -> space, ligatures -> base
    letters) while it leaves ``०-९`` alone. Reverse the order and a full-width
    digit survives into the numeric regexes below, which are ASCII-only.
    """
    return "".join(DEVANAGARI_DIGIT_MAP.get(ch, ch) for ch in nfkc(text))


def _strip_currency(text: str) -> str:
    lowered = text
    lowered = lowered.replace("\u20b9", " ")
    for token in _CURRENCY_TOKENS[1:]:
        lowered = re.sub(rf"(?i)(?<![a-z]){re.escape(token)}(?![a-z])", " ", lowered)
    return lowered


def normalize_amount(raw: Any) -> float | None:
    """Parse an Indian money string into a ``float``; ``None`` if unparseable.

    >>> normalize_amount("₹1,234.50")
    1234.5
    >>> normalize_amount("₹1,03,368.00")   # lakh grouping
    103368.0
    >>> normalize_amount("1.5 लाख")
    150000.0
    >>> normalize_amount("(1,234.50)")     # accounting negative (credit note)
    -1234.5

    Deliberately strict: the *whole* value must be a number once currency
    words, ``/-`` terminators and filler words are removed. Leading prose
    ("Total 1,234") is the label parser's job to strip, not a guess this
    function should make — a wrong number is worse than no number.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    text = collapse_ws(ascii_digits(raw))
    if not text:
        return None

    # "(1,234.50)" is accounting notation for a negative amount (credit
    # notes, refunds); remember the brackets, parse the digits, re-apply the
    # sign at the end. Without this the whole value reads as garbage -> None.
    bracket_negative = len(text) > 1 and text.startswith("(") and text.endswith(")")
    if bracket_negative:
        text = collapse_ws(text[1:-1])
        if not text:
            return None

    # "1.5 लाख" / "2 crore" style multipliers are common in Indian documents.
    multiplier = 1
    match = _MULTIPLIER_RE.search(text)
    if match:
        token = match.group("mult")
        multiplier = _MULTIPLIERS.get(token, _MULTIPLIERS.get(token.lower(), 1))
        text = collapse_ws(text.replace(match.group(0), match.group("num"), 1))

    text = collapse_ws(_strip_currency(text))
    # Trailing filler words that commonly trail an amount line.
    text = collapse_ws(re.sub(r"(?i)\b(?:only|matr|मात्र)\b", " ", text))
    text = text.replace(" ", "")
    # "₹1,03,368/-": the "/-" rupee terminator is punctuation, not digits.
    text = re.sub(r"/-?$", "", text)
    if not text:
        return None

    if _WESTERN_GROUPING.match(text) or _INDIAN_GROUPING.match(text) or _PLAIN_NUMBER.match(text):
        number = text.replace(",", "")
    else:
        return None

    try:
        value = float(number) * multiplier
    except ValueError:
        return None
    return -abs(value) if bracket_negative else value


def _safe_date(year: int, month: int, day: int) -> str | None:
    if not 1900 <= year <= 2100:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_date(raw: Any) -> str | None:
    """Parse a date into ISO ``YYYY-MM-DD``; ``None`` if unparseable.

    Supports ``DD/MM/YYYY``, ``DD-MM-YYYY``, ``YYYY-MM-DD``, ``14 फ़रवरी 2026``
    and ``14 February 2026``. Day-first is the default for slash dates, which
    matches Indian document convention.
    """
    if raw is None:
        return None
    if isinstance(raw, date):
        # ``datetime`` subclasses ``date``; cut the time part off so the
        # validator's ``^\d{4}-\d{2}-\d{2}$`` check still sees a bare date.
        return raw.isoformat()[:10]

    text = collapse_ws(ascii_digits(raw))
    if not text:
        return None

    match = _ISO_FIRST_RE.search(text)
    if match:
        result = _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if result:
            return result

    match = _DAY_FIRST_RE.search(text)
    if match:
        result = _safe_date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        if result:
            return result

    # Hindi / English month names need joiners stripped before the lookup.
    lookup_text = _strip_joiners(text)
    match = _DAY_MONTH_YEAR_RE.search(lookup_text)
    if match:
        month = MONTHS.get(_month_key(match.group(2)))
        if month:
            result = _safe_date(int(match.group(3)), month, int(match.group(1)))
            if result:
                return result

    match = _MONTH_DAY_YEAR_RE.search(lookup_text)
    if match:
        month = MONTHS.get(_month_key(match.group(1)))
        if month:
            result = _safe_date(int(match.group(3)), month, int(match.group(2)))
            if result:
                return result

    return None


def normalize_phone(raw: Any) -> str | None:
    """Return a 10-digit Indian mobile number, or ``None``.

    >>> normalize_phone("+91 98450 12345")
    '9845012345'

    Accepts the ``+91`` / ``91`` country prefix and the ``0`` trunk prefix,
    then insists on the ``6-9`` first digit every Indian mobile allocation
    starts with — anything else (a landline, a 12-digit string that merely
    begins ``91``) is not a mobile and must not be reported as one. Callers
    that only want the digits (scoring) fall back to ``\\D`` stripping.
    """
    if raw is None:
        return None
    digits = re.sub(r"\D", "", ascii_digits(str(raw)))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10 and digits[0] in "6789":
        return digits
    return None


def normalize_email(raw: Any) -> str | None:
    """Lowercase, whitespace-stripped email; ``None`` when it is not an email."""
    if raw is None:
        return None
    value = collapse_ws(nfkc(str(raw))).strip(".,;")
    if not _EMAIL_RE.match(value):
        return None
    return value.lower()


def normalize_person_name(raw: Any) -> str | None:
    """Trim the value and drop punctuation only at its *edges*.

    Inner marks belong to the name — ``Rakesh Yadav @ Rocky`` stays intact —
    while a stray trailing period or comma (``"Meera Krishnan."``) does not.
    """
    if raw is None:
        return None
    value = collapse_ws(nfkc(str(raw)))
    value = value.strip(" ,;:.-")
    if not value:
        return None
    return value


def format_inr(amount: float, *, paise: bool = True) -> str:
    """Format a number with Indian lakh/crore grouping: ``103368 -> 1,03,368``.

    Rounds once, at paise precision, then splits: rounding the fraction of an
    already-split rupee part can produce 100 paise (``5.999`` -> ``₹5.100``).
    """
    negative = amount < 0
    value = abs(float(amount))
    whole, fraction = divmod(round(value * 100), 100)
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        grouped = ",".join(groups + [tail])
    else:
        grouped = digits
    out = f"{'-' if negative else ''}\u20b9{grouped}"
    if paise:
        out += f".{fraction:02d}"
    return out
