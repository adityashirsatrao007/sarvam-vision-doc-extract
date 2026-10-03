"""Offline rule / regex / key-value extraction (the default provider).

Zero third-party dependencies. The parser works line by line:

1. lines starting with ``#`` are comments and are skipped;
2. ``label: value`` (or ``label<tab|2+ spaces>value``) is matched against a
   table of English *and* Devanagari labels;
3. a small set of known document titles (``INVOICE``, ``FIR``,
   ``आवेदन पत्र``, ...) is recognised;
4. the first non-title line **before** the first label line becomes the
   issuing organisation (``org_name``);
5. pipe-delimited tables (``S.No | Description | Qty | Rate | Amount``) become
   ``line_items``.

The design is intentionally boring: deterministic, auditable, and easy to
extend by adding one row to :data:`_LABEL_ALIASES`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .normalize import (
    ascii_digits,
    collapse_ws,
    nfkc,
    normalize_amount,
    normalize_date,
    normalize_email,
    normalize_person_name,
    normalize_phone,
)
from .schema import Document, LineItem

__all__ = [
    "LABEL_FIELDS",
    "KNOWN_TITLES",
    "extract_document",
    "detect_doc_type",
]

# --- label vocabulary -------------------------------------------------------
# (label as written, field name, field type). Lookup is on the whole text left
# of the colon, so "Date" and "Invoice Date" can never collide.
_LABEL_ALIASES: tuple[tuple[str, str, str], ...] = (
    # -- document identifiers -------------------------------------------
    ("invoice no", "invoice_no", "doc_id"),
    ("invoice no.", "invoice_no", "doc_id"),
    ("invoice number", "invoice_no", "doc_id"),
    ("invoice#", "invoice_no", "doc_id"),
    ("invoice #", "invoice_no", "doc_id"),
    ("fir no", "fir_no", "doc_id"),
    ("fir number", "fir_no", "doc_id"),
    ("case no", "case_no", "doc_id"),
    ("reference no", "reference_no", "doc_id"),
    ("आवेदन संख्या", "application_no", "doc_id"),
    # -- dates ------------------------------------------------------------
    ("invoice date", "invoice_date", "date"),
    ("issue date", "invoice_date", "date"),
    ("due date", "due_date", "date"),
    ("occurrence date", "occurrence_date", "date"),
    ("report date", "date", "date"),
    ("date", "date", "date"),
    ("दिनांक", "date", "date"),
    ("जन्म तिथि", "dob", "date"),
    # -- people / organisations ------------------------------------------
    ("complainant name", "complainant_name", "person_name"),
    ("accused name", "accused_name", "person_name"),
    ("applicant name", "applicant_name", "person_name"),
    ("customer name", "customer_name", "person_name"),
    ("vendor name", "vendor_name", "org_name"),
    ("father name", "father_name", "person_name"),
    ("father's name", "father_name", "person_name"),
    ("पिता का नाम", "father_name", "person_name"),
    ("name", "name", "person_name"),
    ("नाम", "name", "person_name"),
    # -- addresses --------------------------------------------------------
    ("registered office", "vendor_address", "address"),
    ("customer address", "customer_address", "address"),
    ("address", "address", "address"),
    ("पता", "address", "address"),
    # -- contacts ---------------------------------------------------------
    ("complainant phone", "complainant_phone", "phone"),
    ("customer phone", "customer_phone", "phone"),
    ("vendor phone", "vendor_phone", "phone"),
    ("mobile number", "phone", "phone"),
    ("phone number", "phone", "phone"),
    ("phone no", "phone", "phone"),
    ("mobile no", "phone", "phone"),
    ("mobile", "phone", "phone"),
    ("phone", "phone", "phone"),
    ("complainant email", "complainant_email", "email"),
    ("customer email", "customer_email", "email"),
    ("vendor email", "vendor_email", "email"),
    ("email id", "email", "email"),
    ("e-mail", "email", "email"),
    ("email", "email", "email"),
    ("मोबाइल नंबर", "phone", "phone"),
    ("मोबाइल", "phone", "phone"),
    ("फोन नंबर", "phone", "phone"),
    ("ईमेल", "email", "email"),
    # -- money ------------------------------------------------------------
    ("grand total", "total", "currency_amount"),
    ("total amount", "total", "currency_amount"),
    ("amount payable", "total", "currency_amount"),
    ("sub total", "subtotal", "currency_amount"),
    ("subtotal", "subtotal", "currency_amount"),
    ("cgst", "cgst", "currency_amount"),
    ("sgst", "sgst", "currency_amount"),
    ("igst", "igst", "currency_amount"),
    ("total", "total", "currency_amount"),
    ("amount", "amount", "currency_amount"),
    ("राशि", "amount", "currency_amount"),
    # -- identifiers / places --------------------------------------------
    ("gstin", "gstin", "id_number"),
    ("gst no", "gstin", "id_number"),
    ("pan", "pan", "id_number"),
    ("aadhaar no", "aadhaar", "id_number"),
    ("police station", "police_station", "text"),
    ("district", "district", "text"),
    ("place", "place", "text"),
    ("state", "state", "text"),
    ("थाना", "police_station", "text"),
)

def _build_label_fields() -> dict[str, tuple[str, str]]:
    table: dict[str, tuple[str, str]] = {}
    for alias, name, field_type in _LABEL_ALIASES:
        table.setdefault(alias.strip().lower(), (name, field_type))
    return table


#: normalised label -> (field name, field type)
LABEL_FIELDS: dict[str, tuple[str, str]] = _build_label_fields()

#: Exact single-line document titles.
KNOWN_TITLES: frozenset[str] = frozenset(
    {
        "invoice",
        "receipt",
        "bill",
        "fir",
        "first information report",
        "application",
        "application form",
        "आवेदन पत्र",
    }
)

# Confidence assigned at extraction time, before validation adjusts it.
_CONF_LABEL = 0.90
_CONF_LABEL_VARIANT = 0.82
_CONF_HEURISTIC = 0.65
_CONF_LINE_ITEM = 0.95
_CONF_UNPARSED = 0.45

_COLON_SEP = re.compile(r"^(?P<label>[^:：]+)[:：]\s*(?P<value>\S.*)$")
_SPACE_SEP = re.compile(r"^(?P<label>\S.*?)(?:\t| {2,})(?P<value>\S.*)$")
_PUNCT_RE = re.compile(r"[^0-9a-zऀ-ॿ ]+")
_AT_SUFFIX_RE = re.compile(r"\s*[@%].*$")

_TABLE_RE = re.compile(r"\|")
_DESC_COL_RE = re.compile(r"^(description|details|item|particulars)", re.IGNORECASE)
_QTY_COL_RE = re.compile(r"^(qty|quantity|nos|units?)", re.IGNORECASE)
_RATE_COL_RE = re.compile(r"^(rate|price|unit)", re.IGNORECASE)
_AMOUNT_COL_RE = re.compile(r"^(amount|total|value)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _norm_label(label: str) -> str:
    return collapse_ws(nfkc(label)).lower().rstrip(".: ")


def _label_lookup(raw_label: str) -> tuple[str, str, bool] | None:
    """Return ``(field_name, field_type, exact)`` for a raw label, or ``None``."""
    key = _norm_label(raw_label)
    if not key:
        return None
    candidates = [key]
    without_qualifier = _AT_SUFFIX_RE.sub("", key).strip()  # "cgst @ 9%" -> "cgst"
    if without_qualifier:
        candidates.append(without_qualifier)
    without_punct = collapse_ws(_PUNCT_RE.sub(" ", key))  # "gst-no" -> "gst no"
    if without_punct:
        candidates.append(without_punct)
    for index, candidate in enumerate(candidates):
        hit = LABEL_FIELDS.get(candidate)
        if hit:
            return hit[0], hit[1], index == 0
    return None


def _parse_label_line(line: str) -> tuple[str, str, str, str, bool] | None:
    """Split ``label: value`` and resolve the label. ``None`` if not a field."""
    match = _COLON_SEP.match(line)
    if match is None:
        match = _SPACE_SEP.match(line)
        if match is None:
            return None
    label = match.group("label")
    value = match.group("value").strip()
    hit = _label_lookup(label)
    if hit is None:
        return None
    field_name, field_type, exact = hit
    return label.strip(), value, field_name, field_type, exact


def _first_number(text: str) -> float | None:
    match = _NUMBER_RE.search(ascii_digits(text))
    if match is None:
        return None
    return float(match.group(0).replace(",", ""))


def _coerce(value: str, field_type: str) -> tuple[Any, bool]:
    """Normalise a raw string to the declared field type.

    Returns ``(normalised_value, ok)``. When ``ok`` is ``False`` the raw string
    is kept so the validator can flag it instead of silently dropping data.
    """
    if field_type == "date":
        parsed = normalize_date(value)
    elif field_type == "currency_amount":
        parsed = normalize_amount(value)
    elif field_type == "phone":
        parsed = normalize_phone(value)
    elif field_type == "email":
        parsed = normalize_email(value)
    elif field_type in ("person_name", "org_name"):
        parsed = normalize_person_name(value)
    else:
        parsed = collapse_ws(nfkc(value))
    if parsed is None or parsed == "":
        return value, False
    return parsed, True


def _find_column(header: list[str], pattern: re.Pattern[str]) -> int | None:
    for index, column in enumerate(header):
        if pattern.match(collapse_ws(column).lower()):
            return index
    return None


def parse_line_items(lines: Iterable[str]) -> list[LineItem]:
    """Parse a ``S.No | Description | Qty | Rate | Amount`` style table."""
    items: list[LineItem] = []
    header_found = False
    indices: tuple[int, int, int, int] | None = None

    for raw_line in lines:
        line = collapse_ws(nfkc(raw_line))
        if not _TABLE_RE.search(line):
            if header_found:
                break  # the table ended
            continue

        parts = [part.strip() for part in line.split("|")]
        if not header_found:
            if not any(_DESC_COL_RE.match(part) for part in parts):
                continue
            desc = _find_column(parts, _DESC_COL_RE)
            amount = _find_column(parts, _AMOUNT_COL_RE)
            if desc is None or amount is None:
                continue
            qty = _find_column(parts, _QTY_COL_RE)
            rate = _find_column(parts, _RATE_COL_RE)
            header_found = True
            indices = (
                desc,
                qty if qty is not None else desc + 1,
                rate if rate is not None else desc + 2,
                amount,
            )
            continue

        if indices is None:  # pragma: no cover - header_found implies indices
            continue
        desc_i, qty_i, rate_i, amount_i = indices
        if len(parts) <= max(desc_i, qty_i, rate_i, amount_i):
            continue
        description = collapse_ws(parts[desc_i])
        amount = normalize_amount(parts[amount_i])
        if not description or amount is None:
            continue
        quantity = _first_number(parts[qty_i])
        items.append(
            LineItem(
                description=description,
                quantity=quantity if quantity is not None else 1.0,
                unit_price=normalize_amount(parts[rate_i]),
                amount=amount,
            )
        )
    return items


def detect_doc_type(doc: Document) -> str:
    """Classify a document from its title and the fields that fired."""
    title = str(doc.value("title", "") or "").casefold().strip()
    if title in {"invoice", "receipt", "bill"} or "invoice_no" in doc.fields or "subtotal" in doc.fields:
        return "invoice"
    if title in {"fir", "first information report"} or "fir_no" in doc.fields or "police_station" in doc.fields:
        return "fir"
    if title in {"आवेदन पत्र", "application form", "application"} or "application_no" in doc.fields:
        return "form"
    return "generic"


def extract_document(
    text: str,
    *,
    source: str | None = None,
    provider: str = "rules",
) -> Document:
    """Run the whole rule pipeline over ``text`` and return a :class:`Document`."""
    doc = Document(doc_type="generic", provider=provider, source=source)
    seen_label_line = False

    for raw_line in text.splitlines():
        # `line_raw` keeps tabs and runs of spaces so the `label<sep>value`
        # fallback can still see them; `line` is the tidy form used for the
        # title / organisation heuristics.
        line_raw = nfkc(raw_line).strip()
        if not line_raw or line_raw.startswith("#"):
            continue
        line = collapse_ws(line_raw)

        parsed = _parse_label_line(line_raw)
        if parsed is not None:
            seen_label_line = True
            _label, value, field_name, field_type, exact = parsed
            if field_name in doc.fields:
                continue  # first occurrence wins
            normalised, ok = _coerce(value, field_type)
            doc.set_field(
                name=field_name,
                value=normalised,
                field_type=field_type,
                confidence=(_CONF_LABEL if exact else _CONF_LABEL_VARIANT) if ok else _CONF_UNPARSED,
                raw=value,
                provider=provider,
            )
            continue

        if ":" in line:
            seen_label_line = True  # a labelled line we do not model
            continue

        if line.casefold() in KNOWN_TITLES:
            if "title" not in doc.fields:
                doc.set_field("title", line, "text", _CONF_HEURISTIC, raw=line, provider=provider)
            continue

        if not seen_label_line and "org_name" not in doc.fields and not _TABLE_RE.search(line):
            doc.set_field("org_name", line, "org_name", _CONF_HEURISTIC, raw=line, provider=provider)

    doc.line_items.extend(parse_line_items(text.splitlines()))

    doc.doc_type = detect_doc_type(doc)
    return doc
