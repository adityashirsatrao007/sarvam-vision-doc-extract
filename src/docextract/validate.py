"""Field validators, cross-field consistency checks and confidence scoring.

Every extracted field is checked against a rule for its ``field_type``. Each
check adjusts the field's confidence:

* pass   -> ``+0.03`` (capped at ``0.99``),
* warning -> ``-0.15``,
* error  -> ``-0.40`` (floor of ``0.05``).

Cross-field checks (line items summing to the subtotal, taxes adding up to the
total, issue/due dates) report issues but never invent data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from datetime import date, timedelta
from typing import Any, Callable

from .schema import Document

__all__ = [
    "Issue",
    "ValidationReport",
    "validate_document",
    "TAX_FIELDS",
]

#: Field names treated as tax lines when checking ``total``.
TAX_FIELDS: frozenset[str] = frozenset({"cgst", "sgst", "igst", "gst", "tax", "vat", "cess"})

_TOLERANCE = 0.01
_PENALTY_PASS = 0.03
_PENALTY_WARNING = 0.15
_PENALTY_ERROR = 0.40

_INDIAN_MOBILE_RE = re.compile(r"^[6-9]\d{9}$")
_TEN_DIGITS_RE = re.compile(r"^\d{10}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/\-_.# ]{2,39}$")
_ID_NUMBER_RE = re.compile(r"^[A-Z0-9][A-Z0-9-]{5,19}$")
_GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z]\d$")
_PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_AADHAAR_RE = re.compile(r"^\d{12}$")
_PINCODE_RE = re.compile(r"\b\d{6}\b")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class Issue:
    """One problem found by the validator."""

    field: str
    code: str
    message: str
    severity: str = "error"  # "error" | "warning"

    def __post_init__(self) -> None:
        if self.severity not in ("error", "warning"):
            raise ValueError(f"severity must be 'error' or 'warning', got {self.severity!r}")

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity}] {self.field}: {self.message} ({self.code})"


@dataclass
class ValidationReport:
    """Result of validating one document."""

    issues: list[Issue] = dc_field(default_factory=list)
    confidences: dict[str, float] = dc_field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def min_confidence(self) -> float:
        return min(self.confidences.values()) if self.confidences else 0.0

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences.values()) / len(self.confidences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "mean_confidence": round(self.mean_confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "issues": [issue.to_dict() for issue in self.issues],
        }


# --- per-field-type validators ---------------------------------------------
# Each returns (severity, code, message) or None when the value is acceptable.


def _check_date(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return "error", "date_invalid", f"{value!r} is not an ISO date (YYYY-MM-DD)"
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return "error", "date_invalid", f"{value!r} is not a real calendar date"
    if not 1900 <= parsed.year <= 2100:
        return "error", "date_out_of_range", f"year {parsed.year} is outside 1900-2100"
    if parsed > date.today() + timedelta(days=365):
        return "warning", "date_in_future", f"{value} is more than a year in the future"
    return None


def _check_currency_amount(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "error", "amount_invalid", f"{value!r} is not a number"
    if value < 0:
        return "error", "amount_negative", f"{value} is negative"
    if value == 0:
        return "warning", "amount_zero", "amount is zero"
    return None


def _check_phone(value: Any) -> tuple[str, str, str] | None:
    text = str(value)
    if not _TEN_DIGITS_RE.match(text):
        return "error", "phone_invalid", f"{text!r} is not a 10-digit number"
    if not _INDIAN_MOBILE_RE.match(text):
        return "warning", "phone_not_mobile", f"{text} is 10 digits but not a 6-9 prefixed mobile"
    return None


def _check_email(value: Any) -> tuple[str, str, str] | None:
    text = str(value)
    if not _EMAIL_RE.match(text):
        return "error", "email_invalid", f"{text!r} is not a valid email address"
    if text != text.lower() or " " in text:
        return "warning", "email_not_normalised", f"{text} should be lowercase without spaces"
    return None


def _check_doc_id(value: Any) -> tuple[str, str, str] | None:
    text = str(value)
    if not _DOC_ID_RE.match(text):
        return "error", "doc_id_invalid", f"{text!r} is not a plausible document id"
    return None


def _check_id_number(value: Any) -> tuple[str, str, str] | None:
    text = str(value).upper()
    if not _ID_NUMBER_RE.match(text):
        return "error", "id_number_invalid", f"{text!r} is not a plausible id number"
    if len(text) == 15 and not _GSTIN_RE.match(text):
        return "error", "gstin_invalid", f"{text} does not match the 15-character GSTIN pattern"
    if len(text) == 10 and text[:5].isalpha() and not _PAN_RE.match(text):
        return "error", "pan_invalid", f"{text} does not match the PAN pattern"
    if len(text) == 12 and text.isdigit() and not _AADHAAR_RE.match(text):
        return "error", "aadhaar_invalid", f"{text} does not match the 12-digit Aadhaar pattern"
    return None


def _check_person_name(value: Any) -> tuple[str, str, str] | None:
    text = str(value)
    if len(text) < 2:
        return "error", "name_too_short", f"{text!r} is too short to be a name"
    if not any(ch.isalpha() for ch in text):
        return "error", "name_no_letters", f"{text!r} contains no letters"
    if any(ch.isdigit() for ch in text):
        return "warning", "name_has_digits", f"{text!r} contains digits"
    return None


def _check_org_name(value: Any) -> tuple[str, str, str] | None:
    text = str(value)
    if len(text) < 2 or not any(ch.isalpha() for ch in text):
        return "error", "org_name_invalid", f"{text!r} is not a plausible organisation name"
    return None


def _check_address(value: Any) -> tuple[str, str, str] | None:
    text = str(value)
    if len(text) < 10:
        return "error", "address_too_short", f"{text!r} is too short to be an address"
    if not _PINCODE_RE.search(text):
        return "warning", "address_no_pincode", "no 6-digit PIN code found in the address"
    return None


def _check_text(value: Any) -> tuple[str, str, str] | None:
    if not str(value).strip():
        return "error", "text_empty", "value is empty"
    return None


_FIELD_CHECKS: dict[str, Callable[[Any], tuple[str, str, str] | None]] = {
    "date": _check_date,
    "currency_amount": _check_currency_amount,
    "phone": _check_phone,
    "email": _check_email,
    "doc_id": _check_doc_id,
    "id_number": _check_id_number,
    "person_name": _check_person_name,
    "org_name": _check_org_name,
    "address": _check_address,
    "text": _check_text,
}


# --- cross-field checks -----------------------------------------------------


def _check_line_item_sum(doc: Document, report: ValidationReport) -> None:
    subtotal = doc.fields.get("subtotal")
    if subtotal is None or not doc.line_items:
        return
    expected = sum(item.amount for item in doc.line_items)
    actual = subtotal.value
    if isinstance(actual, (int, float)) and abs(expected - actual) > _TOLERANCE:
        report.issues.append(
            Issue(
                field="subtotal",
                code="subtotal_mismatch",
                severity="error",
                message=f"line items sum to {expected:.2f} but subtotal says {actual:.2f}",
            )
        )
        report.confidences["subtotal"] = max(
            0.05, report.confidences.get("subtotal", 0.5) - _PENALTY_ERROR
        )


def _check_total(doc: Document, report: ValidationReport) -> None:
    total = doc.fields.get("total")
    subtotal = doc.fields.get("subtotal")
    if total is None or subtotal is None:
        return
    if not isinstance(total.value, (int, float)) or not isinstance(subtotal.value, (int, float)):
        return
    taxes = 0.0
    have_tax = False
    for name in TAX_FIELDS:
        field = doc.fields.get(name)
        if field is not None and isinstance(field.value, (int, float)):
            taxes += float(field.value)
            have_tax = True
    if not have_tax:
        return
    expected = float(subtotal.value) + taxes
    if abs(expected - float(total.value)) > _TOLERANCE:
        report.issues.append(
            Issue(
                field="total",
                code="total_mismatch",
                severity="error",
                message=(
                    f"subtotal {subtotal.value:.2f} + taxes {taxes:.2f} = {expected:.2f} "
                    f"but total says {total.value:.2f}"
                ),
            )
        )
        report.confidences["total"] = max(
            0.05, report.confidences.get("total", 0.5) - _PENALTY_ERROR
        )


def _check_due_date(doc: Document, report: ValidationReport) -> None:
    due = doc.fields.get("due_date")
    invoice = doc.fields.get("invoice_date")
    if due is None or invoice is None:
        return
    if not (isinstance(due.value, str) and isinstance(invoice.value, str)):
        return
    if not (_DATE_RE.match(due.value) and _DATE_RE.match(invoice.value)):
        return
    if due.value < invoice.value:
        report.issues.append(
            Issue(
                field="due_date",
                code="due_date_before_invoice",
                severity="error",
                message=f"due date {due.value} precedes invoice date {invoice.value}",
            )
        )
        report.confidences["due_date"] = max(
            0.05, report.confidences.get("due_date", 0.5) - _PENALTY_ERROR
        )


def _check_line_item_arithmetic(doc: Document, report: ValidationReport) -> None:
    for index, item in enumerate(doc.line_items, start=1):
        if item.unit_price is None:
            continue
        expected = item.quantity * item.unit_price
        tolerance = max(_TOLERANCE, abs(item.amount) * 0.005)
        if abs(expected - item.amount) > tolerance:
            report.issues.append(
                Issue(
                    field=f"line_items[{index}]",
                    code="line_item_arithmetic",
                    severity="error",
                    message=(
                        f"{item.quantity:g} x {item.unit_price:.2f} = {expected:.2f} "
                        f"but amount says {item.amount:.2f}"
                    ),
                )
            )


def validate_document(doc: Document) -> ValidationReport:
    """Validate every field, run the cross-field checks and score confidence.

    Field confidences inside ``doc`` are updated in place so the JSON written
    by the CLI reflects what the validator believes.
    """
    report = ValidationReport()

    for name, field in doc.fields.items():
        check = _FIELD_CHECKS.get(field.field_type, _check_text)
        confidence = float(field.confidence)
        failure = check(field.value)
        if failure is None:
            confidence = min(0.99, confidence + _PENALTY_PASS)
        else:
            severity, code, message = failure
            report.issues.append(Issue(field=name, code=code, severity=severity, message=message))
            penalty = _PENALTY_ERROR if severity == "error" else _PENALTY_WARNING
            floor = 0.05 if severity == "error" else 0.15
            confidence = max(floor, confidence - penalty)
        field.confidence = round(confidence, 4)
        report.confidences[name] = field.confidence

    if not doc.fields:
        report.issues.append(
            Issue(
                field="<document>",
                code="no_fields_extracted",
                severity="warning",
                message="no fields were extracted at all",
            )
        )

    _check_line_item_sum(doc, report)
    _check_total(doc, report)
    _check_due_date(doc, report)
    _check_line_item_arithmetic(doc, report)

    doc.validation = report.to_dict()
    return report
