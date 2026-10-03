"""Document schema: dataclasses plus a JSON Schema dump.

The schema is deliberately small and flat:

* a document has a ``doc_type`` (invoice / form / fir / receipt / generic),
* a mapping of named ``fields`` — each :class:`Field` carries its own value,
  declared ``field_type`` and a ``confidence`` in ``[0, 1]``,
* an optional ``line_items`` list for invoices and receipts.

Everything here is stdlib-only so the schema can be imported by consumers who
have no third-party packages installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from typing import Any, Mapping

__all__ = [
    "FIELD_TYPES",
    "DOC_TYPES",
    "Field",
    "LineItem",
    "Document",
    "json_schema",
]

#: Every field type the schema understands.
FIELD_TYPES: tuple[str, ...] = (
    "person_name",
    "org_name",
    "address",
    "date",
    "currency_amount",
    "phone",
    "email",
    "doc_id",
    "id_number",
    "text",
)

#: Document classes the extractor can report.
DOC_TYPES: tuple[str, ...] = ("invoice", "form", "fir", "receipt", "generic")

_TYPE_HELP: dict[str, str] = {
    "person_name": "A human name (complainant, applicant, customer, ...).",
    "org_name": "An organisation / issuing authority name.",
    "address": "A postal address, free text.",
    "date": "A calendar date, ISO-8601 string YYYY-MM-DD.",
    "currency_amount": "A monetary amount as a JSON number (Indian rupees).",
    "phone": "A 10-digit Indian mobile number as a string.",
    "email": "A lowercase email address.",
    "doc_id": "A document identifier (invoice no, FIR no, application no).",
    "id_number": "A structured identity number (GSTIN, PAN, Aadhaar, ...).",
    "text": "Any other short free-text value.",
}


def _check_field_type(field_type: str) -> str:
    if field_type not in FIELD_TYPES:
        raise ValueError(
            f"unknown field_type {field_type!r}; expected one of {', '.join(FIELD_TYPES)}"
        )
    return field_type


def _check_confidence(confidence: float) -> float:
    value = float(confidence)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be within [0, 1], got {confidence!r}")
    return value


@dataclass
class Field:
    """A single extracted value with its type and confidence.

    ``confidence`` is a *heuristic score*, not a calibrated probability: it
    starts from how the value was found (exact label match > fuzzy label >
    layout heuristic > unparsable) and the validator then nudges it by
    +0.03 / -0.15 / -0.40 per check. It ranks fields within a document; it
    does not mean "93% likely to be correct".
    """

    name: str
    value: Any
    field_type: str = "text"
    confidence: float = 0.5
    raw: str | None = None
    provider: str = "rules"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name must be non-empty")
        self.field_type = _check_field_type(self.field_type)
        self.confidence = _check_confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "type": self.field_type,
            "confidence": round(self.confidence, 4),
            "raw": self.raw,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, name: str, payload: Mapping[str, Any]) -> "Field":
        """Accept both the rich form and a bare scalar shorthand."""
        if isinstance(payload, Mapping) and "value" in payload:
            return cls(
                name=name,
                value=payload["value"],
                field_type=payload.get("type", "text"),
                confidence=float(payload.get("confidence", 0.5)),
                raw=payload.get("raw"),
                provider=payload.get("provider", "rules"),
            )
        return cls(name=name, value=payload)


@dataclass
class LineItem:
    """One row of an invoice / receipt table."""

    description: str
    quantity: float = 1.0
    unit_price: float | None = None
    amount: float = 0.0

    def __post_init__(self) -> None:
        self.description = str(self.description)
        self.quantity = float(self.quantity)
        self.amount = float(self.amount)
        if self.unit_price is not None:
            self.unit_price = float(self.unit_price)

    @property
    def arithmetic_ok(self) -> bool:
        """True when ``quantity * unit_price`` agrees with ``amount``."""
        if self.unit_price is None:
            return True
        expected = self.quantity * self.unit_price
        # The unit price is printed to the paise, so each unit may differ by
        # up to half a paisa (qty * 0.005) and the product itself by another
        # paisa. Anything beyond that is a real inconsistency — a flat "0.5%
        # of the amount" would swallow a ₹100 error on a ₹50,000 line.
        tolerance = 0.01 + abs(self.quantity) * 0.005
        return abs(expected - self.amount) <= tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "amount": self.amount,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LineItem":
        return cls(
            description=payload.get("description", ""),
            quantity=payload.get("quantity", 1.0),
            unit_price=payload.get("unit_price"),
            amount=payload.get("amount", 0.0),
        )


@dataclass
class Document:
    """The unit of output: one document turned into structured JSON."""

    doc_type: str = "generic"
    fields: dict[str, Field] = dc_field(default_factory=dict)
    line_items: list[LineItem] = dc_field(default_factory=list)
    provider: str = "rules"
    source: str | None = None
    validation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.doc_type not in DOC_TYPES:
            raise ValueError(
                f"unknown doc_type {self.doc_type!r}; expected one of {', '.join(DOC_TYPES)}"
            )

    # -- field helpers -------------------------------------------------
    def set_field(
        self,
        name: str,
        value: Any,
        field_type: str = "text",
        confidence: float = 0.5,
        raw: str | None = None,
        provider: str | None = None,
    ) -> Field:
        field = Field(
            name=name,
            value=value,
            field_type=field_type,
            confidence=confidence,
            raw=raw,
            provider=provider or self.provider,
        )
        self.fields[name] = field
        return field

    def get(self, name: str, default: Field | None = None) -> Field | None:
        return self.fields.get(name, default)

    def value(self, name: str, default: Any = None) -> Any:
        field = self.fields.get(name)
        return default if field is None else field.value

    def values(self) -> dict[str, Any]:
        """Flat ``{name: value}`` view — the shape used when scoring."""
        return {name: field.value for name, field in self.fields.items()}

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "doc_type": self.doc_type,
            "provider": self.provider,
            "source": self.source,
            "fields": {name: f.to_dict() for name, f in self.fields.items()},
            "line_items": [item.to_dict() for item in self.line_items],
        }
        if self.validation is not None:
            payload["validation"] = self.validation
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Document":
        raw_fields = payload.get("fields") or {}
        doc = cls(
            doc_type=payload.get("doc_type", "generic"),
            provider=payload.get("provider", "rules"),
            source=payload.get("source"),
            validation=payload.get("validation"),
        )
        if not isinstance(raw_fields, Mapping):
            raise ValueError("'fields' must be a JSON object")
        for name, value in raw_fields.items():
            doc.fields[str(name)] = Field.from_dict(str(name), value)
        for item in payload.get("line_items") or []:
            doc.line_items.append(LineItem.from_dict(item))
        return doc

    @classmethod
    def from_json(cls, text: str) -> "Document":
        return cls.from_dict(json.loads(text))


def json_schema() -> dict[str, Any]:
    """Return a JSON Schema (draft 2020-12 style) describing :class:`Document`."""
    field_schema: dict[str, Any] = {
        "type": "object",
        "required": ["value"],
        "additionalProperties": False,
        "properties": {
            "value": {
                "description": "Normalised value: ISO date, JSON number for money, "
                "10-digit string for phone, otherwise a string.",
            },
            "type": {"enum": list(FIELD_TYPES)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "raw": {"type": ["string", "null"]},
            "provider": {"type": "string"},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.invalid/sarvam-vision-doc-extract/document.schema.json",
        "title": "ExtractedDocument",
        "type": "object",
        "required": ["doc_type", "fields"],
        "additionalProperties": False,
        "properties": {
            "doc_type": {"enum": list(DOC_TYPES)},
            "provider": {"type": "string"},
            "source": {"type": ["string", "null"]},
            "fields": {
                "type": "object",
                "description": "Named fields; each value describes its own type and confidence.",
                "additionalProperties": field_schema,
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["description"],
                    "additionalProperties": False,
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": ["number", "null"]},
                        "amount": {"type": "number"},
                    },
                },
            },
            "validation": {
                "type": ["object", "null"],
                "description": "Validator output: pass/fail flag plus issue list.",
                "properties": {
                    "passed": {"type": "boolean"},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "code": {"type": "string"},
                                "severity": {"enum": ["error", "warning"]},
                                "message": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "$defs": {
            # JSON Schema requires "description" to be a *string*, so the
            # per-type help is folded into one sentence rather than shipped
            # as an object that no validator would accept.
            "fieldType": {
                "type": "string",
                "enum": list(FIELD_TYPES),
                "description": "Field type meanings: "
                + "; ".join(f"{key} = {_TYPE_HELP[key]}" for key in FIELD_TYPES),
            },
        },
    }
