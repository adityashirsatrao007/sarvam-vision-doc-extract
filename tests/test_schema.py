"""Unit tests for the document schema and its JSON Schema dump."""

from __future__ import annotations

import json
import unittest

from docextract.schema import (
    DOC_TYPES,
    FIELD_TYPES,
    Document,
    Field,
    LineItem,
    json_schema,
)


class TestField(unittest.TestCase):
    def test_defaults(self) -> None:
        field = Field(name="title", value="INVOICE")
        self.assertEqual(field.field_type, "text")
        self.assertEqual(field.confidence, 0.5)
        self.assertEqual(field.provider, "rules")
        self.assertIsNone(field.raw)

    def test_rejects_unknown_field_type(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Field(name="x", value="y", field_type="iban")
        self.assertIn("unknown field_type", str(ctx.exception))

    def test_rejects_out_of_range_confidence(self) -> None:
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                Field(name="x", value="y", confidence=bad)

    def test_rejects_empty_name(self) -> None:
        with self.assertRaises(ValueError):
            Field(name="", value="y")

    def test_to_dict_contains_confidence_and_type(self) -> None:
        field = Field(name="phone", value="9876543210", field_type="phone", confidence=0.93)
        payload = field.to_dict()
        self.assertEqual(payload["value"], "9876543210")
        self.assertEqual(payload["type"], "phone")
        self.assertAlmostEqual(payload["confidence"], 0.93)

    def test_from_dict_accepts_rich_and_scalar_forms(self) -> None:
        rich = Field.from_dict(
            "total",
            {"value": 103368.0, "type": "currency_amount", "confidence": 0.9, "raw": "₹1,03,368.00"},
        )
        self.assertEqual(rich.field_type, "currency_amount")
        self.assertEqual(rich.raw, "₹1,03,368.00")

        bare = Field.from_dict("title", "INVOICE")
        self.assertEqual(bare.value, "INVOICE")
        self.assertEqual(bare.field_type, "text")


class TestLineItem(unittest.TestCase):
    def test_arithmetic_ok(self) -> None:
        item = LineItem(description="work", quantity=40, unit_price=1250, amount=50000)
        self.assertTrue(item.arithmetic_ok)

    def test_arithmetic_mismatch(self) -> None:
        item = LineItem(description="work", quantity=40, unit_price=1250, amount=49000)
        self.assertFalse(item.arithmetic_ok)

    def test_missing_unit_price_is_not_flagged(self) -> None:
        item = LineItem(description="work", amount=100)
        self.assertTrue(item.arithmetic_ok)

    def test_round_trip(self) -> None:
        item = LineItem(description="work", quantity=2, unit_price=50, amount=100)
        self.assertEqual(LineItem.from_dict(item.to_dict()).to_dict(), item.to_dict())


class TestDocument(unittest.TestCase):
    def test_rejects_unknown_doc_type(self) -> None:
        with self.assertRaises(ValueError):
            Document(doc_type="passport")

    def test_field_helpers(self) -> None:
        doc = Document(doc_type="invoice")
        doc.set_field("invoice_no", "VA-2026-0417", "doc_id", 0.9)
        self.assertEqual(doc.value("invoice_no"), "VA-2026-0417")
        self.assertEqual(doc.value("nope", "fallback"), "fallback")
        self.assertEqual(doc.values(), {"invoice_no": "VA-2026-0417"})
        self.assertEqual(doc.get("nope"), None)

    def test_json_round_trip_preserves_fields_and_items(self) -> None:
        doc = Document(doc_type="invoice", source="data/samples/invoice_en.txt")
        doc.set_field("total", 103368.0, "currency_amount", 0.93, raw="₹1,03,368.00")
        doc.line_items.append(LineItem("work", 40, 1250.0, 50000.0))
        doc.validation = {"passed": True, "issues": []}

        restored = Document.from_json(doc.to_json())
        self.assertEqual(restored.doc_type, "invoice")
        self.assertEqual(restored.source, "data/samples/invoice_en.txt")
        self.assertEqual(restored.value("total"), 103368.0)
        self.assertEqual(restored.fields["total"].confidence, 0.93)
        self.assertEqual(restored.line_items[0].to_dict(), doc.line_items[0].to_dict())
        self.assertTrue(restored.validation["passed"])

    def test_to_json_is_valid_json_with_unicode(self) -> None:
        doc = Document(doc_type="form")
        doc.set_field("name", "सुनीता देशपांडे", "person_name", 0.9)
        payload = json.loads(doc.to_json())
        self.assertEqual(payload["fields"]["name"]["value"], "सुनीता देशपांडे")
        # ensure_ascii=False means the raw Devanagari survives into the text
        self.assertIn("सुनीता", doc.to_json())


class TestJsonSchema(unittest.TestCase):
    def test_shape(self) -> None:
        schema = json_schema()
        self.assertEqual(schema["title"], "ExtractedDocument")
        self.assertIn("$schema", schema)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["doc_type", "fields"])
        self.assertIn("line_items", schema["properties"])
        self.assertIn("validation", schema["properties"])

    def test_enums_match_module_constants(self) -> None:
        schema = json_schema()
        self.assertEqual(schema["properties"]["doc_type"]["enum"], list(DOC_TYPES))
        field_schema = schema["properties"]["fields"]["additionalProperties"]
        self.assertEqual(field_schema["properties"]["type"]["enum"], list(FIELD_TYPES))
        self.assertEqual(field_schema["properties"]["confidence"]["maximum"], 1.0)

    def test_schema_is_serialisable(self) -> None:
        encoded = json.dumps(json_schema())
        self.assertIn("ExtractedDocument", encoded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
