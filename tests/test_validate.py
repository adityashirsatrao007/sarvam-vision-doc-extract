"""Unit tests for field validators, cross-field checks and confidence scoring."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

from docextract.schema import Document, LineItem
from docextract.validate import TAX_FIELDS, Issue, ValidationReport, validate_document


def build_invoice(**overrides: object) -> Document:
    """A document that should validate cleanly; ``overrides`` poke one field."""
    doc = Document(doc_type="invoice", provider="rules")
    fields: dict[str, tuple[object, str]] = {
        "title": ("INVOICE", "text"),
        "org_name": ("VEDA ANALYTICS PRIVATE LIMITED", "org_name"),
        "gstin": ("29AABCV1234K1Z5", "id_number"),
        "vendor_address": (
            "42, 5th Cross, Indiranagar, Bengaluru 560038, Karnataka",
            "address",
        ),
        "vendor_phone": ("9845012345", "phone"),
        "vendor_email": ("billing@vedaanalytics.in", "email"),
        "invoice_no": ("VA-2026-0417", "doc_id"),
        "invoice_date": ("2026-02-12", "date"),
        "due_date": ("2026-02-26", "date"),
        "customer_name": ("Meera Krishnan", "person_name"),
        "customer_address": ("F-7, Sector 22, Noida, Uttar Pradesh 201301", "address"),
        "customer_phone": ("9876543210", "phone"),
        "customer_email": ("meera.k@example.com", "email"),
        "subtotal": (87600.0, "currency_amount"),
        "cgst": (7884.0, "currency_amount"),
        "sgst": (7884.0, "currency_amount"),
        "total": (103368.0, "currency_amount"),
    }
    for name, (value, field_type) in fields.items():
        if name in overrides and overrides[name] is None:
            continue
        doc.set_field(name, overrides.get(name, value), field_type, 0.9)
    doc.line_items = [
        LineItem("Document digitisation services", 40, 1250.0, 50000.0),
        LineItem("OCR model tuning and evaluation", 12, 2500.0, 30000.0),
        LineItem("Handwritten field review", 8, 950.0, 7600.0),
    ]
    return doc


def codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


class TestIssue(unittest.TestCase):
    def test_severity_is_validated(self) -> None:
        Issue(field="x", code="y", message="z")
        with self.assertRaises(ValueError):
            Issue(field="x", code="y", message="z", severity="fatal")

    def test_to_dict_shape(self) -> None:
        payload = Issue(field="phone", code="phone_invalid", message="bad").to_dict()
        self.assertEqual(
            set(payload), {"field", "code", "severity", "message"}
        )


class TestCleanDocument(unittest.TestCase):
    def test_valid_invoice_passes_without_issues(self) -> None:
        report = validate_document(build_invoice())
        self.assertTrue(report.passed, msg=str(report.issues))
        self.assertEqual(report.issues, [])
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])
        # every field starts at 0.90 and passes its check → +0.03 each:
        # 17 fields, so mean and min are both 0.93
        self.assertAlmostEqual(report.mean_confidence, 0.93, places=6)
        self.assertAlmostEqual(report.min_confidence, 0.93, places=6)

    def test_report_is_written_back_onto_the_document(self) -> None:
        doc = build_invoice()
        report = validate_document(doc)
        self.assertIsNotNone(doc.validation)
        assert doc.validation is not None
        self.assertTrue(doc.validation["passed"])
        self.assertEqual(doc.validation["errors"], len(report.errors))
        self.assertAlmostEqual(
            doc.validation["mean_confidence"], report.mean_confidence, places=3
        )


class TestFieldValidators(unittest.TestCase):
    def _report_for(self, **overrides: object) -> ValidationReport:
        return validate_document(build_invoice(**overrides))

    def test_bad_phone_is_an_error(self) -> None:
        report = self._report_for(customer_phone="12345")
        self.assertIn("phone_invalid", codes(report))
        self.assertFalse(report.passed)

    def test_ten_digits_but_not_a_mobile_is_a_warning(self) -> None:
        report = self._report_for(customer_phone="5123456789")
        self.assertIn("phone_not_mobile", codes(report))
        self.assertEqual(report.errors, [])
        self.assertTrue(report.passed)

    def test_bad_email_is_an_error(self) -> None:
        report = self._report_for(customer_email="meera.k at example.com")
        self.assertIn("email_invalid", codes(report))

    def test_unparseable_date_is_an_error(self) -> None:
        report = self._report_for(invoice_date="13/45/2026")
        self.assertIn("date_invalid", codes(report))

    def test_date_more_than_a_year_ahead_warns(self) -> None:
        future = (date.today() + timedelta(days=900)).isoformat()
        report = self._report_for(due_date=future)
        self.assertIn("date_in_future", codes(report))
        self.assertTrue(report.passed)  # warnings never fail the document

    def test_negative_amount_is_an_error(self) -> None:
        report = self._report_for(total=-1.0)
        self.assertIn("amount_negative", codes(report))

    def test_zero_amount_warns(self) -> None:
        doc = Document(doc_type="receipt")
        doc.set_field("amount", 0.0, "currency_amount", 0.9)
        report = validate_document(doc)
        self.assertIn("amount_zero", codes(report))
        self.assertTrue(report.passed)

    def test_address_without_pincode_warns(self) -> None:
        report = self._report_for(customer_address="Flat 7, Sector 22, Noida")
        self.assertIn("address_no_pincode", codes(report))

    def test_short_address_is_an_error(self) -> None:
        report = self._report_for(customer_address="Pune")
        self.assertIn("address_too_short", codes(report))

    def test_name_with_digits_warns(self) -> None:
        report = self._report_for(customer_name="Meera Krishnan 2")
        self.assertIn("name_has_digits", codes(report))

    def test_bad_doc_id_is_an_error(self) -> None:
        report = self._report_for(invoice_no="!!!")
        self.assertIn("doc_id_invalid", codes(report))
        self.assertFalse(report.passed)

    def test_gstin_pattern_is_enforced(self) -> None:
        report = self._report_for(gstin="29AABCV1234K1ZZ")
        self.assertIn("gstin_invalid", codes(report))

    def test_pan_pattern_is_enforced(self) -> None:
        doc = Document(doc_type="invoice")
        doc.set_field("pan", "AAAA1212A", "id_number", 0.9)
        self.assertTrue(validate_document(doc).passed)

        doc = Document(doc_type="invoice")
        doc.set_field("pan", "AAAA!212A", "id_number", 0.9)
        self.assertIn("id_number_invalid", codes(validate_document(doc)))


class TestCrossFieldChecks(unittest.TestCase):
    def test_subtotal_mismatch_with_line_items(self) -> None:
        doc = build_invoice()
        doc.fields["subtotal"].value = 99999.0
        report = validate_document(doc)
        self.assertIn("subtotal_mismatch", codes(report))
        self.assertFalse(report.passed)
        message = next(i.message for i in report.issues if i.code == "subtotal_mismatch")
        self.assertEqual(
            message,
            "line items sum to ₹87,600.00 but subtotal says ₹99,999.00",
        )

        # Money tolerance is exactly one paise (0.01 + an epsilon for the
        # float representation of 0.01 itself): 87600.01 must pass …
        one_paise = build_invoice()
        one_paise.fields["subtotal"].value = 87600.01
        self.assertNotIn("subtotal_mismatch", codes(validate_document(one_paise)))
        # … and two paise must not.
        two_paise = build_invoice()
        two_paise.fields["subtotal"].value = 87600.02
        self.assertIn("subtotal_mismatch", codes(validate_document(two_paise)))

    def test_taxes_must_add_up_to_total(self) -> None:
        doc = build_invoice()
        doc.fields["total"].value = 100000.0
        report = validate_document(doc)
        self.assertIn("total_mismatch", codes(report))

        # No tax lines at all: the gap between subtotal and total may be a
        # discount or plain round-off, neither of which is modelled, so it is
        # reported as a warning rather than failing a legitimate document.
        untaxed = Document(doc_type="invoice")
        untaxed.set_field("subtotal", 100.0, "currency_amount", 0.9)
        untaxed.set_field("total", 110.0, "currency_amount", 0.9)
        untaxed_report = validate_document(untaxed)
        self.assertIn("total_without_taxes", codes(untaxed_report))
        self.assertNotIn("total_mismatch", codes(untaxed_report))
        self.assertTrue(untaxed_report.passed)
        self.assertEqual(untaxed_report.errors, [])

    def test_tax_fields_constant(self) -> None:
        self.assertIn("cgst", TAX_FIELDS)
        self.assertIn("sgst", TAX_FIELDS)

    def test_due_date_before_invoice_date(self) -> None:
        doc = build_invoice()
        doc.fields["due_date"].value = "2026-01-01"
        report = validate_document(doc)
        self.assertIn("due_date_before_invoice", codes(report))

    def test_line_item_arithmetic(self) -> None:
        doc = build_invoice()
        doc.line_items[1].amount = 123.0
        report = validate_document(doc)
        self.assertIn("line_item_arithmetic", codes(report))

        # The tolerance is 0.01 + ₹0.005 per unit, because the rate is only
        # ever printed to two decimals: 3 × 33.33 = 99.99 against a ₹100.00
        # line is rounding, not a broken total …
        rounded = Document(doc_type="invoice")
        rounded.set_field("subtotal", 100.0, "currency_amount", 0.9)
        rounded.line_items.append(LineItem("rounding", 3, 33.33, 100.0))
        self.assertNotIn("line_item_arithmetic", codes(validate_document(rounded)))

        # … while 50 paise off a single unit is well past that slack.
        sloppy = Document(doc_type="invoice")
        sloppy.set_field("subtotal", 100.5, "currency_amount", 0.9)
        sloppy.line_items.append(LineItem("sloppy", 1, 100.0, 100.5))
        self.assertIn("line_item_arithmetic", codes(validate_document(sloppy)))

    def test_document_with_no_fields_warns_but_passes(self) -> None:
        report = validate_document(Document(doc_type="generic"))
        self.assertIn("no_fields_extracted", codes(report))
        self.assertTrue(report.passed)


class TestConfidenceScoring(unittest.TestCase):
    def test_confidence_rises_on_pass_and_falls_on_error(self) -> None:
        doc = build_invoice()
        validate_document(doc)
        self.assertGreater(doc.fields["customer_phone"].confidence, 0.9)
        self.assertLessEqual(doc.fields["customer_phone"].confidence, 0.99)

        broken = build_invoice(customer_phone="12345")
        validate_document(broken)
        self.assertLess(broken.fields["customer_phone"].confidence, 0.6)

    def test_confidence_never_leaves_bounds(self) -> None:
        doc = build_invoice(
            customer_phone="abc",
            customer_email="nope",
            invoice_date="not-a-date",
            customer_address="x",
        )
        validate_document(doc)
        for field in doc.fields.values():
            self.assertGreaterEqual(field.confidence, 0.0)
            self.assertLessEqual(field.confidence, 1.0)

    def test_report_confidence_summary(self) -> None:
        report = validate_document(build_invoice())
        self.assertGreater(report.mean_confidence, 0.5)
        self.assertLessEqual(report.mean_confidence, 0.99)
        self.assertEqual(report.min_confidence, min(report.confidences.values()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
