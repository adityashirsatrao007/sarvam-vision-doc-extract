"""Tests for the offline rule extractor and the normalisers it depends on."""

from __future__ import annotations

import unittest
from pathlib import Path

from docextract.extract_rules import (
    KNOWN_TITLES,
    LABEL_FIELDS,
    detect_doc_type,
    extract_document,
    parse_line_items,
)
from docextract.normalize import (
    ascii_digits,
    collapse_ws,
    format_inr,
    nfkc,
    normalize_amount,
    normalize_date,
    normalize_email,
    normalize_person_name,
    normalize_phone,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "samples"


def sample(name: str) -> str:
    return (SAMPLES / f"{name}.txt").read_text(encoding="utf-8")


class TestNormalizeAmount(unittest.TestCase):
    def test_western_grouping(self) -> None:
        self.assertEqual(normalize_amount("₹1,234.50"), 1234.50)

    def test_indian_lakh_grouping(self) -> None:
        self.assertEqual(normalize_amount("₹1,03,368.00"), 103368.00)

    def test_rupee_symbol_and_rs_prefix(self) -> None:
        self.assertEqual(normalize_amount("Rs. 2,500.00"), 2500.00)
        self.assertEqual(normalize_amount("INR 2,500"), 2500.0)

    def test_hindi_currency_word(self) -> None:
        self.assertEqual(normalize_amount("रु. 2,500.00"), 2500.00)

    def test_multiplier_words(self) -> None:
        self.assertEqual(normalize_amount("1.5 लाख"), 150000.0)
        self.assertEqual(normalize_amount("2 crore"), 20000000.0)
        self.assertEqual(normalize_amount("3 हज़ार"), 3000.0)

    def test_devanagari_digits(self) -> None:
        self.assertEqual(normalize_amount("₹१,२३४.५०"), 1234.50)

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(normalize_amount("abc"))
        self.assertIsNone(normalize_amount(""))
        self.assertIsNone(normalize_amount(None))

    def test_plain_numbers_pass_through(self) -> None:
        self.assertEqual(normalize_amount(103368.0), 103368.0)
        self.assertEqual(normalize_amount(45000), 45000.0)


class TestNormalizeDate(unittest.TestCase):
    def test_slash_and_dash_day_first(self) -> None:
        self.assertEqual(normalize_date("12/02/2026"), "2026-02-12")
        self.assertEqual(normalize_date("26-02-2026"), "2026-02-26")
        self.assertEqual(normalize_date("07.01.2026"), "2026-01-07")

    def test_iso_input(self) -> None:
        self.assertEqual(normalize_date("2026-03-03"), "2026-03-03")

    def test_hindi_month_names(self) -> None:
        self.assertEqual(normalize_date("14 फ़रवरी 2026"), "2026-02-14")
        self.assertEqual(normalize_date("14 फरवरी 2026"), "2026-02-14")  # no nukta
        self.assertEqual(normalize_date("1 जनवरी 2026"), "2026-01-01")
        self.assertEqual(normalize_date("15 सितंबर 2026"), "2026-09-15")
        self.assertEqual(normalize_date("21 दिसम्बर 2026"), "2026-12-21")

    def test_english_month_names(self) -> None:
        self.assertEqual(normalize_date("14 February 2026"), "2026-02-14")
        self.assertEqual(normalize_date("February 14, 2026"), "2026-02-14")

    def test_invalid_dates_return_none(self) -> None:
        self.assertIsNone(normalize_date("32/01/2026"))
        self.assertIsNone(normalize_date("30/02/2026"))
        self.assertIsNone(normalize_date("not a date"))
        self.assertIsNone(normalize_date(None))


class TestNormaliseMisc(unittest.TestCase):
    def test_phone_variants(self) -> None:
        self.assertEqual(normalize_phone("+91 98450 12345"), "9845012345")
        self.assertEqual(normalize_phone("09876543210"), "9876543210")
        self.assertEqual(normalize_phone("919876543210"), "9876543210")
        self.assertEqual(normalize_phone("९८२२०१४५६७"), "9822014567")
        self.assertIsNone(normalize_phone("12345"))
        self.assertIsNone(normalize_phone("not a phone"))

    def test_email_is_lowercased(self) -> None:
        self.assertEqual(normalize_email("  Meera.K@Example.COM "), "meera.k@example.com")
        self.assertIsNone(normalize_email("meera.k at example.com"))

    def test_person_name_cleanup(self) -> None:
        self.assertEqual(normalize_person_name("  Meera Krishnan. "), "Meera Krishnan")
        self.assertEqual(normalize_person_name("Rakesh Yadav @ Rocky"), "Rakesh Yadav @ Rocky")

    def test_nfkc_and_whitespace(self) -> None:
        self.assertEqual(nfkc("Ａ"), "A")
        self.assertEqual(collapse_ws("  a \t b \n c  "), "a b c")

    def test_devanagari_digits(self) -> None:
        self.assertEqual(ascii_digits("१२३४५"), "12345")

    def test_indian_number_formatting(self) -> None:
        self.assertEqual(format_inr(103368), "₹1,03,368.00")
        self.assertEqual(format_inr(87600), "₹87,600.00")
        self.assertEqual(format_inr(1234.5), "₹1,234.50")


class TestLabelVocabulary(unittest.TestCase):
    def test_english_labels_registered(self) -> None:
        for label in ("invoice no", "date", "total", "name", "address", "email", "phone"):
            self.assertIn(label, LABEL_FIELDS)

    def test_hindi_labels_registered(self) -> None:
        for label in ("नाम", "दिनांक", "राशि", "मोबाइल नंबर", "पता", "आवेदन संख्या", "पिता का नाम"):
            self.assertIn(label, LABEL_FIELDS)

    def test_known_titles(self) -> None:
        self.assertIn("invoice", KNOWN_TITLES)
        self.assertIn("fir", KNOWN_TITLES)
        self.assertIn("आवेदन पत्र", KNOWN_TITLES)


class TestInvoiceExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = extract_document(sample("invoice_en"), source="invoice_en.txt")

    def test_doc_type(self) -> None:
        self.assertEqual(self.doc.doc_type, "invoice")

    def test_header_fields(self) -> None:
        self.assertEqual(self.doc.value("org_name"), "VEDA ANALYTICS PRIVATE LIMITED")
        self.assertEqual(self.doc.value("title"), "INVOICE")
        self.assertEqual(self.doc.value("gstin"), "29AABCV1234K1Z5")
        self.assertEqual(self.doc.value("vendor_address"),
                         "42, 5th Cross, Indiranagar, Bengaluru 560038, Karnataka")

    def test_dates_are_iso(self) -> None:
        self.assertEqual(self.doc.value("invoice_date"), "2026-02-12")
        self.assertEqual(self.doc.value("due_date"), "2026-02-26")

    def test_money_is_numeric(self) -> None:
        self.assertEqual(self.doc.value("subtotal"), 87600.0)
        self.assertEqual(self.doc.value("cgst"), 7884.0)
        self.assertEqual(self.doc.value("sgst"), 7884.0)
        self.assertEqual(self.doc.value("total"), 103368.0)

    def test_contacts(self) -> None:
        self.assertEqual(self.doc.value("vendor_phone"), "9845012345")  # +91 stripped
        self.assertEqual(self.doc.value("customer_phone"), "9876543210")
        self.assertEqual(self.doc.value("customer_email"), "meera.k@example.com")

    def test_line_items(self) -> None:
        self.assertEqual(len(self.doc.line_items), 3)
        first = self.doc.line_items[0]
        self.assertEqual(first.description, "Document digitisation services")
        self.assertEqual(first.quantity, 40.0)
        self.assertEqual(first.unit_price, 1250.0)
        self.assertEqual(first.amount, 50000.0)
        for item in self.doc.line_items:
            self.assertTrue(item.arithmetic_ok, msg=item.to_dict())

    def test_amount_in_words_line_is_not_a_field(self) -> None:
        # "Amount in words: ..." must not be swallowed by the "amount" label
        self.assertNotIn("amount in words", self.doc.fields)

    def test_confidences_are_present(self) -> None:
        for field in self.doc.fields.values():
            self.assertGreater(field.confidence, 0.0)
            self.assertLessEqual(field.confidence, 1.0)


class TestHindiFormExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = extract_document(sample("form_hi"), source="form_hi.txt")

    def test_doc_type_and_title(self) -> None:
        self.assertEqual(self.doc.doc_type, "form")
        self.assertEqual(self.doc.value("title"), "आवेदन पत्र")
        self.assertEqual(self.doc.value("org_name"), "जिला कार्यालय, पुणे")

    def test_hindi_labels(self) -> None:
        self.assertEqual(self.doc.value("application_no"), "MH/PUN/2026/004512")
        self.assertEqual(self.doc.value("date"), "2026-03-03")
        self.assertEqual(self.doc.value("name"), "सुनीता देशपांडे")
        self.assertEqual(self.doc.value("father_name"), "रमेश देशपांडे")
        self.assertEqual(self.doc.value("phone"), "9822014567")
        self.assertEqual(self.doc.value("amount"), 2500.0)

    def test_hindi_month_name_date(self) -> None:
        self.assertEqual(self.doc.value("dob"), "2026-02-14")

    def test_address_kept_intact(self) -> None:
        self.assertEqual(
            self.doc.value("address"),
            "कमरा नं. 12, शिवनगर, कोथरुड, पुणे, महाराष्ट्र 411038",
        )

    def test_unlabelled_lines_are_ignored(self) -> None:
        # the amount-in-words line and the signature line are not fields
        self.assertNotIn("शब्दों में", self.doc.fields)
        self.assertNotIn("हस्ताक्षर", self.doc.fields)
        self.assertEqual(self.doc.line_items, [])


class TestFirExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = extract_document(sample("fir_en_hinglish"), source="fir_en_hinglish.txt")

    def test_doc_type(self) -> None:
        self.assertEqual(self.doc.doc_type, "fir")
        self.assertEqual(self.doc.value("title"), "FIR")

    def test_fir_fields(self) -> None:
        self.assertEqual(self.doc.value("fir_no"), "142/2026")
        self.assertEqual(self.doc.value("date"), "2026-01-07")
        self.assertEqual(self.doc.value("occurrence_date"), "2026-01-05")
        self.assertEqual(self.doc.value("police_station"), "Cyber Crime Police Station, Kolkata")
        self.assertEqual(self.doc.value("district"), "Kolkata, West Bengal")
        self.assertEqual(self.doc.value("amount"), 45000.0)

    def test_complainant_and_accused(self) -> None:
        self.assertEqual(self.doc.value("complainant_name"), "Arjun Mehta")
        self.assertEqual(self.doc.value("complainant_phone"), "9830011224")
        self.assertEqual(self.doc.value("complainant_email"), "arjun.mehta@example.com")
        self.assertEqual(self.doc.value("accused_name"), "Rakesh Yadav @ Rocky")

    def test_hinglish_narrative_lines_are_not_fields(self) -> None:
        self.assertNotIn("details", self.doc.fields)
        self.assertNotIn("signature", self.doc.fields)
        # org_name must not be invented from the free-text narrative
        self.assertIsNone(self.doc.get("org_name"))


class TestParserBehaviour(unittest.TestCase):
    def test_comments_are_ignored(self) -> None:
        doc = extract_document("# just a comment\nName: Alpha\n")
        self.assertEqual(doc.value("name"), "Alpha")
        for field in doc.fields.values():
            self.assertNotIn("#", str(field.value))

    def test_first_label_wins(self) -> None:
        doc = extract_document("Name: First Person\nName: Second Person\n")
        self.assertEqual(doc.value("name"), "First Person")

    def test_organisation_line_only_before_first_label(self) -> None:
        doc = extract_document("ACME PRIVATE LIMITED\nName: Beta\nNarrative text here\n")
        self.assertEqual(doc.value("org_name"), "ACME PRIVATE LIMITED")

    def test_colonless_narrative_is_not_a_label(self) -> None:
        doc = extract_document("Name: Gamma\nthis line has no separator at all\n")
        self.assertEqual(list(doc.fields), ["name"])

    def test_tab_or_double_space_separator(self) -> None:
        doc = extract_document("Invoice No\tINV-9\nName  Delta\n")
        self.assertEqual(doc.value("invoice_no"), "INV-9")
        self.assertEqual(doc.value("name"), "Delta")

    def test_unknown_label_is_skipped(self) -> None:
        doc = extract_document("Shoe Size: 42\nName: Epsilon\n")
        self.assertEqual(list(doc.fields), ["name"])

    def test_tax_label_with_qualifier(self) -> None:
        doc = extract_document("CGST @ 18%: ₹900.00\nName: Zeta\n")
        self.assertEqual(doc.value("cgst"), 900.0)

    def test_unparseable_value_keeps_raw_and_lowers_confidence(self) -> None:
        doc = extract_document("Date: not-a-date\nName: Eta\n")
        self.assertEqual(doc.value("date"), "not-a-date")
        self.assertLess(doc.fields["date"].confidence, 0.6)

    def test_generic_doc_type(self) -> None:
        doc = extract_document("Name: Theta\nAddress: 12 MG Road, Pune 411001\n")
        self.assertEqual(doc.doc_type, "generic")

    def test_detect_doc_type_helper(self) -> None:
        doc = extract_document("Invoice No: A-1\n")
        self.assertEqual(detect_doc_type(doc), "invoice")


class TestLineItemParser(unittest.TestCase):
    def test_parses_table(self) -> None:
        items = parse_line_items(
            [
                "S.No | Description | Qty | Rate | Amount",
                "1 | Widget | 2 | 50.00 | 100.00",
                "2 | Gadget | 3 | 25.50 | 76.50",
            ]
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].amount, 100.0)
        self.assertEqual(items[1].quantity, 3.0)
        self.assertEqual(items[1].unit_price, 25.50)

    def test_no_table_returns_empty(self) -> None:
        self.assertEqual(parse_line_items(["Name: Alpha", "Address: 12 MG Road, Pune"]), [])
        self.assertEqual(parse_line_items([]), [])

    def test_table_stops_at_first_non_table_line(self) -> None:
        items = parse_line_items(
            [
                "Item | Qty | Rate | Amount",
                "1 | 1 | 10.00 | 10.00",
                "Subtotal: ₹10.00",
                "2 | 1 | 99.00 | 99.00",
            ]
        )
        self.assertEqual(len(items), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
