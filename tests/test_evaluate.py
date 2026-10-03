"""Tests for the scorer: canonicalisation, P/R/F1, exact match, report files."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from docextract.evaluate import (
    canonical,
    evaluate_run,
    format_table,
    load_gold,
    score_document,
    write_markdown_report,
    write_metrics_csv,
)
from docextract.schema import Document, LineItem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"


def predicted_invoice() -> Document:
    doc = Document(doc_type="invoice", provider="rules")
    for name, value, field_type in [
        ("invoice_no", "VA-2026-0417", "doc_id"),
        ("invoice_date", "2026-02-12", "date"),
        ("customer_name", "Meera Krishnan", "person_name"),
        ("total", 103368.0, "currency_amount"),
    ]:
        doc.set_field(name, value, field_type, 0.9)
    doc.line_items.append(LineItem("Document digitisation services", 40, 1250.0, 50000.0))
    return doc


def gold_invoice() -> dict:
    return {
        "doc_type": "invoice",
        "fields": {
            "invoice_no": "VA-2026-0417",
            "invoice_date": "2026-02-12",
            "customer_name": "Meera Krishnan",
            "total": 103368.0,
        },
        "line_items": [
            {
                "description": "Document digitisation services",
                "quantity": 40.0,
                "unit_price": 1250.0,
                "amount": 50000.0,
            }
        ],
    }


class TestCanonical(unittest.TestCase):
    def test_numbers_compare_as_two_decimals(self) -> None:
        self.assertEqual(canonical(103368.0), canonical(103368))

    def test_money_strings_and_numbers_agree(self) -> None:
        self.assertEqual(canonical("₹1,03,368.00", "currency_amount"), canonical(103368.0))
        self.assertEqual(canonical("Rs. 2,500", "currency_amount"), canonical(2500.0))

    def test_date_strings_agree(self) -> None:
        self.assertEqual(canonical("12/02/2026", "date"), canonical("2026-02-12"))
        self.assertEqual(canonical("14 फ़रवरी 2026", "date"), "2026-02-14")

    def test_phone_strings_agree(self) -> None:
        self.assertEqual(canonical("+91 98450 12345", "phone"), canonical("9845012345"))

    def test_email_is_case_insensitive(self) -> None:
        self.assertEqual(canonical("Meera.K@Example.COM", "email"), canonical("meera.k@example.com"))

    def test_whitespace_is_collapsed(self) -> None:
        self.assertEqual(canonical("  Meera   Krishnan "), "Meera Krishnan")

    def test_none_is_empty(self) -> None:
        self.assertEqual(canonical(None), "")


class TestScoreDocument(unittest.TestCase):
    def test_perfect_match(self) -> None:
        score = score_document(predicted_invoice(), gold_invoice(), sample="inv")
        self.assertEqual(score.matched_fields, 4)
        self.assertEqual(score.gold_fields, 4)
        self.assertEqual(score.pred_fields, 4)
        self.assertEqual(score.field_precision, 1.0)
        self.assertEqual(score.field_recall, 1.0)
        self.assertEqual(score.field_f1, 1.0)
        self.assertEqual(score.exact_match, 1)
        self.assertEqual(score.matched_line_items, 1)
        self.assertEqual(score.gold_line_items, 1)

    def test_missing_field_lowers_recall_only(self) -> None:
        gold = gold_invoice()
        gold["fields"]["gstin"] = "29AABCV1234K1Z5"
        score = score_document(predicted_invoice(), gold, sample="inv")
        self.assertEqual(score.field_precision, 1.0)
        self.assertAlmostEqual(score.field_recall, 4 / 5)
        self.assertLess(score.field_f1, 1.0)
        self.assertEqual(score.exact_match, 0)
        self.assertEqual(score.missing, ["gstin"])

    def test_extra_field_lowers_precision_only(self) -> None:
        doc = predicted_invoice()
        doc.set_field("gstin", "29AABCV1234K1Z5", "id_number", 0.9)
        score = score_document(doc, gold_invoice(), sample="inv")
        self.assertEqual(score.field_recall, 1.0)
        self.assertAlmostEqual(score.field_precision, 4 / 5)
        self.assertEqual(score.exact_match, 0)
        self.assertEqual(score.extra, ["gstin"])

    def test_mismatched_value_counts_as_both_error_and_miss(self) -> None:
        doc = predicted_invoice()
        doc.fields["total"].value = 99999.0
        score = score_document(doc, gold_invoice(), sample="inv")
        self.assertEqual(score.mismatched, ["total"])
        self.assertIn("total", score.missing)
        self.assertIn("total", score.extra)
        self.assertLess(score.field_precision, 1.0)
        self.assertLess(score.field_recall, 1.0)
        self.assertEqual(score.exact_match, 0)

    def test_wrong_doc_type_breaks_exact_match(self) -> None:
        doc = predicted_invoice()
        doc.doc_type = "receipt"
        score = score_document(doc, gold_invoice(), sample="inv")
        self.assertEqual(score.field_f1, 1.0)
        self.assertFalse(score.doc_type_ok)
        self.assertEqual(score.exact_match, 0)

    def test_line_item_mismatch(self) -> None:
        doc = predicted_invoice()
        doc.line_items[0].amount = 49999.0
        score = score_document(doc, gold_invoice(), sample="inv")
        self.assertEqual(score.matched_line_items, 0)
        self.assertEqual(score.exact_match, 0)
        self.assertEqual(score.field_f1, 1.0)  # fields untouched

    def test_empty_gold_and_empty_prediction_are_perfect(self) -> None:
        score = score_document(Document(doc_type="generic"), {"doc_type": "generic"}, sample="x")
        self.assertEqual(score.field_f1, 1.0)
        self.assertEqual(score.exact_match, 1)

    def test_min_confidence_filters_predictions(self) -> None:
        doc = predicted_invoice()
        doc.set_field("gstin", "29AABCV1234K1Z5", "id_number", 0.2)
        strict = score_document(doc, gold_invoice(), sample="inv", min_confidence=0.5)
        self.assertEqual(strict.pred_fields, 4)
        self.assertEqual(strict.extra, [])

        loose = score_document(doc, gold_invoice(), sample="inv", min_confidence=0.0)
        self.assertEqual(loose.pred_fields, 5)
        self.assertEqual(loose.extra, ["gstin"])

    def test_gold_formatting_is_tolerated(self) -> None:
        gold = gold_invoice()
        gold["fields"]["invoice_date"] = "12/02/2026"
        gold["fields"]["total"] = "₹1,03,368.00"
        score = score_document(predicted_invoice(), gold, sample="inv")
        self.assertEqual(score.field_f1, 1.0)

        # gold may also omit optional line-item keys; the scorer applies the
        # same LineItem defaults (quantity 1.0, amount 0.0) on both sides,
        # so the item still matches instead of splitting on a missing column
        doc = Document(doc_type="invoice", provider="rules")
        doc.line_items.append(LineItem("Consulting", 1, 1200.0, 1200.0))
        sparse_gold = {
            "doc_type": "invoice",
            "fields": {},
            "line_items": [{"description": "Consulting", "unit_price": 1200.0, "amount": 1200.0}],
        }
        sparse = score_document(doc, sparse_gold, sample="inv")
        self.assertEqual(sparse.matched_line_items, 1)
        self.assertEqual(sparse.pred_line_items, 1)
        self.assertEqual(sparse.gold_line_items, 1)
        self.assertEqual(sparse.exact_match, 1)


class TestLoadGold(unittest.TestCase):
    def test_loads_sample_gold_files(self) -> None:
        for path in sorted(GOLD_DIR.glob("*.json")):
            payload = load_gold(path)
            self.assertIn("fields", payload, msg=path.name)
            self.assertIn("doc_type", payload, msg=path.name)
            self.assertIsInstance(payload["fields"], dict)

    def test_rejects_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_gold(path)


class TestEvaluateRun(unittest.TestCase):
    def test_real_corpus_scores_perfectly(self) -> None:
        """Regression net: the shipped rules must keep hitting the shipped gold."""
        report = evaluate_run(SAMPLES_DIR, GOLD_DIR, provider="rules", write_outputs=False)
        self.assertEqual(report.count, 3)
        self.assertEqual(report.skipped, [])
        self.assertEqual(report.exact_matches, 3)
        self.assertEqual(report.macro_field_precision, 1.0)
        self.assertEqual(report.macro_field_recall, 1.0)
        self.assertEqual(report.macro_field_f1, 1.0)
        self.assertEqual(report.micro_field_f1, 1.0)
        self.assertEqual({s.sample for s in report.scores}, {"invoice_en", "form_hi", "fir_en_hinglish"})

    def test_writes_metrics_csv_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_run(SAMPLES_DIR, GOLD_DIR, provider="rules", out_dir=tmp)
            metrics = Path(tmp) / "metrics.csv"
            markdown = Path(tmp) / "report.md"
            self.assertTrue(metrics.is_file())
            self.assertTrue(markdown.is_file())

            with metrics.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            samples = [row["sample"] for row in rows]
            self.assertIn("invoice_en", samples)
            self.assertIn("MACRO-AVERAGE", samples)
            invoice_row = next(r for r in rows if r["sample"] == "invoice_en")
            self.assertEqual(invoice_row["field_f1"], "1.0000")
            self.assertEqual(invoice_row["exact_match"], "1")

            text = markdown.read_text(encoding="utf-8")
            self.assertIn("# Evaluation report", text)
            self.assertIn("MACRO-AVERAGE", text)
            self.assertIn("Provider: `rules`", text)
            self.assertIn("Metric definitions", text)

    def test_skips_samples_without_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples"
            gold = Path(tmp) / "gold"
            samples.mkdir()
            gold.mkdir()
            (samples / "orphan.txt").write_text("Name: Nobody\n", encoding="utf-8")
            (samples / "known.txt").write_text("Name: Somebody\n", encoding="utf-8")
            (gold / "known.json").write_text(
                json.dumps({"doc_type": "generic", "fields": {"name": "Somebody"}}),
                encoding="utf-8",
            )
            report = evaluate_run(samples, gold, provider="rules", write_outputs=False)
            self.assertEqual(report.count, 1)
            self.assertEqual(report.skipped, ["orphan.txt"])
            self.assertEqual(report.scores[0].exact_match, 1)


class TestReportFormatting(unittest.TestCase):
    def test_table_has_expected_columns(self) -> None:
        report = evaluate_run(SAMPLES_DIR, GOLD_DIR, provider="rules", write_outputs=False)
        table = format_table(report)
        header = table.splitlines()[0]
        for column in ("sample", "gold", "pred", "match", "P", "R", "F1", "exact"):
            self.assertIn(column, header)
        self.assertIn("MACRO-AVERAGE", table)
        self.assertEqual(len(table.splitlines()), 1 + 1 + report.count + 1)

    def test_metrics_csv_columns_are_stable(self) -> None:
        from docextract.evaluate import CSV_COLUMNS

        report = evaluate_run(SAMPLES_DIR, GOLD_DIR, provider="rules", write_outputs=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_metrics_csv(report, Path(tmp) / "metrics.csv")
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
            self.assertEqual(tuple(header), CSV_COLUMNS)

    def test_markdown_report_helper(self) -> None:
        report = evaluate_run(SAMPLES_DIR, GOLD_DIR, provider="rules", write_outputs=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_markdown_report(report, Path(tmp) / "report.md")
            text = path.read_text(encoding="utf-8")
            self.assertIn("exact", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
