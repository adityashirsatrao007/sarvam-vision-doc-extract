"""End-to-end CLI tests: real subprocesses writing into a temp directory."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "data" / "samples"


def run_cli(*args: str, env_overrides: dict[str, str | None] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [sys.executable, "-m", "docextract", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


class TestExtractCommand(unittest.TestCase):
    def test_invoice_extract_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                "extract",
                "--input", "data/samples/invoice_en.txt",
                "--provider", "rules",
                "--out", tmp,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["doc_type"], "invoice")
            self.assertEqual(payload["provider"], "rules")
            self.assertEqual(payload["fields"]["total"]["value"], 103368.0)
            self.assertEqual(payload["fields"]["total"]["type"], "currency_amount")
            self.assertIn("confidence", payload["fields"]["invoice_no"])
            self.assertEqual(len(payload["line_items"]), 3)
            self.assertTrue(payload["validation"]["passed"])
            self.assertEqual(payload["validation"]["issues"], [])

            written = Path(tmp) / "invoice_en.extracted.json"
            self.assertTrue(written.is_file())
            self.assertEqual(json.loads(written.read_text(encoding="utf-8")), payload)
            self.assertIn("wrote", result.stderr)

    def test_hindi_form_extract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                "extract", "--input", "data/samples/form_hi.txt", "--provider", "rules", "--out", tmp
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["doc_type"], "form")
            self.assertEqual(payload["fields"]["name"]["value"], "सुनीता देशपांडे")
            self.assertEqual(payload["fields"]["dob"]["value"], "2026-02-14")
            self.assertTrue((Path(tmp) / "form_hi.extracted.json").is_file())

    def test_fir_extract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                "extract", "--input", "data/samples/fir_en_hinglish.txt", "--provider", "rules", "--out", tmp
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["doc_type"], "fir")
            self.assertEqual(payload["fields"]["fir_no"]["value"], "142/2026")
            self.assertTrue(payload["validation"]["passed"])

    def test_quiet_flag_suppresses_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                "extract", "-i", "data/samples/invoice_en.txt", "-o", tmp, "--quiet"
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(result.stdout.strip(), "")
            self.assertTrue((Path(tmp) / "invoice_en.extracted.json").is_file())

    def test_creates_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "out"
            result = run_cli(
                "extract", "-i", "data/samples/invoice_en.txt", "-o", str(target), "-q"
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((target / "invoice_en.extracted.json").is_file())


class TestEvaluateCommand(unittest.TestCase):
    def test_evaluate_prints_table_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                "evaluate",
                "--samples", "data/samples",
                "--gold", "data/gold",
                "--provider", "rules",
                "--out", tmp,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("MACRO-AVERAGE", result.stdout)
            self.assertIn("invoice_en", result.stdout)
            self.assertIn("form_hi", result.stdout)
            self.assertIn("fir_en_hinglish", result.stdout)
            self.assertIn("macro F1=1.000", result.stdout)
            self.assertIn("exact=3/3", result.stdout)

            metrics = Path(tmp) / "metrics.csv"
            report = Path(tmp) / "report.md"
            self.assertTrue(metrics.is_file())
            self.assertTrue(report.is_file())
            with metrics.open(encoding="utf-8", newline="") as handle:
                rows = {row["sample"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["invoice_en"]["field_f1"], "1.0000")
            self.assertEqual(rows["invoice_en"]["exact_match"], "1")
            self.assertEqual(rows["MACRO-AVERAGE"]["field_f1"], "1.0000")
            self.assertIn("Evaluation report", report.read_text(encoding="utf-8"))

    def test_missing_samples_directory(self) -> None:
        result = run_cli("evaluate", "--samples", "does/not/exist", "--gold", "data/gold")
        self.assertEqual(result.returncode, 1)
        self.assertIn("samples directory not found", result.stderr)

    def test_directory_without_gold_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                "evaluate", "--samples", tmp, "--gold", "data/gold", "--out", tmp
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("no sample/gold pairs", result.stderr)


class TestProvidersCommand(unittest.TestCase):
    def test_lists_all_backends(self) -> None:
        result = run_cli("providers")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("rules", result.stdout)
        self.assertIn("ocr", result.stdout)
        self.assertIn("llm", result.stdout)
        self.assertIn("default", result.stdout)
        self.assertIn("optional", result.stdout)


class TestErrorPaths(unittest.TestCase):
    def test_missing_input_file(self) -> None:
        result = run_cli("extract", "--input", "data/samples/nope.txt")
        self.assertEqual(result.returncode, 1)
        self.assertIn("input file not found", result.stderr)

        # A file that exists but is not UTF-8 must come back as a clean
        # error line (the provider reads the input), never a traceback.
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "scan.txt"
            binary.write_bytes(b"\xff\xfe\x00\x01not utf-8")
            unreadable = run_cli("extract", "-i", str(binary), "-o", tmp)
            self.assertEqual(unreadable.returncode, 1)
            self.assertIn("not UTF-8", unreadable.stderr)
            self.assertNotIn("Traceback", unreadable.stderr)

    def test_unknown_provider(self) -> None:
        result = run_cli("extract", "-i", "data/samples/invoice_en.txt", "-p", "magic")
        self.assertEqual(result.returncode, 1)
        self.assertIn("unknown provider", result.stderr)

    def test_llm_provider_without_key_is_actionable(self) -> None:
        result = run_cli(
            "extract",
            "-i", "data/samples/invoice_en.txt",
            "-p", "llm",
            env_overrides={"SARVAM_API_KEY": None, "DOCEXTRACT_ENV_FILE": "/nonexistent/docextract.env"},
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("SARVAM_API_KEY", result.stderr)
        self.assertIn("--provider rules", result.stderr)

    def test_ocr_provider_without_dependencies_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "scan.png"
            image.write_bytes(b"not really a png")
            result = run_cli(
                "extract",
                "-i", str(image),
                "-p", "ocr",
                "-o", tmp,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("pip install", result.stderr)

    def test_ocr_provider_reuses_sibling_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "scan.txt"
            text_path.write_text("Invoice No: SIB-1\nTotal: ₹10.00\n", encoding="utf-8")
            result = run_cli("extract", "-i", str(text_path), "-p", "ocr", "-o", tmp)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["provider"], "ocr")
            self.assertEqual(payload["fields"]["invoice_no"]["value"], "SIB-1")
            self.assertEqual(payload["fields"]["total"]["value"], 10.0)

    def test_version_flag(self) -> None:
        result = run_cli("--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn("docextract", result.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
