"""Score predicted extractions against ``data/gold`` JSON.

Field-level precision / recall / F1 plus an exact-match flag per document,
then macro-averaged across the corpus. Writes ``results/metrics.csv`` and
``results/report.md``.

Comparison is done on *canonical* values, so ``₹1,03,368.00``, ``103368.0``
and ``103368`` all score as the same amount, and ``12/02/2026`` and
``2026-02-12`` as the same date.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .normalize import collapse_ws, nfkc, normalize_amount, normalize_date, normalize_email, normalize_phone
from .providers import get_provider
from .schema import Document, LineItem
from .validate import validate_document

__all__ = [
    "DocScore",
    "EvalReport",
    "canonical",
    "score_document",
    "evaluate_run",
    "format_table",
]

CSV_COLUMNS: tuple[str, ...] = (
    "sample",
    "provider",
    "doc_type_gold",
    "doc_type_pred",
    "doc_type_ok",
    "gold_fields",
    "pred_fields",
    "matched_fields",
    "field_precision",
    "field_recall",
    "field_f1",
    "gold_line_items",
    "pred_line_items",
    "matched_line_items",
    "exact_match",
    "missing",
    "extra",
    "mismatched",
)


def canonical(value: Any, field_type: str | None = None) -> str:
    """Reduce a value to a comparable string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"

    text = collapse_ws(nfkc(str(value)))
    if field_type == "date":
        return normalize_date(text) or text
    if field_type == "currency_amount":
        amount = normalize_amount(text)
        return f"{amount:.2f}" if amount is not None else text
    if field_type == "phone":
        return normalize_phone(text) or re.sub(r"\D", "", text)
    if field_type == "email":
        return normalize_email(text) or text.lower()
    return text


def _number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return collapse_ws(str(value))


def item_key(item: Mapping[str, Any] | LineItem) -> tuple[str, str, str, str]:
    payload = item.to_dict() if isinstance(item, LineItem) else item
    return (
        collapse_ws(str(payload.get("description", ""))).casefold(),
        _number(payload.get("quantity")),
        _number(payload.get("unit_price")),
        _number(payload.get("amount")),
    )


def _ratio(numerator: int, denominator: int, *, empty_is_one: bool) -> float:
    if denominator == 0:
        return 1.0 if empty_is_one else 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class DocScore:
    """Scoring result for a single sample document."""

    sample: str
    provider: str
    gold_doc_type: str = ""
    pred_doc_type: str = ""
    gold_fields: int = 0
    pred_fields: int = 0
    matched_fields: int = 0
    field_precision: float = 0.0
    field_recall: float = 0.0
    field_f1: float = 0.0
    gold_line_items: int = 0
    pred_line_items: int = 0
    matched_line_items: int = 0
    item_precision: float = 0.0
    item_recall: float = 0.0
    item_f1: float = 0.0
    doc_type_ok: bool = False
    exact_match: int = 0
    missing: list[str] = dc_field(default_factory=list)
    extra: list[str] = dc_field(default_factory=list)
    mismatched: list[str] = dc_field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "provider": self.provider,
            "doc_type_gold": self.gold_doc_type,
            "doc_type_pred": self.pred_doc_type,
            "doc_type_ok": int(self.doc_type_ok),
            "gold_fields": self.gold_fields,
            "pred_fields": self.pred_fields,
            "matched_fields": self.matched_fields,
            "field_precision": f"{self.field_precision:.4f}",
            "field_recall": f"{self.field_recall:.4f}",
            "field_f1": f"{self.field_f1:.4f}",
            "gold_line_items": self.gold_line_items,
            "pred_line_items": self.pred_line_items,
            "matched_line_items": self.matched_line_items,
            "exact_match": self.exact_match,
            "missing": ";".join(self.missing),
            "extra": ";".join(self.extra),
            "mismatched": ";".join(self.mismatched),
        }


@dataclass
class EvalReport:
    """Aggregate scoring result for a whole run."""

    provider: str
    samples_dir: str
    gold_dir: str
    scores: list[DocScore] = dc_field(default_factory=list)
    skipped: list[str] = dc_field(default_factory=list)
    min_confidence: float = 0.0

    @property
    def count(self) -> int:
        return len(self.scores)

    @property
    def macro_field_precision(self) -> float:
        return sum(s.field_precision for s in self.scores) / self.count if self.count else 0.0

    @property
    def macro_field_recall(self) -> float:
        return sum(s.field_recall for s in self.scores) / self.count if self.count else 0.0

    @property
    def macro_field_f1(self) -> float:
        return sum(s.field_f1 for s in self.scores) / self.count if self.count else 0.0

    @property
    def exact_matches(self) -> int:
        return sum(s.exact_match for s in self.scores)

    @property
    def micro_field_precision(self) -> float:
        predicted = sum(s.pred_fields for s in self.scores)
        matched = sum(s.matched_fields for s in self.scores)
        if predicted == 0:
            return 1.0 if sum(s.gold_fields for s in self.scores) == 0 else 0.0
        return matched / predicted

    @property
    def micro_field_recall(self) -> float:
        gold = sum(s.gold_fields for s in self.scores)
        if gold == 0:
            return 1.0
        return sum(s.matched_fields for s in self.scores) / gold

    @property
    def micro_field_f1(self) -> float:
        return _f1(self.micro_field_precision, self.micro_field_recall)


def score_document(
    predicted: Document,
    gold: Mapping[str, Any],
    *,
    sample: str = "sample",
    min_confidence: float = 0.0,
) -> DocScore:
    """Compare one predicted document against its gold JSON."""
    gold_fields: Mapping[str, Any] = gold.get("fields") or {}
    gold_items = list(gold.get("line_items") or [])

    predicted_fields = {
        name: field
        for name, field in predicted.fields.items()
        if field.confidence >= min_confidence
    }

    gold_names = set(gold_fields)
    pred_names = set(predicted_fields)
    matched = sorted(
        name
        for name in gold_names & pred_names
        if canonical(gold_fields[name], predicted_fields[name].field_type)
        == canonical(predicted_fields[name].value, predicted_fields[name].field_type)
    )
    missing = sorted(gold_names - set(matched))
    extra = sorted(pred_names - set(matched))
    mismatched = sorted((gold_names & pred_names) - set(matched))

    precision = _ratio(len(matched), len(pred_names), empty_is_one=not gold_names)
    recall = _ratio(len(matched), len(gold_names), empty_is_one=not pred_names)

    gold_counter = Counter(item_key(item) for item in gold_items)
    pred_counter = Counter(item_key(item) for item in predicted.line_items)
    matched_items = sum((gold_counter & pred_counter).values())

    item_precision = _ratio(matched_items, sum(pred_counter.values()), empty_is_one=not gold_counter)
    item_recall = _ratio(matched_items, sum(gold_counter.values()), empty_is_one=not pred_counter)

    gold_doc_type = str(gold.get("doc_type", "") or "")
    doc_type_ok = not gold_doc_type or gold_doc_type == predicted.doc_type

    exact = int(
        not missing
        and not extra
        and not mismatched
        and gold_counter == pred_counter
        and doc_type_ok
    )

    return DocScore(
        sample=sample,
        provider=predicted.provider,
        gold_doc_type=gold_doc_type,
        pred_doc_type=predicted.doc_type,
        gold_fields=len(gold_names),
        pred_fields=len(pred_names),
        matched_fields=len(matched),
        field_precision=precision,
        field_recall=recall,
        field_f1=_f1(precision, recall),
        gold_line_items=sum(gold_counter.values()),
        pred_line_items=sum(pred_counter.values()),
        matched_line_items=matched_items,
        item_precision=item_precision,
        item_recall=item_recall,
        item_f1=_f1(item_precision, item_recall),
        doc_type_ok=doc_type_ok,
        exact_match=exact,
        missing=missing,
        extra=extra,
        mismatched=mismatched,
    )


def load_gold(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def evaluate_run(
    samples_dir: str | Path,
    gold_dir: str | Path,
    *,
    provider: str = "rules",
    out_dir: str | Path | None = None,
    min_confidence: float = 0.0,
    write_outputs: bool = True,
) -> EvalReport:
    """Extract every sample, score it against gold, and optionally write reports."""
    samples_path = Path(samples_dir)
    gold_path = Path(gold_dir)
    engine = get_provider(provider)

    report = EvalReport(
        provider=engine.name,
        samples_dir=str(samples_path),
        gold_dir=str(gold_path),
        min_confidence=min_confidence,
    )

    for sample_path in sorted(samples_path.glob("*.txt")):
        gold_file = gold_path / f"{sample_path.stem}.json"
        if not gold_file.is_file():
            report.skipped.append(sample_path.name)
            continue
        text = sample_path.read_text(encoding="utf-8")
        document = engine.extract(source=sample_path, text=text)
        validate_document(document)
        gold = load_gold(gold_file)
        report.scores.append(
            score_document(
                document,
                gold,
                sample=sample_path.stem,
                min_confidence=min_confidence,
            )
        )

    if write_outputs and out_dir is not None:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_metrics_csv(report, destination / "metrics.csv")
        write_markdown_report(report, destination / "report.md")
    return report


def write_metrics_csv(report: EvalReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for score in report.scores:
            writer.writerow(score.to_row())
        writer.writerow(
            {
                "sample": "MACRO-AVERAGE",
                "provider": report.provider,
                "doc_type_ok": "",
                "field_precision": f"{report.macro_field_precision:.4f}",
                "field_recall": f"{report.macro_field_recall:.4f}",
                "field_f1": f"{report.macro_field_f1:.4f}",
                "exact_match": f"{report.exact_matches}/{report.count}",
                "missing": "",
                "extra": "",
                "mismatched": "",
            }
        )
    return destination


def _table_rows(report: EvalReport) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = [
        (
            "sample",
            "gold",
            "pred",
            "match",
            "P",
            "R",
            "F1",
            "items",
            "exact",
        )
    ]
    for score in report.scores:
        rows.append(
            (
                score.sample,
                str(score.gold_fields),
                str(score.pred_fields),
                str(score.matched_fields),
                f"{score.field_precision:.3f}",
                f"{score.field_recall:.3f}",
                f"{score.field_f1:.3f}",
                f"{score.matched_line_items}/{score.gold_line_items}",
                str(score.exact_match),
            )
        )
    rows.append(
        (
            "MACRO-AVERAGE",
            "",
            "",
            "",
            f"{report.macro_field_precision:.3f}",
            f"{report.macro_field_recall:.3f}",
            f"{report.macro_field_f1:.3f}",
            "",
            f"{report.exact_matches}/{report.count}",
        )
    )
    return rows


def format_table(report: EvalReport) -> str:
    """Fixed-width ASCII table, the one the CLI prints."""
    rows = _table_rows(report)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines: list[str] = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(row))))
    return "\n".join(lines)


def _detail_block(score: DocScore) -> list[str]:
    if not (score.missing or score.extra or score.mismatched):
        return []
    lines = [f"### {score.sample}"]
    if score.mismatched:
        lines.append(f"- mismatched values: {', '.join(score.mismatched)}")
    if score.missing:
        lines.append(f"- missed (in gold, not predicted): {', '.join(score.missing)}")
    if score.extra:
        lines.append(f"- spurious (predicted, not in gold): {', '.join(score.extra)}")
    lines.append("")
    return lines


def write_markdown_report(report: EvalReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Evaluation report",
        "",
        f"- Provider: `{report.provider}`",
        f"- Samples: `{report.samples_dir}`",
        f"- Gold: `{report.gold_dir}`",
        f"- Min field confidence: `{report.min_confidence}`",
        f"- Generated: {generated}",
        "",
        format_table(report),
        "",
        "## Metric definitions",
        "",
        "- **field precision** = correctly predicted fields / all predicted fields",
        "- **field recall** = correctly predicted fields / all gold fields",
        "- **field F1** = harmonic mean of the two",
        "- **exact** = 1 only when every gold field matched, no spurious field was",
        "  produced, line items matched, and `doc_type` was right",
        "- values are compared after canonicalisation (ISO dates, plain numbers,",
        "  normalised phone/email), so `₹1,03,368.00` equals `103368.0`",
        "",
    ]
    if report.skipped:
        lines += ["## Skipped (no gold file)", ""]
        lines += [f"- `{name}`" for name in report.skipped]
        lines.append("")

    details: list[str] = []
    for score in report.scores:
        details.extend(_detail_block(score))
    if details:
        lines += ["## Per-sample discrepancies", ""] + details

    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
