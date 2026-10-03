"""Command line interface.

    python3 -m docextract extract   --input data/samples/invoice_en.txt --provider rules --out results/
    python3 -m docextract evaluate  --samples data/samples --gold data/gold --provider rules --out results/
    python3 -m docextract providers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .evaluate import evaluate_run, format_table
from .providers import ProviderError, available_providers, get_provider
from .validate import validate_document

__all__ = ["main", "build_parser"]

PROG = "docextract"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Document / form -> validated structured JSON.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="extract one document and print/write the JSON",
    )
    extract.add_argument("--input", "-i", required=True, help="path to a .txt (or image for --provider ocr)")
    extract.add_argument(
        "--provider",
        "-p",
        default="rules",
        help="extraction backend: rules (default, offline) | ocr | llm",
    )
    extract.add_argument("--out", "-o", default="results", help="directory for the JSON output")
    extract.add_argument("--indent", type=int, default=2, help="JSON indent (default 2)")
    extract.add_argument("--quiet", "-q", action="store_true", help="write the file but do not print it")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="score a provider against data/gold and write metrics.csv + report.md",
    )
    evaluate.add_argument("--samples", default="data/samples", help="directory of .txt samples")
    evaluate.add_argument("--gold", default="data/gold", help="directory of gold .json files")
    evaluate.add_argument("--provider", "-p", default="rules", help="extraction backend (default: rules)")
    evaluate.add_argument("--out", "-o", default="results", help="directory for metrics.csv and report.md")
    evaluate.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="ignore predicted fields below this confidence (default 0.0)",
    )

    subparsers.add_parser("providers", help="list available extraction backends")
    return parser


def _command_extract(args: argparse.Namespace) -> int:
    source = Path(args.input)
    if not source.is_file():
        raise ProviderError(f"input file not found: {source}")

    provider = get_provider(args.provider)
    # The provider decides how to read the source (text file, sibling .txt,
    # OCR) and reports unreadable input as a ProviderError, not a traceback.
    document = provider.extract(source=source)
    validate_document(document)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem}.extracted.json"
    payload = document.to_json(indent=args.indent)
    out_path.write_text(payload + "\n", encoding="utf-8")

    if not args.quiet:
        print(payload)
    print(f"wrote {out_path}", file=sys.stderr)
    report = document.validation or {}
    print(
        f"doc_type={document.doc_type} fields={len(document.fields)} "
        f"line_items={len(document.line_items)} validation_passed={report.get('passed')}",
        file=sys.stderr,
    )
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    samples_dir = Path(args.samples)
    gold_dir = Path(args.gold)
    if not samples_dir.is_dir():
        raise ProviderError(f"samples directory not found: {samples_dir}")
    if not gold_dir.is_dir():
        raise ProviderError(f"gold directory not found: {gold_dir}")

    report = evaluate_run(
        samples_dir,
        gold_dir,
        provider=args.provider,
        out_dir=args.out,
        min_confidence=args.min_confidence,
    )
    if not report.scores:
        raise ProviderError(
            f"no sample/gold pairs found in {samples_dir} + {gold_dir} (expected matching .txt / .json)"
        )

    print(format_table(report))
    print()
    print(
        f"provider={report.provider}  documents={report.count}  "
        f"macro F1={report.macro_field_f1:.3f}  exact={report.exact_matches}/{report.count}"
    )
    for name in report.skipped:
        print(f"warning: skipped {name} (no gold file)", file=sys.stderr)
    print(f"wrote {Path(args.out) / 'metrics.csv'} and {Path(args.out) / 'report.md'}")
    return 0


def _command_providers(_args: argparse.Namespace) -> int:
    for provider in available_providers():
        tag = "optional" if provider.optional else "default"
        print(f"{provider.name:<8} [{tag}] {provider.description}")
        print(f"{'':<8}   requires: {provider.requires}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    handlers = {
        "extract": _command_extract,
        "evaluate": _command_evaluate,
        "providers": _command_providers,
    }
    try:
        return handlers[args.command](args)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
