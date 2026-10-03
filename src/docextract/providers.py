"""Extraction providers: the offline default plus optional OCR / LLM backends.

``rules`` is the default and needs nothing beyond the standard library.
``ocr`` and ``llm`` are *optional*: their third-party imports are guarded so
that selecting them without the dependency raises a clear
:class:`ProviderUnavailableError` telling you exactly what to ``pip install``
(or which key to set) instead of an opaque ``ImportError``.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from .extract_rules import extract_document
from .schema import Document

__all__ = [
    "Provider",
    "ProviderError",
    "ProviderUnavailableError",
    "RuleBasedProvider",
    "OcrTextProvider",
    "LlmProvider",
    "available_providers",
    "get_provider",
    "load_dotenv",
]

TEXT_SUFFIXES = frozenset({".txt", ".text", ".md", ".csv", ".json", ".log"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"})

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProviderError(RuntimeError):
    """Base class for provider failures that should reach the user as a message."""


class ProviderUnavailableError(ProviderError):
    """The selected provider cannot run in this environment."""


def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Read a ``.env``-style file into ``os.environ`` without overriding it.

    Lookup order: explicit ``path`` -> ``$DOCEXTRACT_ENV_FILE`` -> ``./.env`` ->
    ``<project>/.env``. Values already present in the environment win, so a
    real shell export always beats a file.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        override = os.environ.get("DOCEXTRACT_ENV_FILE")
        candidates = [Path(override)] if override else [Path.cwd() / ".env", _PROJECT_ROOT / ".env"]

    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            content = candidate.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file, just skip
            continue
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def _require(module_name: str, pip_name: str, feature: str) -> Any:
    """Import an optional dependency or raise an actionable error."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ProviderUnavailableError(
            f"The '{feature}' provider needs the optional package '{pip_name}' "
            f"(importing '{module_name}' failed: {exc}).\n"
            f"Install it with:  pip install {pip_name}"
        ) from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ProviderError(
            f"{path} is not UTF-8 text ({exc}). For images use --provider ocr."
        ) from exc
    except OSError as exc:
        raise ProviderError(f"cannot read {path}: {exc}") from exc


def _ocr_image(path: Path, engine: str = "auto") -> str:
    """Run real OCR over an image; both engines are optional."""
    if not path.is_file():
        raise ProviderError(f"input file not found: {path}")
    failures: list[str] = []

    if engine in ("auto", "tesseract"):
        try:
            pytesseract = _require("pytesseract", "pytesseract Pillow", "ocr")
            pil_image = _require("PIL.Image", "Pillow", "ocr")
        except ProviderUnavailableError as exc:
            failures.append(str(exc))
        else:
            try:
                language = os.environ.get("DOCEXTRACT_OCR_LANG", "eng+hin")
                image = pil_image.open(path)
                return str(pytesseract.image_to_string(image, lang=language))
            except Exception as exc:  # missing tesseract binary, bad image, ...
                failures.append(f"tesseract failed: {exc}")

    if engine in ("auto", "easyocr"):
        try:
            easyocr = _require("easyocr", "easyocr", "ocr")
        except ProviderUnavailableError as exc:
            failures.append(str(exc))
        else:
            try:
                reader = easyocr.Reader(["en", "hi"], gpu=False)
                return "\n".join(str(line) for line in reader.readtext(str(path), detail=0))
            except Exception as exc:
                failures.append(f"easyocr failed: {exc}")

    raise ProviderUnavailableError(
        "No working OCR engine. Tried:\n- " + "\n- ".join(failures)
        + "\nInstall one with:  pip install pytesseract Pillow"
        + "\n(and the binary:   sudo apt install tesseract-ocr tesseract-ocr-hin)"
    )


def _resolve_text(source: str | os.PathLike[str] | None, text: str | None) -> str:
    """Get document text from explicit text, a text file, a sibling .txt, or OCR."""
    if text is not None:
        return text
    if source is None:
        raise ProviderError("no input: pass --input <file>")

    path = Path(source)
    if not path.is_file():
        raise ProviderError(f"input file not found: {path}")
    if path.suffix.lower() in TEXT_SUFFIXES:
        return _read_text(path)

    sibling = path.with_suffix(".txt")
    if sibling.is_file():
        # Pre-extracted text next to the image: the honest, cheap path.
        return _read_text(sibling)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return _ocr_image(path)
    return _read_text(path)


class Provider(ABC):
    """Common interface every extraction backend implements."""

    name: ClassVar[str]
    description: ClassVar[str]
    requires: ClassVar[str] = "nothing (Python standard library only)"
    optional: ClassVar[bool] = False

    @abstractmethod
    def extract(
        self,
        *,
        source: str | os.PathLike[str] | None = None,
        text: str | None = None,
    ) -> Document:
        """Turn document text (or a path to it) into a :class:`Document`."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"


class RuleBasedProvider(Provider):
    """Default backend: offline label/regex rules. Deterministic, zero deps."""

    name = "rules"
    description = "Offline label + regex rules; deterministic and fully offline."
    requires = "nothing (Python standard library only)"
    optional = False

    def extract(
        self,
        *,
        source: str | os.PathLike[str] | None = None,
        text: str | None = None,
    ) -> Document:
        if text is None:
            if source is None:
                raise ProviderError("rules provider needs --input <file> or some text")
            path = Path(source)
            if not path.is_file():
                raise ProviderError(f"input file not found: {path}")
            text = _read_text(path)
        return extract_document(
            text,
            source=str(source) if source is not None else None,
            provider=self.name,
        )


class OcrTextProvider(Provider):
    """Image -> text -> the rule parser.

    Reuses a sibling ``.txt`` when present (that is what the sample corpus
    does), otherwise runs pytesseract or easyocr, both behind guarded imports.
    """

    name = "ocr"
    description = "OCR images (tesseract/easyocr) or reuse a sibling .txt, then parse."
    requires = "pytesseract + Pillow, or easyocr  (pip install pytesseract Pillow)"
    optional = True

    def extract(
        self,
        *,
        source: str | os.PathLike[str] | None = None,
        text: str | None = None,
    ) -> Document:
        resolved = _resolve_text(source, text)
        return extract_document(
            resolved,
            source=str(source) if source is not None else None,
            provider=self.name,
        )


_LLM_SYSTEM_PROMPT = """You are a document-intelligence extractor.
Return ONLY a JSON object (no prose, no markdown fence) with this shape:
{
  "doc_type": "invoice" | "form" | "fir" | "receipt" | "generic",
  "fields": {"<field_name>": {"value": <primitive>, "type": "<field_type>", "confidence": <0..1>}},
  "line_items": [{"description": "...", "quantity": 0, "unit_price": 0, "amount": 0}]
}
Rules: dates as YYYY-MM-DD; money as plain JSON numbers (no commas, no currency
symbols); phone as a 10-digit string; email lowercase; type must be one of
person_name, org_name, address, date, currency_amount, phone, email, doc_id,
id_number, text. Use an empty line_items array when there is no table."""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LlmProvider(Provider):
    """Optional LLM backend, gated on ``SARVAM_API_KEY``.

    Wired against an OpenAI-compatible ``/chat/completions`` endpoint. It is
    intentionally untested in this repo (no key is committed, and none ships
    with the sample corpus) — see "Honest limits" in the README.
    """

    name = "llm"
    description = "Optional LLM extraction via SARVAM_API_KEY (network, untested here)."
    requires = "requests (pip install requests) + SARVAM_API_KEY in .env"
    optional = True

    def extract(
        self,
        *,
        source: str | os.PathLike[str] | None = None,
        text: str | None = None,
    ) -> Document:
        load_dotenv()
        api_key = os.environ.get("SARVAM_API_KEY", "").strip()
        if not api_key:
            raise ProviderUnavailableError(
                "The 'llm' provider is gated on SARVAM_API_KEY.\n"
                "Copy .env.example to .env and fill in SARVAM_API_KEY, "
                "SARVAM_API_BASE and VISION_MODEL (never commit .env),\n"
                "or export SARVAM_API_KEY in your shell. "
                "The default --provider rules needs no key at all."
            )
        base = os.environ.get("SARVAM_API_BASE", "").strip().rstrip("/")
        model = os.environ.get("VISION_MODEL", "").strip()
        if not base or not model:
            raise ProviderUnavailableError(
                "SARVAM_API_BASE and VISION_MODEL must both be set for --provider llm "
                "(see .env.example)."
            )
        requests = _require("requests", "requests", "llm")
        resolved = _resolve_text(source, text)

        payload: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": resolved},
            ],
        }
        try:
            response = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=float(os.environ.get("DOCEXTRACT_LLM_TIMEOUT", "60")),
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ProviderError(f"LLM request to {base} failed: {exc}") from exc

        try:
            parsed = json.loads(_strip_json_fence(content))
            document = Document.from_dict(parsed)
        except Exception as exc:
            raise ProviderError(f"LLM response was not a usable document JSON: {exc}") from exc
        document.provider = self.name
        document.source = str(source) if source is not None else None
        return document


def _strip_json_fence(content: str) -> str:
    """Unwrap ```json fences, or fall back to the outermost braces."""
    match = _JSON_FENCE_RE.search(content)
    if match:
        return match.group(1).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        return content[start : end + 1]
    return content.strip()


_PROVIDERS: dict[str, type[Provider]] = {
    RuleBasedProvider.name: RuleBasedProvider,
    OcrTextProvider.name: OcrTextProvider,
    LlmProvider.name: LlmProvider,
}


def available_providers() -> list[Provider]:
    """One instance of every registered provider, default first."""
    order = ["rules", "ocr", "llm"]
    return [_PROVIDERS[name]() for name in order if name in _PROVIDERS]


def get_provider(name: str | None) -> Provider:
    """Look up a provider by name, raising a helpful error when unknown."""
    key = (name or "rules").strip().lower()
    provider_class = _PROVIDERS.get(key)
    if provider_class is None:
        raise ProviderError(
            f"unknown provider {name!r}; available: {', '.join(sorted(_PROVIDERS))}"
        )
    return provider_class()
