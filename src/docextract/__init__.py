"""sarvam-vision-doc-extract — document / form -> validated structured JSON.

Public surface::

    from docextract import Document, extract, json_schema

The default :class:`~docextract.providers.RuleBasedProvider` runs fully offline
with **zero** third-party dependencies. OCR and LLM backends are optional and
only load their libraries when explicitly selected.
"""

from __future__ import annotations

from .schema import DOC_TYPES, FIELD_TYPES, Document, Field, LineItem, json_schema

__version__ = "1.0.0"

__all__ = [
    "DOC_TYPES",
    "FIELD_TYPES",
    "Document",
    "Field",
    "LineItem",
    "json_schema",
    "extract",
    "__version__",
]


def extract(text: str, *, provider: str = "rules", source: str | None = None) -> Document:
    """Convenience wrapper: pick a provider by name and extract ``text``."""
    from .providers import get_provider

    return get_provider(provider).extract(source=source, text=text)
