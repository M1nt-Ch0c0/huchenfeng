"""
Utility helpers for loading and formatting prompt templates.
"""

from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import Any, Mapping


class MissingPlaceholderError(ValueError):
    """Raised when required placeholders are missing from the input arguments."""


_PROMPTS_DIR = Path(__file__).resolve().parent
_TEMPLATE_CACHE: dict[str, str] = {}


def _load_template(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    cached = _TEMPLATE_CACHE.get(filename)
    if cached is not None:
        return cached
    content = path.read_text(encoding="utf-8")
    _TEMPLATE_CACHE[filename] = content
    return content


def render_denoise_and_split(**kwargs: Any) -> str:
    """
    Render the ``denoise_and_split.md`` template with the provided placeholders.

    Example:
        render_denoise_and_split(text="原始文本")
    """

    template = _load_template("denoise_and_split.md")
    required_fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    missing = required_fields - kwargs.keys()
    if missing:
        raise MissingPlaceholderError(
            f"Missing values for placeholders: {', '.join(sorted(missing))}"
        )
    try:
        return template.format(**kwargs)
    except KeyError as exc:  # pragma: no cover - defensive
        raise MissingPlaceholderError(f"Missing placeholder: {exc}") from exc
