"""Text cleaning module for engineering document RAG pipeline.

Preserves engineering units, numerical values, dates, percentages,
and structural formatting while removing extraction noise and whitespace clutter.
"""

import re


def clean_text(text: str) -> str:
    """Clean and normalize extracted document text while preserving engineering data.

    Args:
        text: Raw text extracted from PDF, DOCX, or TXT.

    Returns:
        Cleaned, readable text preserving units, numbers, dates, and punctuation.
    """
    if not text:
        return ""

    # Normalize carriage returns and line feeds
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable control characters and page breaks/form feeds,
    # but keep standard newlines, tabs, and unicode characters (like °C, µm, etc.)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Normalize horizontal whitespace (multiple spaces/tabs to a single space)
    # while preserving newlines
    lines = cleaned.split("\n")
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    # Rejoin lines
    cleaned = "\n".join(cleaned_lines)

    # Collapse 3 or more consecutive newlines into 2 (paragraph break)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
