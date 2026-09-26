"""CSV processing module for the RAG/SHM pipeline.

Reads tabular CSV data, converts rows into structured semantic text
representations, and produces chunk metadata compatible with the existing
FAISS + Sentence Transformer pipeline.

Design decisions
----------------
* Preserves numeric precision exactly as stored in the CSV.
* Never imputes missing values with zero — represents them as "missing".
* Does NOT apply NLP preprocessing (no stemming / stop-word removal).
* Row grouping size is configurable via ``rows_per_chunk`` (default 20).
* Each CSV chunk carries full provenance metadata (document_id, row_start,
  row_end, column_names, etc.) so retrieved chunks can be traced back to
  their exact location in the original file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.logging_config import logger


# ---------------------------------------------------------------------------
# CSV-specific exceptions (mirrors the pattern in document_loader.py)
# ---------------------------------------------------------------------------

class CSVProcessingError(Exception):
    """Base exception for CSV processing errors."""


class EmptyCSVError(CSVProcessingError):
    """Raised when the uploaded CSV contains no usable rows or columns."""


class CorruptedCSVError(CSVProcessingError):
    """Raised when the CSV cannot be parsed (encoding issues, malformed data)."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CSVChunk:
    """A semantic text record derived from a group of CSV rows.

    Attributes
    ----------
    chunk_id:       Globally unique identifier for this chunk.
    document_id:    ID of the parent CSV document.
    source_file:    Original filename of the CSV.
    file_type:      Always ``"csv"`` for CSV chunks.
    text:           Structured semantic text representation of the rows.
    row_start:      Inclusive start row index (0-based, data rows only).
    row_end:        Inclusive end row index (0-based, data rows only).
    chunk_index:    Sequential 1-based index within this document.
    num_rows:       Number of data rows in this chunk.
    column_names:   List of column headers from the original CSV.
    original_row_indices: Actual DataFrame index values for these rows.
    """

    chunk_id: str
    document_id: str
    source_file: str
    file_type: str
    text: str
    row_start: int
    row_end: int
    chunk_index: int
    num_rows: int
    column_names: list[str]
    original_row_indices: list[int] = field(default_factory=list)

    # page_number kept for compatibility with RetrievedChunk / ChunkResponse
    page_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for storage in FAISS metadata."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "file_type": self.file_type,
            "text": self.text,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "chunk_index": self.chunk_index,
            "num_rows": self.num_rows,
            "column_names": self.column_names,
            "original_row_indices": self.original_row_indices,
            # Keep these fields so retrieval.py / routes.py can access them
            # without special-casing the CSV type.
            "page_number": self.page_number,
        }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _format_value(value: Any) -> str:
    """Render a single cell value as a human-readable string.

    Rules
    -----
    * ``NaN`` / ``None`` / ``pd.NA`` → ``"missing"``
    * Integers stored as floats with no fractional part → strip the ``.0``
      (e.g. 120.0 → ``"120"``).
    * All other floats → use Python's default ``str()`` to preserve precision.
    * Everything else → ``str(value)``.
    """
    try:
        if pd.isna(value):
            return "missing"
    except (TypeError, ValueError):
        pass  # pd.isna raises for some types; treat as non-missing

    if isinstance(value, float):
        # Preserve full precision but strip unnecessary trailing .0
        if value == int(value) and not (value != value):  # not NaN
            # Only strip .0 for "round" floats that are actually integers
            # but keep decimal precision for values like 32.5, 0.0045
            int_val = int(value)
            if float(int_val) == value:
                return str(int_val)
        return repr(value)  # repr preserves full float precision

    return str(value)


def _clean_column_name(name: str) -> str:
    """Strip leading/trailing whitespace from a column name."""
    return str(name).strip()


def rows_to_semantic_text(
    df: pd.DataFrame,
    row_indices: list[int],
    column_names: list[str],
) -> str:
    """Convert a group of DataFrame rows into a structured semantic text block.

    Each row is formatted as a newline-separated list of ``Column: Value``
    pairs, then rows are separated by blank lines.

    Example output::

        Timestamp: 2026-01-01. Temperature: 32.5. Stress: 120.4.
        Deflection: 2.1. Crack_width: 1.2.

        Timestamp: 2026-01-02. Temperature: 33.1. Stress: 121.8.
        Deflection: 2.3. Crack_width: 1.4.

    Args:
        df:           The full DataFrame (used for `.loc` access).
        row_indices:  List of DataFrame index labels for this chunk's rows.
        column_names: Cleaned column names to use as labels.

    Returns:
        A multi-line string ready for embedding generation.
    """
    row_texts: list[str] = []

    for idx in row_indices:
        try:
            row = df.loc[idx]
        except KeyError:
            continue

        parts: list[str] = []
        for col in column_names:
            raw_val = row.get(col, pd.NA)
            formatted = _format_value(raw_val)
            parts.append(f"{col}: {formatted}")

        if parts:
            # Join all key-value pairs for this row with ". " separator
            row_text = ". ".join(parts) + "."
            row_texts.append(row_text)

    return "\n\n".join(row_texts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(content: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV bytes into a pandas DataFrame.

    Attempts UTF-8 decoding first, then falls back to latin-1.
    Strips whitespace from column names.

    Args:
        content:  Raw bytes of the uploaded CSV file.
        filename: Original filename (used in error messages).

    Returns:
        A pandas DataFrame with cleaned column names.

    Raises:
        CorruptedCSVError: If the CSV cannot be parsed.
        EmptyCSVError:     If the CSV has no columns or no data rows.
    """
    for encoding in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
                # Keep all values as-is; do not infer booleans from strings
                dtype=None,
                # Let pandas handle date columns as strings for now
                # (avoids silent coercion to NaT)
                keep_default_na=True,
            )
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise EmptyCSVError(
                f"CSV file '{filename}' is empty or contains no data."
            ) from exc
        except Exception as exc:
            raise CorruptedCSVError(
                f"Failed to parse CSV file '{filename}': {exc}"
            ) from exc
    else:
        raise CorruptedCSVError(
            f"Could not decode CSV file '{filename}' with UTF-8 or latin-1 encoding."
        )

    # Clean column names
    df.columns = [_clean_column_name(c) for c in df.columns]

    if df.empty or len(df.columns) == 0:
        raise EmptyCSVError(
            f"CSV file '{filename}' contains no usable columns."
        )

    if len(df) == 0:
        raise EmptyCSVError(
            f"CSV file '{filename}' has headers but no data rows."
        )

    logger.info(
        f"Loaded CSV '{filename}': {len(df)} rows × {len(df.columns)} columns"
    )
    return df


def process_csv(
    content: bytes,
    filename: str,
    document_id: str,
    rows_per_chunk: int = 20,
) -> list[CSVChunk]:
    """Convert a CSV file into a list of semantic CSVChunk objects.

    Pipeline::

        bytes → DataFrame → row groups → semantic text → CSVChunk list

    Args:
        content:        Raw bytes of the CSV file.
        filename:       Original filename (for metadata / error messages).
        document_id:    Unique document identifier assigned by the upload pipeline.
        rows_per_chunk: Maximum number of data rows per semantic chunk.
                        Defaults to 20.

    Returns:
        A list of :class:`CSVChunk` objects (non-empty).

    Raises:
        EmptyCSVError:    If the CSV has no usable data.
        CorruptedCSVError: If the CSV cannot be parsed.
    """
    df = load_csv(content, filename)
    column_names: list[str] = list(df.columns)
    all_indices: list[int] = list(df.index)

    chunks: list[CSVChunk] = []
    chunk_index = 1

    # Walk through the DataFrame in groups of ``rows_per_chunk`` rows
    for group_start in range(0, len(all_indices), rows_per_chunk):
        group_indices = all_indices[group_start: group_start + rows_per_chunk]

        semantic_text = rows_to_semantic_text(df, group_indices, column_names)
        if not semantic_text.strip():
            continue  # Skip groups that produce no text (all-null rows, etc.)

        row_start = group_start
        row_end = group_start + len(group_indices) - 1
        chunk_id = f"{document_id}-csv-chunk-{chunk_index:04d}"

        chunks.append(
            CSVChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                source_file=filename,
                file_type="csv",
                text=semantic_text,
                row_start=row_start,
                row_end=row_end,
                chunk_index=chunk_index,
                num_rows=len(group_indices),
                column_names=column_names,
                original_row_indices=list(group_indices),
                page_number=1,
            )
        )
        chunk_index += 1

    if not chunks:
        raise EmptyCSVError(
            f"CSV file '{filename}' produced no semantic chunks "
            "(all rows may be empty or null)."
        )

    logger.info(
        f"CSV '{filename}' → {len(chunks)} semantic chunks "
        f"({rows_per_chunk} rows/chunk)"
    )
    return chunks


def get_csv_summary(content: bytes, filename: str) -> dict[str, Any]:
    """Return a lightweight summary dict for display in the Streamlit UI.

    Does not produce chunks; used only for UI feedback after upload.

    Returns keys: ``num_rows``, ``num_columns``, ``column_names``,
    ``preview_df`` (a small DataFrame of the first 10 rows).
    """
    df = load_csv(content, filename)
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": list(df.columns),
        "preview_df": df.head(10),
    }
