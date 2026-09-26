"""Document ingestion and storage pipeline.

Coordinates loading, cleaning, chunking, and persistence.
Supports PDF, DOCX, TXT (text pipeline) and CSV (tabular pipeline).
"""

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import uuid
from typing import Any
from fastapi import UploadFile

from app.core.config import settings
from app.core.logging_config import logger
from rag.chunker import TextChunk, TextChunker
from rag.csv_processor import CSVChunk, CSVProcessingError, process_csv
from rag.document_loader import (
    DocumentExtractionError,
    DocumentHandlingError,
    DocumentPage,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_document,
)
from rag.embeddings import EmbeddingService
from rag.vector_store import FAISSVectorStore

# Resolve data directories using settings
DOCUMENTS_DIR = Path(settings.DOCUMENTS_DIR)
PROCESSED_DIR = Path(settings.PROCESSED_DIR)


@dataclass
class IngestionResult:
    """Summary of document ingestion (PDF/DOCX/TXT or CSV).

    The ``rows``, ``columns``, and ``column_names`` fields are only populated
    for CSV uploads; they default to ``None`` / empty list for other types.
    """
    document_id: str
    filename: str
    page_count: int
    chunks_created: int
    chunks: list[TextChunk | CSVChunk]
    preview_text: str
    # CSV-specific summary fields (None / empty for non-CSV documents)
    rows: int | None = None
    columns: int | None = None
    column_names: list[str] = field(default_factory=list)


def save_uploaded_file(file: UploadFile) -> str:
    """Saves an uploaded file to the documents directory and returns a document ID."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix if file.filename else ""
    safe_filename = f"{document_id}{file_extension}"
    file_path = DOCUMENTS_DIR / safe_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise DocumentHandlingError(f"Failed to save file: {e}") from e

    return document_id


def get_document_path(document_id: str) -> Path | None:
    """Retrieves the path to the saved document by its ID."""
    if not DOCUMENTS_DIR.exists():
        return None

    for file_path in DOCUMENTS_DIR.iterdir():
        if file_path.stem == document_id:
            return file_path

    return None


def delete_document(document_id: str) -> bool:
    """Deletes a document by its ID."""
    file_path = get_document_path(document_id)
    if file_path and file_path.exists():
        try:
            file_path.unlink()
            return True
        except Exception as e:
            raise DocumentHandlingError(f"Failed to delete file: {e}") from e
    return False


def extract_document_chunks(
    filename: str, content: bytes, max_chars: int | None = None
) -> list[tuple[str, int]]:
    """Extract and chunk text from a supported document.

    Maintained for backward compatibility.
    """
    pages = load_document(filename, content)
    chunker = TextChunker(chunk_size=max_chars or settings.CHUNK_SIZE)
    chunks = chunker.chunk_pages(pages, document_id="legacy-doc")
    return [(chunk.text, chunk.page_number) for chunk in chunks]


def ingest_document(
    filename: str,
    content: bytes,
    document_id: str,
    embedding_service: EmbeddingService | None = None,
    vector_store: FAISSVectorStore | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionResult:
    """Complete document ingestion pipeline:

    1. Load and parse document (.pdf, .docx, .txt)
    2. Clean text
    3. Generate overlapping chunks with deterministic metadata
    4. Generate local Sentence Transformer embeddings
    5. Store in FAISS vector store & persist index
    """
    logger.info(f"Ingesting document '{filename}' as {document_id}")
    pages: list[DocumentPage] = load_document(filename, content)

    # Chunk text
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_pages(pages, document_id=document_id)

    if not chunks:
        raise EmptyDocumentError(f"No chunks generated for document '{filename}'")

    # Generate preview text (first ~3000 chars across initial pages)
    preview_parts = []
    total_preview_len = 0
    for page in pages:
        preview_parts.append(f"--- Page {page.page_number} ---\n{page.text}")
        total_preview_len += len(page.text)
        if total_preview_len >= 3000:
            break
    preview_text = "\n\n".join(preview_parts)

    # Vector indexing if services are provided
    if embedding_service is not None and vector_store is not None:
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_documents(chunk_texts)
        chunk_dicts = [c.to_dict() for c in chunks]
        vector_store.add_chunks(chunk_dicts, embeddings)
        vector_store.save()

    return IngestionResult(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        chunks_created=len(chunks),
        chunks=chunks,
        preview_text=preview_text,
    )


def ingest_csv(
    filename: str,
    content: bytes,
    document_id: str,
    embedding_service: EmbeddingService | None = None,
    vector_store: FAISSVectorStore | None = None,
    rows_per_chunk: int = 20,
) -> IngestionResult:
    """CSV ingestion pipeline:

    1. Parse CSV with pandas (robust encoding detection)
    2. Convert rows → structured semantic text records (CSVChunks)
    3. Generate Sentence Transformer embeddings for each chunk
    4. Store embeddings + metadata in FAISS vector store & persist

    Args:
        filename:        Original CSV filename.
        content:         Raw bytes of the uploaded CSV file.
        document_id:     Unique document ID assigned at upload time.
        embedding_service: Optional EmbeddingService; if None, embeddings
                           are skipped (useful for unit-testing the chunking).
        vector_store:    Optional FAISSVectorStore; if None, FAISS insertion
                         is skipped.
        rows_per_chunk:  Number of data rows grouped into each semantic chunk.

    Returns:
        :class:`IngestionResult` with CSV-specific ``rows``, ``columns``,
        and ``column_names`` populated.

    Raises:
        EmptyCSVError / CorruptedCSVError from csv_processor on bad input.
    """
    logger.info(f"Ingesting CSV '{filename}' as document '{document_id}'")

    csv_chunks: list[CSVChunk] = process_csv(
        content=content,
        filename=filename,
        document_id=document_id,
        rows_per_chunk=rows_per_chunk,
    )

    # Build a brief preview from the first chunk's text
    preview_text = csv_chunks[0].text[:3000] if csv_chunks else ""

    # Embed and index
    if embedding_service is not None and vector_store is not None:
        chunk_texts = [c.text for c in csv_chunks]
        embeddings = embedding_service.embed_documents(chunk_texts)
        chunk_dicts = [c.to_dict() for c in csv_chunks]
        vector_store.add_chunks(chunk_dicts, embeddings)
        vector_store.save()

    # Compute summary statistics for the UI / API response
    total_rows = csv_chunks[-1].row_end + 1 if csv_chunks else 0
    col_names = csv_chunks[0].column_names if csv_chunks else []

    return IngestionResult(
        document_id=document_id,
        filename=filename,
        # CSV has no concept of "pages"; we use chunk count as a proxy
        page_count=len(csv_chunks),
        chunks_created=len(csv_chunks),
        chunks=csv_chunks,  # type: ignore[arg-type]
        preview_text=preview_text,
        rows=total_rows,
        columns=len(col_names),
        column_names=col_names,
    )
