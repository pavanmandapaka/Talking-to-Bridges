"""Text chunking module with configurable overlap and deterministic metadata.
"""

from dataclasses import asdict, dataclass
from typing import Any
from app.core.config import settings
from rag.document_loader import DocumentPage


@dataclass
class TextChunk:
    """Represents a text chunk with complete traceable metadata."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    document_id: str
    chunk_index: int
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextChunker:
    """Configurable text chunker preserving page metadata and deterministic numbering."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

    def chunk_text(
        self,
        text: str,
        source_file: str,
        page_number: int,
        document_id: str,
        start_chunk_index: int = 1,
    ) -> list[TextChunk]:
        """Split single page/document text into overlapping chunks."""
        if not text or not text.strip():
            return []

        chunks: list[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        text_len = len(text)
        current_index = start_chunk_index

        start = 0
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If we're not at the very end, try to break at a natural boundary (newline or whitespace)
            if end < text_len:
                # Look backwards for paragraph boundary or space within the overlap window
                boundary = -1
                for sep in ["\n\n", "\n", ". ", " "]:
                    found = text.rfind(sep, start + step, end)
                    if found != -1:
                        boundary = found + len(sep)
                        break
                if boundary != -1 and boundary > start:
                    end = boundary

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunk_id = f"{document_id}-chunk-{current_index:04d}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_content,
                        source_file=source_file,
                        page_number=page_number,
                        document_id=document_id,
                        chunk_index=current_index,
                        char_start=start,
                        char_end=end,
                    )
                )
                current_index += 1

            if end >= text_len:
                break
            
            start = end - self.chunk_overlap
            if start <= chunks[-1].char_start:
                start = chunks[-1].char_start + step

        return chunks

    def chunk_pages(
        self,
        pages: list[DocumentPage],
        document_id: str,
    ) -> list[TextChunk]:
        """Chunk a list of DocumentPage objects, maintaining sequential chunk indexing."""
        all_chunks: list[TextChunk] = []
        current_index = 1

        for page in pages:
            page_chunks = self.chunk_text(
                text=page.text,
                source_file=page.source_file,
                page_number=page.page_number,
                document_id=document_id,
                start_chunk_index=current_index,
            )
            all_chunks.extend(page_chunks)
            current_index += len(page_chunks)

        return all_chunks
