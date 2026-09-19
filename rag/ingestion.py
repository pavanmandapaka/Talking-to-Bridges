import io
import re
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile
from docx import Document
from pypdf import PdfReader

# Use the project root to resolve the data directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

class DocumentHandlingError(Exception):
    """Base exception for document handling errors."""


def extract_document_chunks(
    filename: str, content: bytes, max_chars: int = 1600
) -> list[tuple[str, int]]:
    """Extract text from a supported document and split it into chunks."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            pages = [
                page.extract_text() or ""
                for page in PdfReader(io.BytesIO(content)).pages
            ]
        except Exception:  # noqa: BLE001
            pages = [content.decode("utf-8", errors="replace")]
    elif suffix == ".docx":
        document = Document(io.BytesIO(content))
        pages = ["\n".join(paragraph.text for paragraph in document.paragraphs)]
    elif suffix == ".txt":
        pages = [content.decode("utf-8", errors="replace")]
    else:
        raise DocumentHandlingError("Only PDF, DOCX, and TXT files are supported")

    chunks: list[tuple[str, int]] = []
    for page_number, page_text in enumerate(pages, start=1):
        normalized = re.sub(r"\s+", " ", page_text).strip()
        for start in range(0, len(normalized), max_chars):
            text = normalized[start : start + max_chars].strip()
            if text:
                chunks.append((text, page_number))

    if not chunks:
        raise DocumentHandlingError("The uploaded document contains no readable text")
    return chunks

def save_uploaded_file(file: UploadFile) -> str:
    """Saves an uploaded file to the documents directory and returns a document ID.

    Args:
        file: The uploaded file from FastAPI.

    Returns:
        A unique document ID.
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix if file.filename else ""
    safe_filename = f"{document_id}{file_extension}"
    file_path = DOCUMENTS_DIR / safe_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:  # noqa: BLE001
        raise DocumentHandlingError(f"Failed to save file: {e}")

    return document_id

def get_document_path(document_id: str) -> Path | None:
    """Retrieves the path to the saved document by its ID.

    Args:
        document_id: The ID of the document.

    Returns:
        The Path to the document if it exists, else None.
    """
    if not DOCUMENTS_DIR.exists():
        return None

    for file_path in DOCUMENTS_DIR.iterdir():
        if file_path.stem == document_id:
            return file_path

    return None

def delete_document(document_id: str) -> bool:
    """Deletes a document by its ID.

    Args:
        document_id: The ID of the document.

    Returns:
        True if the document was deleted, False if not found.
    """
    file_path = get_document_path(document_id)
    if file_path and file_path.exists():
        try:
            file_path.unlink()
            return True
        except Exception as e:  # noqa: BLE001
            raise DocumentHandlingError(f"Failed to delete file: {e}")
    return False
