import io
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from rag.ingestion import (
    DOCUMENTS_DIR,
    delete_document,
    get_document_path,
    save_uploaded_file,
)


@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: Ensure the directory exists
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Teardown: Clean up any files created during tests
    for file_path in DOCUMENTS_DIR.iterdir():
        if file_path.is_file() and file_path.name != ".gitkeep":
            file_path.unlink()


def test_save_uploaded_file():
    # Create a mock UploadFile
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_document.pdf"
    
    # We need to simulate the actual file object inside UploadFile
    # A simple BytesIO buffer will do
    content = b"Dummy PDF content"
    mock_file.file = io.BytesIO(content)

    document_id = save_uploaded_file(mock_file)

    assert document_id is not None
    
    # Verify file was saved correctly
    file_path = get_document_path(document_id)
    assert file_path is not None
    assert file_path.exists()
    assert file_path.suffix == ".pdf"
    assert file_path.read_bytes() == content


def test_get_document_path_not_found():
    assert get_document_path(str(uuid.uuid4())) is None


def test_delete_document():
    # First, save a document
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.txt"
    mock_file.file = io.BytesIO(b"content")
    
    document_id = save_uploaded_file(mock_file)
    
    # Verify it exists
    assert get_document_path(document_id) is not None
    
    # Now delete it
    assert delete_document(document_id) is True
    
    # Verify it's gone
    assert get_document_path(document_id) is None


def test_delete_document_not_found():
    assert delete_document(str(uuid.uuid4())) is False
