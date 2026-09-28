import tempfile
from pathlib import Path
import pytest

from app.api.routes import _uploaded_documents, vector_store
import rag.ingestion
import app.core.config


@pytest.fixture(autouse=True, scope="session")
def isolate_test_environment(tmp_path_factory):
    """Ensure automated tests write to a temporary sandbox, never polluting data/."""
    temp_dir = tmp_path_factory.mktemp("test_db")
    temp_docs = tmp_path_factory.mktemp("test_docs")
    
    # Redirect vector store directory and documents directory
    original_db_dir = vector_store.db_dir
    original_docs_dir = rag.ingestion.DOCUMENTS_DIR

    vector_store.db_dir = Path(temp_dir)
    rag.ingestion.DOCUMENTS_DIR = Path(temp_docs)

    yield

    vector_store.db_dir = original_db_dir
    rag.ingestion.DOCUMENTS_DIR = original_docs_dir


@pytest.fixture(autouse=True)
def clear_uploaded_documents():
    """Keep endpoint tests isolated while documents persist during app runtime."""
    _uploaded_documents.clear()
    vector_store.clear()
    yield

