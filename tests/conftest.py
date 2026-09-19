import pytest

from app.api.routes import _uploaded_documents


@pytest.fixture(autouse=True)
def clear_uploaded_documents():
    """Keep endpoint tests isolated while documents persist during app runtime."""
    _uploaded_documents.clear()
    yield
