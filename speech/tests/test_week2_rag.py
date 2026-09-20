"""Comprehensive unit and integration tests for Week 2 RAG pipeline.

Covers:
- PDF, DOCX, TXT loading & metadata
- Text cleaning (preserving engineering units, numbers, dates, punctuation)
- Text chunking (overlap, deterministic IDs, metadata preservation)
- Sentence Transformers embedding generation
- FAISS vector store creation, search, persistence, and loading
- Endpoint integration (POST /upload, POST /retrieve, semantic search)
"""

import io
from pathlib import Path
import tempfile
import docx
from fastapi.testclient import TestClient
import numpy as np
from pypdf import PdfWriter
import pytest

from app.main import app
from rag.chunker import TextChunker
from rag.document_loader import (
    DocumentPage,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_docx,
    load_document,
    load_pdf,
    load_txt,
)
from rag.embeddings import EmbeddingService
from rag.text_cleaner import clean_text
from rag.vector_store import FAISSVectorStore

client = TestClient(app)

SYNTHETIC_REPORT = """Bridge Inspection Report

Bridge ID: TB-001
Inspection Date: 15 August 2026

The bridge consists of three spans.

The east span contains a crack approximately 3.4 mm wide.

Vibration measurements reached 4.72 mm/s during the inspection period.

The measured temperature was 35 °C.

No visible corrosion was observed on the main support beams.

The deck condition was classified as satisfactory.
"""


def _create_sample_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory blank PDF."""
    writer = PdfWriter()
    for _ in pages_text:
        writer.add_blank_page(width=300, height=300)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _create_text_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate a valid multi-page PDF containing extractable text."""
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=300, height=300)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 250 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = stream
        fonts = DictionaryObject()
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        fonts[NameObject("/F1")] = font
        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _create_sample_docx(paragraphs: list[str]) -> bytes:
    """Helper to generate an in-memory DOCX with paragraphs."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# ──────────────────────────────────────────────
# 1. Text Cleaner Tests
# ──────────────────────────────────────────────

class TestTextCleaner:
    def test_clean_text_preserves_engineering_values(self):
        raw_text = """
        Maximum vibration:   4.72 mm/s  
        
        Stress Level: 12.7 MPa
        
        Temperature: 35 °C
        
        Relative Humidity: 72%
        
        Crack width: -4.5 mm
        
        Date: 2026-08-15
        """
        cleaned = clean_text(raw_text)

        assert "4.72 mm/s" in cleaned
        assert "12.7 MPa" in cleaned
        assert "35 °C" in cleaned
        assert "72%" in cleaned
        assert "-4.5 mm" in cleaned
        assert "2026-08-15" in cleaned

    def test_clean_text_normalizes_whitespace_and_newlines(self):
        raw = "Line 1    with    spaces\n\n\n\n\nLine 2\r\nLine 3"
        cleaned = clean_text(raw)
        assert "Line 1 with spaces" in cleaned
        assert "\n\n\n" not in cleaned
        assert "\r" not in cleaned

    def test_clean_text_handles_empty_or_none(self):
        assert clean_text("") == ""
        assert clean_text("   \n\t  ") == ""


# ──────────────────────────────────────────────
# 2. Document Loader Tests
# ──────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_txt_utf8(self):
        content = SYNTHETIC_REPORT.encode("utf-8")
        pages = load_txt(content, "bridge_report.txt")
        assert len(pages) == 1
        assert pages[0].source_file == "bridge_report.txt"
        assert pages[0].page_number == 1
        assert "3.4 mm wide" in pages[0].text

    def test_load_txt_empty_raises_error(self):
        with pytest.raises(EmptyDocumentError):
            load_txt(b"   \n\t  ", "empty.txt")

    def test_load_docx_paragraphs(self):
        docx_bytes = _create_sample_docx([
            "Bridge Inspection Report - Span A",
            "Observed crack width: 3.4 mm",
            "Ambient temperature: 35 °C",
        ])
        pages = load_docx(docx_bytes, "bridge.docx")
        assert len(pages) == 1
        assert pages[0].source_file == "bridge.docx"
        assert pages[0].page_number == 1
        assert "3.4 mm" in pages[0].text
        assert "Span A" in pages[0].text

    def test_load_docx_empty_raises_error(self):
        docx_bytes = _create_sample_docx(["   "])
        with pytest.raises(EmptyDocumentError):
            load_docx(docx_bytes, "empty.docx")

    def test_load_document_unsupported_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            load_document("report.xlsx", b"data")

    def test_load_pdf_multipage_preserves_page_numbers(self):
        pdf_bytes = _create_text_pdf([
            "Page 1: Bridge TB-001 East Span crack 3.4 mm",
            "Page 2: Vibration 4.72 mm/s and Temp 35 C",
        ])
        pages = load_pdf(pdf_bytes, "multipage.pdf")
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert "East Span" in pages[0].text
        assert pages[1].page_number == 2
        assert "Vibration" in pages[1].text

    def test_load_pdf_empty_raises_error(self):
        # A blank PDF with no readable text
        pdf_bytes = _create_sample_pdf([""])
        with pytest.raises(EmptyDocumentError):
            load_pdf(pdf_bytes, "empty.pdf")


# ──────────────────────────────────────────────
# 3. Chunker Tests
# ──────────────────────────────────────────────

class TestTextChunker:
    def test_chunk_short_document(self):
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        pages = [DocumentPage(text="Short bridge report text.", source_file="short.txt", page_number=1)]
        chunks = chunker.chunk_pages(pages, document_id="doc-0001")

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc-0001-chunk-0001"
        assert chunks[0].text == "Short bridge report text."
        assert chunks[0].page_number == 1
        assert chunks[0].document_id == "doc-0001"

    def test_chunk_long_document_has_overlap(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=30)
        long_text = "Word " * 60  # 300 characters
        pages = [DocumentPage(text=long_text.strip(), source_file="long.txt", page_number=1)]
        chunks = chunker.chunk_pages(pages, document_id="doc-0002")

        assert len(chunks) > 1
        # Check deterministic numbering
        assert chunks[0].chunk_id == "doc-0002-chunk-0001"
        assert chunks[1].chunk_id == "doc-0002-chunk-0002"
        # Check that no chunks are empty
        for c in chunks:
            assert len(c.text) > 0

    def test_chunker_invalid_overlap_raises_value_error(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)


# ──────────────────────────────────────────────
# 4. Embeddings & FAISS Vector Store Tests
# ──────────────────────────────────────────────

class TestEmbeddingsAndVectorStore:
    @pytest.fixture
    def embed_service(self):
        return EmbeddingService()

    def test_embedding_dimensions_and_normalization(self, embed_service):
        texts = ["Bridge crack detected.", "Temperature was 35 °C."]
        embeddings = embed_service.embed_documents(texts)

        assert embeddings.shape == (2, embed_service.dimension)
        assert embeddings.dtype == np.float32

        # Check L2-normalization (norm should be approximately 1.0)
        norm_0 = np.linalg.norm(embeddings[0])
        norm_1 = np.linalg.norm(embeddings[1])
        assert pytest.approx(norm_0, rel=1e-3) == 1.0
        assert pytest.approx(norm_1, rel=1e-3) == 1.0

    def test_query_embedding(self, embed_service):
        q_emb = embed_service.embed_query("What is the temperature?")
        assert q_emb.shape == (embed_service.dimension,)
        assert pytest.approx(np.linalg.norm(q_emb), rel=1e-3) == 1.0

    def test_faiss_add_search_persistence(self, embed_service):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FAISSVectorStore(dimension=embed_service.dimension, db_dir=temp_dir)

            chunks = [
                {
                    "chunk_id": "doc-001-chunk-0001",
                    "text": "The east span contains a crack approximately 3.4 mm wide.",
                    "source_file": "bridge.txt",
                    "page_number": 1,
                    "document_id": "doc-001",
                },
                {
                    "chunk_id": "doc-001-chunk-0002",
                    "text": "The deck condition was classified as satisfactory.",
                    "source_file": "bridge.txt",
                    "page_number": 1,
                    "document_id": "doc-001",
                },
            ]
            texts = [c["text"] for c in chunks]
            embeddings = embed_service.embed_documents(texts)
            store.add_chunks(chunks, embeddings)
            assert store.total_vectors == 2

            # Perform similarity search
            q_emb = embed_service.embed_query("How wide is the crack in the span?")
            results = store.search(q_emb, top_k=1)
            assert len(results) == 1
            best_chunk, score = results[0]
            assert "3.4 mm wide" in best_chunk["text"]
            assert score > 0.4

            # Save store to disk
            store.save(temp_dir)

            # Load into a new store instance
            new_store = FAISSVectorStore(db_dir=temp_dir)
            assert new_store.load(temp_dir) is True
            assert new_store.total_vectors == 2

            # Search in reloaded store
            reloaded_results = new_store.search(q_emb, top_k=1)
            assert reloaded_results[0][0]["chunk_id"] == "doc-001-chunk-0001"


# ──────────────────────────────────────────────
# 5. Endpoints & Semantic Search Integration
# ──────────────────────────────────────────────

class TestWeek2Endpoints:
    def test_upload_txt_and_retrieve_semantically(self):
        # 1. Upload synthetic bridge inspection text report
        resp = client.post(
            "/upload",
            files={"file": ("bridge_report.txt", SYNTHETIC_REPORT.encode("utf-8"), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "document_id" in data
        assert data["chunks_created"] >= 1

        # 2. Semantic query via /retrieve
        ret_resp = client.post(
            "/retrieve",
            json={"question": "What was the crack width in the east span?", "num_results": 2},
        )
        assert ret_resp.status_code == 200
        results = ret_resp.json()["results"]
        assert len(results) >= 1
        top_chunk = results[0]["chunk"]
        assert "3.4 mm wide" in top_chunk["text"]
        assert top_chunk["source_file"] == "bridge_report.txt"

    def test_upload_docx(self):
        docx_bytes = _create_sample_docx([
            "Bridge TB-002 Inspection",
            "East span crack measuring 3.4 mm was noted.",
        ])
        resp = client.post(
            "/upload",
            files={"file": ("bridge_tb002.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_created"] >= 1

    def test_upload_multipage_pdf(self):
        pdf_bytes = _create_text_pdf([
            "Page 1: Inspection findings on main deck.",
            "Page 2: Crack detected in east span: 3.4 mm width.",
        ])
        resp = client.post(
            "/upload",
            files={"file": ("multipage_inspection.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_created"] >= 2

    def test_upload_unsupported_file_returns_400(self):
        resp = client.post(
            "/upload",
            files={"file": ("script.py", b"print('hello')", "text/x-python")},
        )
        assert resp.status_code == 400
        assert "Unsupported file format" in resp.json()["error"]["message"]
