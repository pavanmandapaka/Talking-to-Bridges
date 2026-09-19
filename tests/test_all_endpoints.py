"""Comprehensive endpoint validation tests.

Validates every endpoint in docs/api-contract.md against its expected
input/output schema and the global error format.
"""

import base64
import io
import wave
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.llm_service import (
    GroqModelNotFoundError,
    GroqServiceError,
    GroqUnavailableError,
)

client = TestClient(app)


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_ok_when_groq_up(self):
        with patch(
            "app.api.routes.llm_service.health_check", new_callable=AsyncMock
        ) as mock:
            mock.return_value = True
            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["groq_connected"] is True

    def test_health_returns_ok_when_groq_down(self):
        with patch(
            "app.api.routes.llm_service.health_check", new_callable=AsyncMock
        ) as mock:
            mock.return_value = False
            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["groq_connected"] is False


# ──────────────────────────────────────────────
# POST /upload
# ──────────────────────────────────────────────


class TestUploadEndpoint:
    def test_upload_returns_document_id_and_chunks(self):
        resp = client.post(
            "/upload",
            files={"file": ("test_report.pdf", b"fake-pdf-content", "application/pdf")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "document_id" in body
        assert isinstance(body["chunks_created"], int)
        assert body["chunks_created"] >= 1

    def test_upload_without_file_returns_422(self):
        resp = client.post("/upload")
        assert resp.status_code == 422


# ──────────────────────────────────────────────
# POST /retrieve
# ──────────────────────────────────────────────


class TestRetrieveEndpoint:
    def test_retrieve_returns_results_list(self):
        # Upload first so there's data to retrieve
        client.post(
            "/upload",
            files={"file": ("bridge_data.pdf", b"data", "application/pdf")},
        )
        resp = client.post(
            "/retrieve",
            json={"question": "What is the bridge condition?", "num_results": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert isinstance(body["results"], list)
        # Each result should have chunk and score
        for result in body["results"]:
            assert "chunk" in result
            assert "score" in result
            chunk = result["chunk"]
            assert "text" in chunk
            assert "source_file" in chunk
            assert "page_number" in chunk
            assert "chunk_id" in chunk

    def test_retrieve_empty_question_returns_422(self):
        resp = client.post("/retrieve", json={"question": ""})
        assert resp.status_code == 422

    def test_retrieve_invalid_num_results_returns_422(self):
        resp = client.post(
            "/retrieve", json={"question": "test", "num_results": 0}
        )
        assert resp.status_code == 422


# ──────────────────────────────────────────────
# POST /chat  and  POST /api/chat
# ──────────────────────────────────────────────


class TestChatEndpoint:
    def _mock_generate(self):
        return patch(
            "app.api.routes.llm_service.generate", new_callable=AsyncMock
        )

    def test_chat_success_has_answer_and_sources(self):
        with self._mock_generate() as mock:
            mock.return_value = "Test answer"
            resp = client.post("/chat", json={"message": "Hello"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Test answer"
        assert "sources" in body
        assert isinstance(body["sources"], list)

    def test_chat_uses_uploaded_document_context(self):
        client.post(
            "/upload",
            files={
                "file": (
                    "bridge.txt",
                    b"The bridge inspection found a 4 millimeter crack in the east span.",
                    "text/plain",
                )
            },
        )
        with self._mock_generate() as mock:
            mock.return_value = "The east span has a 4 millimeter crack."
            resp = client.post("/chat", json={"message": "Summarize the document"})

        assert resp.status_code == 200
        assert resp.json()["sources"][0]["source_file"] == "bridge.txt"
        prompt = mock.await_args.args[0]
        assert "4 millimeter crack" in prompt

    def test_api_chat_alias_works(self):
        """The /api/chat path must remain available as a compatibility alias."""
        with self._mock_generate() as mock:
            mock.return_value = "Alias answer"
            resp = client.post("/api/chat", json={"message": "Hi"})

        assert resp.status_code == 200
        assert resp.json()["answer"] == "Alias answer"

    def test_chat_empty_message_returns_422(self):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_chat_missing_message_returns_422(self):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_groq_unavailable_returns_503(self):
        with self._mock_generate() as mock:
            mock.side_effect = GroqUnavailableError("down")
            resp = client.post("/chat", json={"message": "test"})

        assert resp.status_code == 503

    def test_chat_model_not_found_returns_404(self):
        with self._mock_generate() as mock:
            mock.side_effect = GroqModelNotFoundError("missing model")
            resp = client.post("/chat", json={"message": "test"})

        assert resp.status_code == 404

    def test_chat_service_error_returns_500(self):
        with self._mock_generate() as mock:
            mock.side_effect = GroqServiceError("timeout")
            resp = client.post("/chat", json={"message": "test"})

        assert resp.status_code == 500

    def test_chat_unhandled_error_returns_500(self):
        with self._mock_generate() as mock:
            mock.side_effect = RuntimeError("unexpected")
            resp = client.post("/chat", json={"message": "test"})

        assert resp.status_code == 500


# ──────────────────────────────────────────────
# POST /transcribe
# ──────────────────────────────────────────────


class TestTranscribeEndpoint:
    def test_transcribe_returns_text(self):
        resp = client.post(
            "/transcribe",
            files={"audio": ("speech.wav", b"fake-audio", "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        assert isinstance(body["text"], str)
        assert len(body["text"]) > 0

    def test_transcribe_without_file_returns_422(self):
        resp = client.post("/transcribe")
        assert resp.status_code == 422


# ──────────────────────────────────────────────
# POST /speak
# ──────────────────────────────────────────────


class TestSpeakEndpoint:
    def test_speak_returns_valid_base64_wav(self):
        resp = client.post("/speak", json={"message": "Hello bridge"})
        assert resp.status_code == 200
        body = resp.json()
        assert "audio_base64" in body
        assert body["content_type"] == "audio/wav"
        # Verify it's valid base64 that decodes to a valid WAV
        audio_bytes = base64.b64decode(body["audio_base64"])
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    def test_speak_empty_message_returns_422(self):
        resp = client.post("/speak", json={"message": ""})
        assert resp.status_code == 422


# ──────────────────────────────────────────────
# Global error format consistency
# ──────────────────────────────────────────────


class TestErrorFormat:
    """Every error response must use the fixed format:
    {"error": {"message": "...", "type": "..."}}
    """

    def test_http_exception_uses_error_format(self):
        with patch(
            "app.api.routes.llm_service.generate", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = GroqUnavailableError("down")
            resp = client.post("/chat", json={"message": "test"})

        body = resp.json()
        assert "error" in body
        assert "message" in body["error"]
        assert "type" in body["error"]

    def test_validation_error_uses_error_format(self):
        resp = client.post("/chat", json={"message": ""})
        body = resp.json()
        assert "error" in body
        assert "message" in body["error"]
        assert body["error"]["type"] == "ValidationError"

    def test_404_nonexistent_path_uses_error_format(self):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body
        assert "message" in body["error"]
        assert "type" in body["error"]
