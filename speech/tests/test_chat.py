from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.llm_service import GroqUnavailableError

client = TestClient(app)


def test_chat_endpoint_success():
    with patch(
        "app.api.routes.llm_service.generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = (
            "Structural Health Monitoring involves sensing and analyzing structural integrity."
        )
        response = client.post(
            "/api/chat",
            json={"message": "What is structural health monitoring?"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "answer": "Structural Health Monitoring involves sensing and analyzing structural integrity.",
            "sources": []
        }
        mock_generate.assert_awaited_once_with(
            "What is structural health monitoring?"
        )


def test_chat_endpoint_validation_error():
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_endpoint_groq_unavailable():
    with patch(
        "app.api.routes.llm_service.generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = GroqUnavailableError(
            "Groq service is unavailable"
        )
        response = client.post(
            "/api/chat",
            json={"message": "Hello?"},
        )

        assert response.status_code == 503
        assert response.json()["error"]["message"] == "Groq service is unavailable"
