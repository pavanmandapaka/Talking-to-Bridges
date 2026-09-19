from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_success():
    with patch(
        "app.api.routes.llm_service.health_check", new_callable=AsyncMock
    ) as mock_health:
        mock_health.return_value = True
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "groq_connected": True}


def test_health_endpoint_groq_offline():
    with patch(
        "app.api.routes.llm_service.health_check", new_callable=AsyncMock
    ) as mock_health:
        mock_health.return_value = False
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "groq_connected": False}
