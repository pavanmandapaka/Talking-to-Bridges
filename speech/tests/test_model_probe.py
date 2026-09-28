import asyncio
from unittest.mock import AsyncMock

from app.services.llm_service import GroqLLMService, GroqUnavailableError
from llm.model import probe_installed_models


def test_probe_installed_models_tests_each_installed_model(monkeypatch):
    service = GroqLLMService(model="configured-model", api_key="test-key")
    monkeypatch.setattr(
        service,
        "list_models",
        AsyncMock(return_value=["openai/gpt-oss-20b", "openai/gpt-oss-120b"]),
    )

    async def fake_generate(self, prompt):
        return f"response from {self.model}"

    monkeypatch.setattr(GroqLLMService, "generate", fake_generate)

    results = asyncio.run(probe_installed_models(service))

    assert [(result.model, result.available) for result in results] == [
        ("openai/gpt-oss-20b", True),
        ("openai/gpt-oss-120b", True),
    ]
    assert results[0].response == "response from openai/gpt-oss-20b"


def test_probe_installed_models_records_model_failure(monkeypatch):
    service = GroqLLMService(model="configured-model", api_key="test-key")
    monkeypatch.setattr(service, "list_models", AsyncMock(return_value=["broken-model"]))
    monkeypatch.setattr(
        GroqLLMService,
        "generate",
        AsyncMock(side_effect=GroqUnavailableError("offline")),
    )

    results = asyncio.run(probe_installed_models(service))

    assert results == [
        results[0].__class__(
            model="broken-model",
            available=False,
            error="offline",
        )
    ]