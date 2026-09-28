import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import asyncio
from dataclasses import dataclass

from app.services.llm_service import GroqLLMService, GroqServiceError


def ask_llm(prompt: str) -> str:
    """Helper function for Groq inference."""
    service = GroqLLMService()
    return asyncio.run(service.generate(prompt))


@dataclass(frozen=True)
class ModelProbeResult:
    """Result of the Week 1 local-model smoke test."""

    model: str
    available: bool
    response: str = ""
    error: str = ""


async def probe_installed_models(
    service: GroqLLMService | None = None,
    prompt: str = "In one sentence, what is structural health monitoring?",
) -> list[ModelProbeResult]:
    """Discover installed models and test each with a small prompt."""
    groq = service or GroqLLMService()
    model_names = await groq.list_models()
    results: list[ModelProbeResult] = []

    for model_name in model_names:
        model_service = GroqLLMService(
            base_url=groq.base_url,
            api_key=groq.api_key,
            model=model_name,
            timeout=groq.timeout,
        )
        try:
            response = await model_service.generate(prompt)
            results.append(ModelProbeResult(model_name, True, response=response))
        except Exception as error:  # noqa: BLE001
            results.append(ModelProbeResult(model_name, False, error=str(error)))
    return results


if __name__ == "__main__":
    try:
        probe_results = asyncio.run(probe_installed_models())
    except GroqServiceError as error:
        print(f"FAIL Groq: {error}")
        raise SystemExit(1) from error

    for result in probe_results:
        status = "PASS" if result.available else "FAIL"
        detail = result.response or result.error
        print(f"{status} {result.model}: {detail}")