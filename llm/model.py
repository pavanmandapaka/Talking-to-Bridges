import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import asyncio
from app.services.llm_service import OllamaLLMService


def ask_llm(prompt: str) -> str:
    """Helper function for local LLM inference via Ollama."""
    service = OllamaLLMService()
    return asyncio.run(service.generate(prompt))


if __name__ == "__main__":
    answer = ask_llm("Explain structural health monitoring in simple terms.")
    print(answer)