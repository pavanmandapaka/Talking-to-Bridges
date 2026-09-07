import httpx
from app.core.config import settings
from app.core.logging_config import logger


class OllamaServiceError(Exception):
    """Base exception for Ollama LLM service failures."""

    pass


class OllamaUnavailableError(OllamaServiceError):
    """Raised when Ollama service is offline or unreachable."""

    pass


class OllamaModelNotFoundError(OllamaServiceError):
    """Raised when the specified model is not found in Ollama."""

    pass


class OllamaLLMService:
    """Service abstraction for communicating with local Ollama HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout

    async def health_check(self) -> bool:
        """Checks whether the Ollama server is reachable and responsive."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def generate(self, prompt: str) -> str:
        """Sends prompt to local Ollama model and returns text output.

        Args:
            prompt: User message string.

        Returns:
            Generated text string from the model.

        Raises:
            OllamaUnavailableError: If Ollama server is unreachable.
            OllamaModelNotFoundError: If requested model is not found.
            OllamaServiceError: For timeouts or other API errors.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 404:
                    logger.error(f"Ollama model '{self.model}' not found.")
                    raise OllamaModelNotFoundError(
                        f"Model '{self.model}' not found in local Ollama instance."
                    )

                if response.status_code != 200:
                    logger.error(
                        f"Ollama returned status {response.status_code}: {response.text}"
                    )
                    raise OllamaServiceError(
                        f"Ollama error status: {response.status_code}"
                    )

                data = response.json()
                return data.get("response", "")

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            logger.error(f"Connection failure to Ollama at {self.base_url}: {e}")
            raise OllamaUnavailableError("Local LLM service is unavailable")
        except httpx.TimeoutException as e:
            logger.error(f"Ollama generation request timed out: {e}")
            raise OllamaServiceError("Local LLM request timed out")
        except httpx.RequestError as e:
            logger.error(f"Ollama network request error: {e}")
            raise OllamaServiceError("Failed to communicate with local LLM service")
