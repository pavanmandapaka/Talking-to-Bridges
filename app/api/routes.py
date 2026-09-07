from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.logging_config import logger
from app.services.llm_service import (
    OllamaLLMService,
    OllamaModelNotFoundError,
    OllamaServiceError,
    OllamaUnavailableError,
)

router = APIRouter()
llm_service = OllamaLLMService()


class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, description="User query or prompt message"
    )


class ChatResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str = "ok"
    ollama_connected: bool = False


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint to verify backend status and Ollama availability."""
    ollama_status = await llm_service.health_check()
    return HealthResponse(status="ok", ollama_connected=ollama_status)


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Processes user query and generates answer from local Ollama LLM."""
    logger.info("Received request on POST /api/chat")
    try:
        answer = await llm_service.generate(request.message)
        return ChatResponse(answer=answer)
    except OllamaUnavailableError:
        logger.error("Chat endpoint error: Local LLM service is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local LLM service is unavailable",
        )
    except OllamaModelNotFoundError as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except OllamaServiceError as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal service error communicating with LLM",
        )
    except Exception:
        logger.exception("Unhandled error in chat endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
