from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Talking to Bridges API Server")
    logger.info(f"Ollama Target URL: {settings.OLLAMA_BASE_URL}")
    logger.info(f"Ollama Target Model: {settings.OLLAMA_MODEL}")
    yield
    logger.info("Shutting down Talking to Bridges API Server")


app = FastAPI(
    title="Talking to Bridges API",
    description="An LLM-Based Intelligent Interface for Structural Health Monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )
