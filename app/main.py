from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router as api_router, vector_store
from app.core.config import settings
from app.core.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Talking to Bridges API Server")
    logger.info(f"Groq Target URL: {settings.GROQ_BASE_URL}")
    logger.info(f"Groq Target Model: {settings.GROQ_MODEL}")
    # Load persistent vector database if available
    loaded = vector_store.load()
    if loaded:
        logger.info(f"Vector store loaded with {vector_store.total_vectors} chunks")
    else:
        logger.info("Vector store initialized (empty)")
    yield
    logger.info("Shutting down Talking to Bridges API Server")


app = FastAPI(
    title="Talking to Bridges API",
    description="An LLM-Based Intelligent Interface for Structural Health Monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": str(exc.detail), "type": "HTTPException"}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": {"message": "Invalid request parameters", "details": exc.errors(), "type": "ValidationError"}},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled application error")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "InternalError"}},
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )
