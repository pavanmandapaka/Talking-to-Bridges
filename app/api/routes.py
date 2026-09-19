import base64
import io
import re
import wave
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.logging_config import logger
from rag.ingestion import DocumentHandlingError, extract_document_chunks
from app.services.llm_service import (
    GroqLLMService,
    GroqModelNotFoundError,
    GroqServiceError,
    GroqUnavailableError,
)

router = APIRouter()
llm_service = GroqLLMService()
_uploaded_documents: dict[str, list["ChunkResponse"]] = {}


class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, description="User query or prompt message"
    )


class ChunkResponse(BaseModel):
    text: str
    source_file: str
    page_number: int
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChunkResponse] = Field(default_factory=list)


class UploadResponse(BaseModel):
    document_id: str
    chunks_created: int


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    num_results: int = Field(default=3, ge=1, le=20)


class RetrievalResult(BaseModel):
    chunk: ChunkResponse
    score: float


class RetrieveResponse(BaseModel):
    results: list[RetrievalResult]


class TranscribeResponse(BaseModel):
    text: str


class SpeakResponse(BaseModel):
    audio_base64: str
    content_type: str = "audio/wav"


class HealthResponse(BaseModel):
    status: str = "ok"
    groq_connected: bool = False


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint to verify backend status and Groq availability."""
    groq_status = await llm_service.health_check()
    return HealthResponse(status="ok", groq_connected=groq_status)


@router.post("/upload", response_model=UploadResponse)
async def upload(file: Annotated[UploadFile, File(...)]):
    """Extract and store a document for later chat and retrieval requests."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A file name is required",
        )

    document_id = f"doc-{len(_uploaded_documents) + 1:04d}"
    try:
        content = await file.read()
        extracted_chunks = extract_document_chunks(file.filename, content)
    except (DocumentHandlingError, ValueError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    # The UI has one active upload; prevent earlier sessions from contaminating
    # questions about the newly selected document.
    _uploaded_documents.clear()
    chunks = [
        ChunkResponse(
            text=text,
            source_file=file.filename,
            page_number=page_number,
            chunk_id=f"{document_id}-chunk-{index:04d}",
        )
        for index, (text, page_number) in enumerate(extracted_chunks, start=1)
    ]
    _uploaded_documents[document_id] = chunks
    return UploadResponse(document_id=document_id, chunks_created=len(chunks))


def _document_results(question: str, limit: int = 3) -> list[RetrievalResult]:
    """Select relevant chunks, keeping all chunks available for summaries."""
    chunks = [chunk for document in _uploaded_documents.values() for chunk in document]
    if not chunks:
        return []

    terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored = []
    for chunk in chunks:
        chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
        score = len(terms & chunk_terms) / max(len(terms), 1)
        scored.append(RetrievalResult(chunk=chunk, score=score))
    scored.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
    if any(word in terms for word in {"summarize", "summary", "overview", "describe"}):
        return scored
    return scored[:limit]


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Return document chunks ranked by simple lexical relevance."""
    return RetrieveResponse(results=_document_results(request.question, request.num_results))


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: Annotated[UploadFile, File(...)]):
    """Return placeholder speech text until Whisper is integrated."""
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An audio file name is required",
        )
    return TranscribeResponse(text="This is dummy transcribed speech.")


def _dummy_wav_base64() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@router.post("/speak", response_model=SpeakResponse)
async def speak(request: ChatRequest):
    """Return valid silent WAV audio until Piper TTS is integrated."""
    return SpeakResponse(audio_base64=_dummy_wav_base64())


@router.post("/chat", response_model=ChatResponse)
@router.post("/api/chat", response_model=ChatResponse, include_in_schema=False)
async def chat(request: ChatRequest):
    """Processes user query and generates an answer from Groq."""
    logger.info("Received request on POST /api/chat")
    try:
        document_results = _document_results(request.message)
        if document_results:
            context = "\n\n".join(
                f"[Source: {result.chunk.source_file}, page {result.chunk.page_number}]\n"
                f"{result.chunk.text}"
                for result in document_results
            )
            prompt = (
                "Answer the user's question using only the document context below. "
                "For a summary, cover the main points from all provided context. "
                "If the answer is not in the context, say so clearly.\n\n"
                f"Document context:\n{context}\n\nUser question: {request.message}"
            )
        else:
            prompt = request.message
        answer = await llm_service.generate(prompt)
        return ChatResponse(
            answer=answer,
            sources=[result.chunk for result in document_results],
        )
    except GroqUnavailableError:
        logger.error("Chat endpoint error: Groq service is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq service is unavailable",
        )
    except GroqModelNotFoundError as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except GroqServiceError as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal service error communicating with LLM",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in chat endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
