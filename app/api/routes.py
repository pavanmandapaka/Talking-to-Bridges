import asyncio
import base64
import io
import re
import uuid
import wave
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status, Form
import pandas as pd
from app.services.visualization_service import analyze_and_plot
from app.services.tts_service import generate_speech
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging_config import logger
from rag.document_loader import (
    DocumentExtractionError,
    DocumentHandlingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from rag.embeddings import EmbeddingError, EmbeddingService
from rag.ingestion import DOCUMENTS_DIR, ingest_csv, ingest_document
from rag.csv_processor import CorruptedCSVError, EmptyCSVError
from rag.retrieval import RetrievedChunk, VectorRetriever
from rag.vector_store import FAISSVectorStore, VectorStoreError
from app.services.llm_service import (
    GroqLLMService,
    GroqModelNotFoundError,
    GroqServiceError,
    GroqUnavailableError,
)

router = APIRouter()
llm_service = GroqLLMService()
embedding_service = EmbeddingService()
vector_store = FAISSVectorStore()
retriever = VectorRetriever(embedding_service=embedding_service, vector_store=vector_store)

_uploaded_documents: dict[str, list["ChunkResponse"]] = {}


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, description="User query or prompt message"
    )
    chat_history: list[Message] = Field(default_factory=list)


class ChunkResponse(BaseModel):
    text: str
    source_file: str
    page_number: int
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChunkResponse] = Field(default_factory=list)
    fig: str | None = None  # Plotly figure JSON when a chart_data tool result exists


class UploadResponse(BaseModel):
    document_id: str
    chunks_created: int
    # Optional fields populated for CSV uploads; None for PDF/DOCX/TXT
    file_type: str | None = None
    rows: int | None = None
    columns: int | None = None
    column_names: list[str] | None = None


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
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A file name is required",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".csv"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Supported formats are: .pdf, .docx, .txt, .csv",
        )

    document_id = f"doc-{uuid.uuid4().hex[:8]}"
    try:
        content = await file.read()
        
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
            
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        import os
        safe_filename = os.path.basename(file.filename)
        file_path = DOCUMENTS_DIR / f"{document_id}_{safe_filename}"
        
        temp_vector_store = FAISSVectorStore()
        
        if suffix == ".csv":
            import asyncio
            ingest_result = await asyncio.to_thread(
                ingest_csv,
                file.filename,
                content,
                document_id,
                embedding_service,
                temp_vector_store,
            )
        else:
            import asyncio
            ingest_result = await asyncio.to_thread(
                ingest_document,
                file.filename,
                content,
                document_id,
                embedding_service,
                temp_vector_store,
            )
            
        global vector_store
        vector_store.index = temp_vector_store.index
        vector_store.dimension = temp_vector_store.dimension
        vector_store.metadata = temp_vector_store.metadata

    except (UnsupportedFileTypeError, EmptyDocumentError, DocumentExtractionError, DocumentHandlingError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except (EmptyCSVError, CorruptedCSVError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except (EmbeddingError, VectorStoreError) as error:
        raise HTTPException(status_code=500, detail=str(error))

    # Keep the raw upload so /chat and the analytical tools can load it as a DataFrame
    file_path.write_bytes(content)

    _uploaded_documents.clear()
    chunks = [
        ChunkResponse(
            text=chunk.text,
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
        )
        for chunk in ingest_result.chunks
    ]
    _uploaded_documents[document_id] = chunks

    return UploadResponse(
        document_id=document_id,
        chunks_created=ingest_result.chunks_created,
        file_type=suffix.lstrip("."),
        rows=ingest_result.rows if hasattr(ingest_result, 'rows') else None,
        columns=ingest_result.columns if hasattr(ingest_result, 'columns') else None,
        column_names=ingest_result.column_names if hasattr(ingest_result, 'column_names') else None,
    )

def _get_all_chunks() -> list[ChunkResponse]:
    """Retrieve all available document chunks from active uploads or loaded vector store metadata."""
    if _uploaded_documents:
        return [chunk for document in _uploaded_documents.values() for chunk in document]
    if vector_store and vector_store.metadata:
        return [
            ChunkResponse(
                text=m.get("text", ""),
                source_file=m.get("source_file", ""),
                page_number=m.get("page_number", 1),
                chunk_id=m.get("chunk_id", ""),
            )
            for m in vector_store.metadata
        ]
    return []


def _lexical_document_results(question: str, limit: int = 3) -> list[RetrievalResult]:
    """Fallback lexical search across in-memory or persisted document chunks."""
    chunks = _get_all_chunks()
    if not chunks:
        return []

    clean_question = question.strip() if question else ""
    if not clean_question:
        return []

    terms = set(re.findall(r"[a-z0-9]+", clean_question.lower()))
    if not terms:
        return []

    scored = []
    for chunk in chunks:
        chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
        matched = terms & chunk_terms
        if matched:
            score = len(matched) / max(len(terms), 1)
            scored.append(RetrievalResult(chunk=chunk, score=score))
    scored.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
    if any(word in terms for word in {"summarize", "summary", "overview", "describe"}):
        return scored
    return scored[:limit]


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant document chunks using FAISS semantic vector search."""
    clean_question = request.question.strip()
    if not clean_question:
        return RetrieveResponse(results=[])

    try:
        retrieved_chunks = retriever.retrieve(clean_question, top_k=request.num_results)
        if retrieved_chunks:
            results = [
                RetrievalResult(
                    chunk=ChunkResponse(
                        text=r.text,
                        source_file=r.source_file,
                        page_number=r.page_number,
                        chunk_id=r.chunk_id,
                    ),
                    score=r.score,
                )
                for r in retrieved_chunks
            ]
            return RetrieveResponse(results=results)
    except Exception as e:
        logger.warning(f"Vector search failed, falling back to lexical search: {e}")

    # Fallback to lexical results if vector search yielded no results
    return RetrieveResponse(results=_lexical_document_results(clean_question, request.num_results))


import tempfile
import os
from app.services.stt_service import transcribe as run_stt

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: Annotated[UploadFile, File(...)]):
    """Transcribes audio using Eswar's GPU-accelerated faster-whisper model."""
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An audio file name is required",
        )
        
    try:
        content = await audio.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty.",
            )

        # Create a temporary file to save the incoming audio bytes
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)
            
        # Run Eswar's fast transcription
        try:
            import asyncio
            transcribed_text = await asyncio.to_thread(run_stt, temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return TranscribeResponse(text=transcribed_text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")


def _dummy_wav_base64() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


from app.services.tts_service import generate_speech

@router.post("/speak", response_model=SpeakResponse)
async def speak(request: ChatRequest):
    """Converts text to speech using the active TTS engine (ElevenLabs, Edge, or Piper)."""
    clean_message = request.message.strip()
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )
    try:
        audio_b64, content_type = await asyncio.to_thread(generate_speech, clean_message)
        return SpeakResponse(audio_base64=audio_b64, content_type=content_type)
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        # Fallback to silent wav if everything breaks
        return SpeakResponse(audio_base64=_dummy_wav_base64(), content_type="audio/wav")


@router.post("/chat", response_model=ChatResponse)
@router.post("/api/chat", response_model=ChatResponse, include_in_schema=False)
async def chat(request: ChatRequest):
    """Main conversational endpoint.

    Flow:
    1. Resolve active DataFrame (from last uploaded file, if any).
    2. Run the ToolDispatcher to detect analytical intent and execute the tool.
    3. Run FAISS + lexical RAG retrieval in parallel.
    4. Merge tool result + retrieved chunks into the LLM context.
    5. Generate and return the LLM response.

    RAG and Analytics are complementary: both can contribute context to the
    same LLM call.  If no analytical intent is found, only RAG context is used.
    If no RAG results exist, only the analytical result is used.
    """
    clean_message = request.message.strip()
    if not clean_message:
        raise HTTPException(status_code=400, detail="Message empty")

    try:
        # ------------------------------------------------------------------
        # 1. Resolve active DataFrame from the last uploaded CSV/document
        # ------------------------------------------------------------------
        from analysis.tools import data_access
        from analysis.dispatcher import dispatcher

        active_df = None
        if _uploaded_documents:
            doc_id = list(_uploaded_documents.keys())[0]
            matching_files = list(DOCUMENTS_DIR.glob(f"{doc_id}_*"))
            if matching_files:
                try:
                    active_df = await asyncio.to_thread(
                        data_access.load_dataset, matching_files[0]
                    )
                except Exception as load_err:
                    logger.debug("Could not load active dataset: %s", load_err)

        # ------------------------------------------------------------------
        # 2. Analytical tool dispatch (ToolDispatcher)
        # ------------------------------------------------------------------
        dispatch_result = None
        try:
            dispatch_result = await asyncio.to_thread(
                dispatcher.detect_and_dispatch, clean_message, active_df
            )
        except Exception as disp_err:
            logger.warning("Dispatcher error (non-fatal): %s", disp_err)

        # ------------------------------------------------------------------
        # 3. Vector + Lexical RAG retrieval
        # ------------------------------------------------------------------
        retrieved_chunks = []
        try:
            semantic_results = retriever.retrieve(clean_message, top_k=1)
            if semantic_results:
                retrieved_chunks = [
                    RetrievalResult(
                        chunk=ChunkResponse(
                            text=r.text,
                            source_file=r.source_file,
                            page_number=r.page_number,
                            chunk_id=r.chunk_id,
                        ),
                        score=r.score,
                    )
                    for r in semantic_results
                ]
        except Exception:
            pass

        if not retrieved_chunks:
            retrieved_chunks = _lexical_document_results(clean_message, limit=3)

        # ------------------------------------------------------------------
        # 4. Merge contexts for the LLM prompt
        # ------------------------------------------------------------------
        context_blocks = []

        if retrieved_chunks:
            context_blocks.append(
                "\n\n".join(
                    f"[Source: {r.chunk.source_file}]\n{r.chunk.text}"
                    for r in retrieved_chunks
                )
            )

        if dispatch_result and dispatch_result.status == "success":
            context_blocks.append(
                f"[Analytical Tool: {dispatch_result.tool_name}]\n"
                f"{dispatch_result.llm_summary}"
            )
        elif dispatch_result and dispatch_result.status == "error":
            # Still surface the error to the LLM so it can explain to the user
            context_blocks.append(
                f"[Analytical Tool Error: {dispatch_result.tool_name}]\n"
                f"{dispatch_result.message}"
            )

        if context_blocks:
            context = "\n\n".join(context_blocks)
            from rag.prompts import SYSTEM_PROMPT, USER_PROMPT
            system_content = SYSTEM_PROMPT.format(context=context)
            user_content = USER_PROMPT.format(query=clean_message)
        else:
            system_content = (
                "You are a helpful assistant for the Talking to Bridges platform. "
                "The user has not uploaded any document or dataset yet. "
                "If they ask about sensor data or a document, invite them to upload one."
            )
            user_content = clean_message

        # ------------------------------------------------------------------
        # 5. LLM generation
        # ------------------------------------------------------------------
        messages = [{"role": "system", "content": system_content}]
        for msg in request.chat_history[-5:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_content})

        answer = await llm_service.generate(messages)
        chart_json = None
        if (
            dispatch_result
            and dispatch_result.status == "success"
            and dispatch_result.tool_name == "chart_data"
        ):
            candidate = dispatch_result.data.get("plot_json") if dispatch_result.data else None
            if candidate and candidate != "{}":
                chart_json = candidate
        return ChatResponse(
            answer=answer,
            sources=[r.chunk for r in retrieved_chunks],
            fig=chart_json,
        )

    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(e))

class ToolExecuteRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the analytical tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Structured arguments for the tool")
    document_id: str | None = Field(default=None, description="Optional uploaded document_id to run tool against")


class ToolExecuteResponse(BaseModel):
    tool: str
    status: str
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    message: str | None = None
    error_type: str | None = None
    details: dict[str, Any] | None = None


@router.get("/api/tools")
async def list_analytical_tools():
    """List available analytical tools registered in the platform."""
    from analysis.tools import registry
    return {"tools": registry.list_tools()}


@router.post("/api/tools/execute", response_model=ToolExecuteResponse)
async def execute_analytical_tool(request: ToolExecuteRequest):
    """Execute a registered analytical tool with structured arguments and dataset context."""
    from analysis.tools import registry, data_access
    
    df = None
    if request.document_id:
        # Search for document file in DOCUMENTS_DIR matching document_id
        matching_files = list(DOCUMENTS_DIR.glob(f"{request.document_id}_*"))
        if matching_files:
            try:
                df = data_access.load_dataset(matching_files[0])
            except Exception as e:
                logger.warning(f"Could not load dataset for '{request.document_id}': {e}")
                
    result = registry.execute(request.tool_name, request.arguments, df=df)
    return ToolExecuteResponse(**result)


@router.post("/analyze_csv")
async def analyze_csv(
    file: UploadFile = File(...),
    time_col: str = Form(...),
    val_col: str = Form(...)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), comment="#")
        
        from analysis.tools import registry
        tool_result = registry.execute("chart_data", {"x_col": time_col, "y_col": val_col}, df=df)
        
        if tool_result["status"] == "error":
            raise HTTPException(status_code=400, detail=tool_result["message"])
            
        chart_data = tool_result["data"]
        explanation = chart_data.get("explanation", "")
        fig_json = chart_data.get("plot_json", "{}")

        # Calculate stats for speech
        valid_data = df.dropna(subset=[val_col]) if val_col in df.columns else pd.DataFrame()
        if not valid_data.empty:
            max_val = pd.to_numeric(valid_data[val_col], errors="coerce").max()
            min_val = pd.to_numeric(valid_data[val_col], errors="coerce").min()
            avg_val = pd.to_numeric(valid_data[val_col], errors="coerce").mean()
            spoken_text = (
                f"Here is the chart for {val_col} over {time_col}. "
                f"The maximum value is {max_val:.2f}, "
                f"the minimum value is {min_val:.2f}, "
                f"and the average value is {avg_val:.2f}."
            )
        else:
            spoken_text = "I couldn't find any valid numerical data to analyze."
            
        # Generate TTS audio
        audio_base64_str, content_type = await asyncio.to_thread(generate_speech, spoken_text)
        
        return {
            "fig_json": fig_json,
            "explanation": explanation,
            "audio_base64": audio_base64_str,
            "content_type": content_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyze CSV error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

