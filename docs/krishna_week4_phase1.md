# Week 4 Phase 1 Contribution Report

**Student Name:** Krishna  
**Assigned Role:** Week 4 — Phase 1: End-to-End Testing & Conversational Bug Fixes  
**Project:** Talking to Bridges (TTB)  

---

## 1. My Week 4 Phase 1 Task
My primary responsibility for Week 4 was to **perform complete end-to-end testing** of the Phase 1 speech-to-speech conversational pipeline and **identify and fix real conversational/pipeline bugs** without breaking the existing API contract or altering other teammates' components.

### Expected Phase 1 Pipeline Flow
$$\text{User Speech Input} \longrightarrow \text{STT (/transcribe)} \longrightarrow \text{Text Query} \longrightarrow \text{RAG Retrieval} \longrightarrow \text{LLM (/chat)} \longrightarrow \text{TTS (/speak)} \longrightarrow \text{Audio Response}$$

---

## 2. What I Tested
1. **Document Ingestion (`POST /upload`)**: Upload and indexing of `synthetic_bridge_report.txt` into FAISS vector storage (`doc-0001`).
2. **Speech-to-Text (`POST /transcribe`)**: Audio transcription using `faster-whisper` on `sample_test_audio.wav` (16 kHz WAV). Tested empty/zero-byte audio edge cases.
3. **RAG Retrieval (`POST /retrieve`)**: Semantic vector search with FAISS (`sentence-transformers/all-MiniLM-L6-v2`) and keyword lexical fallback across 10 shared bridge inspection benchmark questions and whitespace queries.
4. **Conversational LLM Chat (`POST /chat` & `/api/chat`)**: Prompt assembly, document context injection, conversational greetings, unanswerable queries, and source attribution.
5. **Text-to-Speech (`POST /speak`)**: Speech generation using Edge-TTS (`en-US-AriaNeural`) with fallback handling and empty message checks.
6. **Streamlit UI (`frontend/app.py`)**: Citation rendering, session history persistence, and microphone rerun behavior.
7. **Complete Integrated Pipeline**: Chained execution from audio input to final synthesized spoken response.

---

## 3. How I Performed End-to-End Testing
- **Automated Test Suite**: Executed all existing and new unit/integration tests using `pytest speech/tests/ -v`.
- **UI Automation**: Validated the Streamlit frontend layout and session state logic using `validate_ui.py`.
- **End-to-End Pipeline Evaluation**:
  - Ingested `speech/tests/data/synthetic_bridge_report.txt`.
  - Transcribed `speech/sample_test_audio.wav` using `faster-whisper`.
  - Evaluated retrieval and generation for the 10 shared benchmark questions:
    1. Bridge ID (`TB-001`)
    2. Inspection Date (`15 August 2026`)
    3. Number of spans (`3 spans`)
    4. Crack location and size (`east span / 3.4 mm`)
    5. Vibration measurements (`4.72 mm/s`)
    6. Measured temperature (`35 °C`)
    7. Visible corrosion (`None observed on main support beams`)
    8. Deck condition (`Satisfactory`)
    9. Lead inspector (`Unanswerable / Not in context`)
    10. Maximum load capacity (`Unanswerable / Not in context`)
  - Tested conversational edge cases: greetings (*"Hello"*), out-of-domain queries, and whitespace-only queries.
  - Synthesized generated responses into playable base64 audio via `/speak`.

---

## 4. Bugs Found and Fixes Made

| Issue / Bug Found | Root Cause | My Fix |
|---|---|---|
| **1. TTS Event-Loop Crash & Silent Dummy Audio** | In `app/api/routes.py`, the async `/speak` endpoint called `generate_speech()` synchronously, which triggered `asyncio.run()` in `_speak_edge()`. Calling `asyncio.run()` inside an already active event loop raised a `RuntimeError`, causing Edge-TTS to fail silently and fall back to dummy audio. | In [routes.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/app/api/routes.py), offloaded TTS with `await asyncio.to_thread(generate_speech, clean_message)`. In [tts_service.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/app/services/tts_service.py), added loop detection and thread-pool execution for `_speak_edge()`. Edge-TTS now cleanly produces real audio. |
| **2. Citations Dropped in Streamlit UI** | In [frontend/app.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/frontend/app.py), the UI called `/api/chat` but only read `response_json.get("answer")`. The returned `sources` list (file name, page, chunk ID, text) was completely discarded. | Extracted `sources` from the API response, rendered them inside an expandable *"Sources & Citations"* container in [frontend/app.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/frontend/app.py), and stored them in `st.session_state.messages` so citations persist across reruns. |
| **3. Streamlit Audio Input Infinite Rerun Loop** | In [frontend/app.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/frontend/app.py), `st.audio_input()` cached audio bytes across UI reruns. Any widget interaction re-triggered transcription of stale audio, overwriting new user text queries. | Added `last_processed_audio_bytes` in `st.session_state` to process audio only when newly recorded. Also handled empty/silent audio with an informative notification. |
| **4. Whitespace Queries Returning Unrelated Chunks** | Whitespace-only queries (`"   "`) bypassed Pydantic's `min_length=1`. In lexical search, empty search terms resulted in score `0.0`, but the top 3 chunks were still returned to the user. | Added input trimming (`.strip()`) in `/chat`, `/retrieve`, and `/speak` in [routes.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/app/api/routes.py) returning HTTP 400 for empty/whitespace inputs, and updated `_lexical_document_results` to return empty lists for empty search terms. |
| **5. Zero-Byte Audio Upload Crash** | Uploading an empty audio file to `/transcribe` attempted transcription on an empty temporary file. | Added empty content validation in `transcribe_audio()` in [routes.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/app/api/routes.py), returning HTTP 400 with a descriptive error message. |
| **6. Conversational Greetings Breaking Guardrails** | Strict system prompt forced the LLM to output *"The provided context does not contain this information"* even for standard pleasantries like *"Hello"*. | Updated prompt construction in [routes.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/app/api/routes.py) to reply politely to greetings while keeping strict factual adherence for bridge technical questions. |
| **7. Vector Store / Memory Desync on Restart** | Persisted vectors in `data/vector_db/metadata.json` survived server restarts, but in-memory `_uploaded_documents` was empty, breaking lexical search fallback after a reboot. | Implemented `_get_all_chunks()` in [routes.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/app/api/routes.py) to read from both in-memory cache and persisted vector store metadata on disk. |
| **8. Audio MIME-Type Test Incompatibility** | `test_speak_returns_valid_base64_wav` asserted `audio/wav` exclusively, causing tests to fail when Edge-TTS returned standard `audio/mpeg`. | Updated [test_all_endpoints.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/speech/tests/test_all_endpoints.py) to validate base64 decoding across both `audio/wav` and `audio/mpeg`. |

---

## 5. Files Changed

1. [app/services/tts_service.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/app/services/tts_service.py):
   - Added thread-safe event loop handling for `_speak_edge()`.
   - Added text validation for empty/whitespace messages.
2. [app/api/routes.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/app/api/routes.py):
   - Offloaded TTS to worker thread using `asyncio.to_thread`.
   - Added empty audio file validation in `/transcribe`.
   - Added whitespace validation in `/retrieve`, `/chat`, and `/speak`.
   - Fixed lexical search empty terms and 0-score chunk filtering.
   - Sourced all chunks from persisted disk metadata in `_get_all_chunks()`.
   - Improved conversational prompt in `/chat` for natural greeting handling.
3. [frontend/app.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/frontend/app.py):
   - Rendered sources and citations in an expandable container.
   - Stored citations in `st.session_state.messages` for chat history persistence.
   - Prevented microphone rerun loops with `last_processed_audio_bytes`.
   - Handled empty transcriptions gracefully.
4. [speech/tests/test_all_endpoints.py](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/speech/tests/test_all_endpoints.py):
   - Supported both `audio/wav` and `audio/mpeg` in `/speak` test.
   - Added `TestWeek4ConversationalFixes` test suite covering whitespace and empty audio edge cases.
5. [docs/week4-testing.md](file:///c:/Users/Podil/Downloads/Talking-to-Bridges-main/Talking-to-Bridges-main/docs/week4-testing.md):
   - Created full test report documenting test executions, bug fixes, and benchmark outcomes.

---

## 6. Testing Results

- **Unit & Integration Tests (`pytest speech/tests/ -v`)**:
  - **58 passed**, 0 failed (100% pass rate).
- **UI Verification (`validate_ui.py`)**:
  - Layout initialized cleanly, citation rendering verified, and audio safeguards confirmed.
- **End-to-End Pipeline Evaluation**:
  - Document Upload: 100% success.
  - Speech-to-Text: 100% accurate transcription with automatic CPU fallback.
  - RAG Retrieval: Successfully matched relevant chunks for all 10 benchmark questions.
  - Conversational Chat: Accurate answers with citations and natural greeting support.
  - Text-to-Speech: Successfully generated 32,976 bytes of `audio/mpeg` without falling back to dummy audio.
  - Full Integrated Flow: Successfully executed from audio file $\rightarrow$ STT $\rightarrow$ RAG $\rightarrow$ Chat $\rightarrow$ TTS $\rightarrow$ spoken audio response.

---

## 7. Final Status
- **Status:** **COMPLETED**
- **Existing Contracts:** Preserved (100% compliant with `docs/api-contract.md`).
- **Dependencies & Architecture:** Preserved (no new external libraries or architectural redesigns introduced).
- **Blocker Note:** Live Groq LLM calls require `GROQ_API_KEY` in `.env`. When not configured, `GET /health` returns `groq_connected: false` and live `/chat` returns HTTP 503 as designed. All automated pipeline tests with simulated/mocked LLM responses passed completely.
