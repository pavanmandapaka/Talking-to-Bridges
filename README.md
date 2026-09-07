# Talking to Bridges 🌉🤖

An LLM-Based Intelligent Interface for Structural Health Monitoring (SHM).

## Overview
**Talking to Bridges** is a zero-cost academic and industry-oriented project designed to provide an interactive interface for monitoring bridge health and querying sensor data & maintenance documentation.

---

## Phase 1 / Week 1 Scope
Current implementation focuses on establishing a clean, modular backend and local LLM foundation:
- **FastAPI REST API**: Core application server with health and chat endpoints.
- **Local LLM Integration**: Ollama integration (`http://localhost:11434`) using standard local models (e.g., `llama3.2` or `llama3`).
- **Configuration & Logging**: Environment variable management (`.env`) and application logging.
- **Graceful Error Handling**: Clean HTTP error status codes (503 when Ollama is offline) without dumping raw stack traces.
- **Unit Testing**: Pytest suite with mocked LLM service integration.

*Note: Document RAG ingestion, vector databases, and speech audio layers will be implemented in subsequent phases.*

---

## Prerequisites
- **Python**: 3.10 or higher (Python 3.11/3.12 recommended)
- **Ollama**: Installed and running locally ([https://ollama.com](https://ollama.com))

---

## Installation & Setup

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-org/talking-to-bridges.git
cd talking-to-bridges

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration values in `.env`:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
APP_HOST=127.0.0.1
APP_PORT=8000
```

---

## Running Ollama Locally

1. Install Ollama from [ollama.com](https://ollama.com).
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull your target model (e.g., `llama3.2` or `llama3`):
   ```bash
   ollama pull llama3.2
   ```

---

## Running the Application

Start the FastAPI application server:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## API Documentation & Examples

### Health Check Endpoint
```bash
curl -X GET http://127.0.0.1:8000/health
```
**Example Response:**
```json
{
  "status": "ok",
  "ollama_connected": true
}
```

### Chat Endpoint
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is structural health monitoring?"}'
```
**Example Response:**
```json
{
  "answer": "Structural Health Monitoring (SHM) refers to the process of implementing a damage detection and characterization strategy for engineering structures such as bridges and buildings."
}
```

If Ollama is not running:
```json
{
  "detail": "Local LLM service is unavailable"
}
```

---

## Running Tests
Run unit tests with `pytest`:
```bash
pytest tests/
```

---

## Project Structure
```
talking-to-bridges/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # REST API endpoints (/health, /api/chat)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Environment configuration
│   │   └── logging_config.py# Centralized logging setup
│   └── services/
│       ├── __init__.py
│       └── llm_service.py   # Ollama LLM HTTP client abstraction
│
├── data/
│   ├── documents/           # Raw PDF/Docx files (future phase)
│   ├── processed/           # Processed chunks (future phase)
│   └── vector_db/           # Vector index storage (future phase)
│
├── rag/                     # RAG pipeline modules (future phase)
│   └── __init__.py
├── analysis/                # Data analysis modules (future phase)
│   └── __init__.py
├── models/                  # Pydantic / Data models
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_health.py       # Health check endpoint tests
│   └── test_chat.py         # Chat endpoint and error handling tests
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Troubleshooting
1. **`503 Service Unavailable` on `/api/chat`**:
   - Ensure Ollama is running (`ollama serve`).
   - Test `curl http://localhost:11434/api/tags` in terminal.
2. **`404 Not Found` model error**:
   - Verify the model name in `.env` matches your pulled model (`ollama list`).
   - Run `ollama pull llama3.2` to download the model.
