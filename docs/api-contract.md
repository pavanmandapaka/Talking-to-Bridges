# Week 1 API Contract

This is the shared Week 1 boundary. Real implementations replace the dummy
behavior behind the same inputs and outputs in later weeks.

The `/chat` endpoint uses the Groq API. Configure `GROQ_API_KEY` in `.env`;
the key is sent only by the backend and is never required in the Streamlit UI.

`GET /health` returns `groq_connected` to show whether the backend can reach
the configured Groq API.

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /upload` | Multipart file | `document_id`, `chunks_created` |
| `POST /retrieve` | `question`, `num_results` | Chunks with `score` |
| `POST /chat` | `message` | `answer`, `sources` |
| `POST /transcribe` | Multipart audio file | Spoken `text` |
| `POST /speak` | `message` | Base64 `audio_base64`, `content_type` |

Every chunk contains `text`, `source_file`, `page_number`, and `chunk_id`.
The existing `/api/chat` path remains available as a compatibility alias.