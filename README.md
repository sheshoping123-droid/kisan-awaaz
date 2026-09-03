# Kisan Awaaz

WhatsApp-based AI agricultural advisory system for smallholder farmers in Pakistan.

## Architecture

```
Farmer (WhatsApp)
       │
       ▼
WhatsApp Bridge (Twilio)
       │
       ▼
Message Router (FastAPI)
       │
       ├── Image Pipeline  → Vision Model (Qwen2-VL)
       ├── Voice Pipeline  → Speech-to-Text (Whisper)
       └── Text Pipeline   → RAG Retrieval → LLM (Qwen2.5)
                                  │
                                  ▼
                          Response Validator
                                  │
                                  ▼
                           Urdu Reply
```

The system identifies crop diseases from photos, transcribes Urdu/Punjabi voice notes, and generates evidence-grounded agricultural advice using a RAG knowledge base. Responses are always in simple Urdu and include uncertainty disclaimers when confidence is low.

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Environment Setup

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Key variables for each phase:

| Phase | Required Variables |
|-------|-------------------|
| Phase 1 (foundation) | `APP_ENV`, `LOG_LEVEL` only |
| Phase 2 (RAG) | `EMBEDDING_MODEL`, `RAG_RAW_DIR`, `RAG_PROCESSED_DIR` |
| Phase 3 (vision) | `VISION_PROVIDER`, `DASHSCOPE_API_KEY` |
| Phase 4 (WhatsApp) | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` |
| Phase 5 (voice) | `SPEECH_PROVIDER`, `OPENAI_API_KEY` |

## Running Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Health check: `http://localhost:8000/health`

API docs: `http://localhost:8000/docs`

## Running Tests

```bash
pytest -v
```

## Project Structure

```
app/
  main.py              FastAPI application entry point
  router.py            Routes messages to pipelines by media type
  adapters/            Swappable wrappers for external services
    whatsapp/          WhatsApp messaging (Twilio)
    vision/            Image analysis (Qwen2-VL)
    speech/            Speech-to-text / text-to-speech (Whisper)
    llm/               Language model (Qwen2.5)
  pipelines/           Orchestration logic per media type
    image_pipeline.py
    voice_pipeline.py
    text_pipeline.py
    response_validator.py   Safety gate for all outgoing responses
  rag/                 Knowledge base retrieval
    models.py          Document, DocumentChunk, RetrievalResult
    loaders.py         Discover and load documents from knowledge/raw/
    chunker.py         Split documents into overlapping chunks
    embeddings.py      EmbeddingProvider interface + sentence-transformers impl
    index.py           FAISS vector index (create, persist, search)
    retriever.py       Query → embed → FAISS search → top-k results
    service.py         Coordinates indexing and retrieval
  models/              Database models (Phase 4)
  core/                Config, logging, error handling
tests/                 pytest test suite
knowledge/
  raw/                 Source knowledge documents (.txt, .md)
  processed/           FAISS index and metadata (build artifact)
scripts/
  build_index.py       CLI script to build FAISS index from documents
```

## RAG Knowledge Base (Phase 2)

### Adding Knowledge Documents

Place `.txt` or `.md` files in `knowledge/raw/`. These should be agricultural knowledge documents about crop diseases, pest management, etc.

### Building the Index

```bash
python scripts/build_index.py
```

This discovers documents, chunks them, generates embeddings, and saves a FAISS index to `knowledge/processed/`.

### Index Storage

The FAISS index and metadata are saved to `knowledge/processed/`:
- `index.faiss` — the vector index
- `metadata.json` — chunk text and metadata mapping

### Retrieval

The `RAGService` class coordinates retrieval. Given a query string, it embeds the query and searches the FAISS index for the top-k most relevant chunks, returning `RetrievalResult` objects with text, scores, and source metadata.

### RAG Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformer model name |
| `RAG_RAW_DIR` | `knowledge/raw` | Directory for source documents |
| `RAG_PROCESSED_DIR` | `knowledge/processed` | Directory for FAISS index |
| `RAG_CHUNK_SIZE` | `512` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Default retrieval count |

## Development Status

**Phase 1**: Backend foundation — project structure, config, logging, error handling, placeholder interfaces.

**Phase 2** (current): RAG foundation — document loading, chunking, embeddings, FAISS indexing, retrieval.

See `ARCHITECTURE.md` for the full system design.
