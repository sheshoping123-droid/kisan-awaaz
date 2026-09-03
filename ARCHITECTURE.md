# Kisan Awaaz — MVP Architecture

> WhatsApp-based AI agricultural advisory for smallholder farmers in Pakistan.

---

## 1. System Architecture

```
┌─────────────┐
│   Farmer    │
│  (WhatsApp) │
└──────┬──────┘
       │  photo / voice note / text
       ▼
┌──────────────────┐       ┌───────────────────┐
│  WhatsApp Bridge │──────▶│   Message Router   │
│  (Twilio/Meta)   │◀──────│   (FastAPI)        │
└──────────────────┘       └──────┬────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
             ┌──────────┐ ┌───────────┐  ┌──────────┐
             │  Voice   │ │  Vision   │  │  Text    │
             │ Pipeline │ │ Pipeline  │  │ Pipeline │
             └────┬─────┘ └─────┬─────┘  └────┬─────┘
                  │             │              │
                  ▼             ▼              ▼
             ┌────────────────────────────────────┐
             │         RAG + Reasoning Engine      │
             │  (Qwen LLM + FAISS knowledge base)  │
             └────────────────┬───────────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │   Response   │
                     │  Formatter   │
                     │ (Urdu text)  │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   WhatsApp   │
                     │   Reply      │
                     └──────────────┘
```

**Principle:** every external service (WhatsApp provider, speech model, vision model, LLM) sits behind an interface. The core router calls interfaces, never concrete clients. This lets you swap providers without touching business logic.

---

## 2. Recommended Technology Stack

| Layer | MVP Choice | Why |
|---|---|---|
| Language | Python 3.11+ | You know it; entire AI ecosystem is Python-native |
| Web framework | **FastAPI** | Async, simple, auto-docs, easy to debug |
| WhatsApp | **Twilio WhatsApp Sandbox** | Free sandbox, instant setup, no business verification needed for MVP |
| Vision model | **Qwen2-VL** via DashScope API or HuggingFace Inference | You know Qwen; API avoids GPU hosting cost |
| Speech-to-text | **OpenAI Whisper API** or `faster-whisper` locally | Urdu support is solid; simple single call |
| Text-to-speech (later) | **gTTS** (free) → **ElevenLabs** (quality) | gTTS has Urdu; good enough for demo |
| LLM | **Qwen2.5** via Together AI / DashScope API | Low cost, good multilingual, you know the family |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual (covers Urdu), small, fast |
| Vector store | **FAISS** (local file) | You know it; zero infra for MVP |
| Database | **SQLite** via SQLAlchemy | Zero setup; upgrade to Postgres later |
| Task queue | None in MVP | Process requests synchronously; add Celery only if load demands it |
| Deployment | **Railway** or **Render** | One-click deploy, free/cheap tier, good for demos |
| RAG framework | **LangChain** | You know it; use for retrieval + prompt assembly only |

---

## 3. Backend Architecture

```
app/
  main.py              ← FastAPI app, webhook endpoints
  router.py            ← MessageRouter: dispatches by media type
  
  adapters/            ← Thin wrappers around external services (swappable)
    whatsapp/
      base.py          ← abstract WhatsAppAdapter
      twilio_adapter.py
    vision/
      base.py          ← abstract VisionAdapter
      qwen_adapter.py
    speech/
      base.py          ← abstract SpeechAdapter
      whisper_adapter.py
    llm/
      base.py          ← abstract LLMAdapter
      qwen_adapter.py
  
  pipelines/           ← Orchestration logic
    image_pipeline.py
    voice_pipeline.py
    text_pipeline.py
  
  rag/                 ← Knowledge base
    embedder.py
    retriever.py
    knowledge_loader.py
  
  models/              ← SQLAlchemy models + Pydantic schemas
  core/                ← Config, logging, error handling
```

**Key design decisions:**

- **Synchronous request/response.** WhatsApp allows up to 15 seconds for webhook response. The MVP stays within this by keeping the pipeline simple. If it gets tight, return 202 immediately and send the reply via the WhatsApp API asynchronously.
- **No task queue.** Adding Celery/Redis for a solo MVP is over-engineering. If you need async later, FastAPI's `BackgroundTasks` is sufficient for fire-and-forget.
- **One conversation table.** Track `farmer_phone`, `message_in`, `media_type`, `response_out`, `timestamps`, and `confidence`. Simple and auditable.

---

## 4. AI/ML Pipeline

### 4.1 Vision Pipeline

```
Image received
    │
    ▼
Resize / normalize (max 1024px, JPEG)
    │
    ▼
Qwen2-VL: "Describe the crop in this image. Identify any visible
           disease symptoms, pest damage, nutrient deficiency, or stress.
           Respond in structured JSON: {crop, condition, symptoms[], confidence}"
    │
    ▼
Parse structured output
    │
    ▼
Pass {crop, condition, symptoms} to RAG retriever as query context
```

**Guardrails:**
- Prompt the vision model to output a confidence level (high / medium / low).
- If confidence is low, the final response explicitly says: *"We are not fully certain. Please visit your local agricultural extension office."*
- The vision model **never** recommends treatments. It only identifies symptoms. Treatment recommendations come from the RAG knowledge base only.

### 4.2 Text / Query Pipeline

```
Farmer text (Urdu or Roman Urdu)
    │
    ▼
Language detection (simple heuristic or `langdetect`)
    │
    ▼
Combine with vision output (if image was also sent)
    │
    ▼
RAG retrieval → top-K relevant passages
    │
    ▼
Qwen2.5 LLM with system prompt:
  - "You are Kisan Awaaz, an agricultural advisor."
  - "Answer ONLY using the retrieved knowledge passages."
  - "If the knowledge base does not contain relevant information, say so clearly."
  - "Respond in simple Urdu."
  - "Never invent chemical names, dosages, or treatment schedules."
  - "If uncertain, recommend visiting the local agricultural extension office."
    │
    ▼
Response in Urdu text
```

### 4.3 Uncertainty & Safety Contract

Every response must pass through a **ResponseValidator** before being sent:

1. Does the response reference at least one RAG source? If not → append uncertainty disclaimer.
2. Does the response contain specific chemical dosages not present in the knowledge base? If yes → strip and replace with general guidance + referral.
3. Is the vision confidence low? If yes → prepend uncertainty framing.

This is the single most important design decision for a competition demo. Judges will test edge cases.

---

## 5. RAG Architecture

### 5.1 Knowledge Base Sources

For MVP, curate 50–200 high-quality documents:

- **FAO crop disease guides** (freely available, structured)
- **Pakistan Agricultural Research Council** publications
- **Crop-specific fact sheets** for wheat, rice, cotton, sugarcane, maize (Pakistan's major crops)
- **Pest and disease management guides** from CABI or similar

### 5.2 Ingestion Pipeline (run once, offline)

```
Source documents (PDF, markdown, text)
    │
    ▼
Chunking: 512 tokens, 50-token overlap
(Language-aware — don't split mid-Urdu sentence)
    │
    ▼
Embed: paraphrase-multilingual-MiniLM-L12-v2
    │
    ▼
Store in FAISS index (flat L2, saved as .faiss file)
    │
    ▼
Save metadata store (chunk_id → source, page, crop, section)
```

### 5.3 Retrieval Pipeline (per request)

```
Query (text from farmer + vision symptoms)
    │
    ▼
Embed query with same model
    │
    ▼
FAISS similarity search → top 5 chunks
    │
    ▼
Re-rank by metadata relevance (crop match, symptom match)
    │
    ▼
Pass top 3 chunks as context to LLM
```

### 5.4 Why FAISS and not a hosted vector DB

For a competition MVP with <200 documents and one developer:
- FAISS is a single file, zero infrastructure.
- You already know it.
- Search latency at this scale is <10ms.
- Migration path to Pinecone/Weaviate/Qdrant is trivial — change the `retriever.py` adapter.

---

## 6. Voice Pipeline

### 6.1 MVP (Speech-to-Text only)

```
WhatsApp voice note (.ogg)
    │
    ▼
Download media via Twilio API
    │
    ▼
Convert .ogg → .wav (pydub / ffmpeg)
    │
    ▼
Whisper API (or faster-whisper locally)
    language=ur, task=transcribe
    │
    ▼
Urdu text → feed into Text Pipeline
```

### 6.2 Phase 2 (Text-to-Speech reply)

```
Urdu response text
    │
    ▼
gTTS (lang='ur') → .mp3
    │
    ▼
Send as WhatsApp audio message
```

**MVP recommendation:** skip TTS for the competition demo. Text reply is sufficient. Judges will understand the architecture from your slides. TTS adds complexity and failure modes with minimal demo value.

---

## 7. WhatsApp Integration Architecture

### 7.1 Twilio Sandbox (recommended for MVP)

```
Twilio WhatsApp Sandbox
    │
    │  POST /webhook/whatsapp (on incoming message)
    ▼
FastAPI endpoint
    │
    ├── If message has image → download via Twilio media URL → image pipeline
    ├── If message has audio → download via Twilio media URL → voice pipeline
    ├── If message is text   → text pipeline
    │
    ▼
Pipeline returns Urdu response text
    │
    ▼
Twilio API: send message back to farmer's WhatsApp number
```

### 7.2 Webhook Handler Contract

```python
# Pseudocode — not implementation
POST /webhook/whatsapp
  → validate Twilio signature (security)
  → extract: From, Body, NumMedia, MediaUrl0
  → persist conversation record
  → route to appropriate pipeline
  → send reply via Twilio client
  → update conversation record with response
  → return 200
```

### 7.3 Why Twilio Sandbox over Meta WhatsApp Business API

| Factor | Twilio Sandbox | Meta API |
|---|---|---|
| Setup time | 5 minutes | 1–2 weeks (business verification) |
| Cost | Free tier (limited messages) | Per-conversation pricing |
| Demo quality | Identical user experience | Identical |
| Production path | Upgrade to Twilio paid | Direct |

For a competition: Twilio Sandbox wins easily.

---

## 8. Database Requirements

### 8.1 MVP: SQLite with SQLAlchemy

**Tables:**

```sql
-- Every interaction, fully auditable
conversations (
    id              TEXT PRIMARY KEY,  -- UUID
    farmer_phone    TEXT NOT NULL,
    message_in      TEXT,              -- transcribed / original text
    media_type      TEXT,              -- 'text' | 'image' | 'voice'
    media_url       TEXT,              -- original media URL from Twilio
    vision_result   TEXT,              -- JSON: crop, symptoms, confidence
    rag_sources     TEXT,              -- JSON: list of source references
    response_out    TEXT,              -- final Urdu response
    confidence_flag TEXT,              -- 'high' | 'medium' | 'low'
    created_at      DATETIME,
    responded_at    DATETIME
)

-- Pre-loaded, not queried often
knowledge_chunks (
    id              TEXT PRIMARY KEY,
    source_title    TEXT,
    source_url      TEXT,
    crop            TEXT,
    section         TEXT,
    content         TEXT,
    embedding_id    INTEGER            -- FAISS index position
)
```

### 8.2 Why not PostgreSQL for MVP

SQLite is a single file, no server, zero config. For a solo developer building a demo, this removes an entire category of problems. SQLAlchemy makes the migration to Postgres a one-line config change later.

---

## 9. API Structure

```
POST   /webhook/whatsapp          ← Twilio incoming webhook
GET    /health                    ← health check (deployment, monitoring)
GET    /admin/conversations       ← list recent conversations (debugging)
GET    /admin/conversations/{id}  ← single conversation detail
POST   /admin/knowledge/reload    ← re-ingest knowledge base
```

**That's it.** No REST CRUD, no auth tokens on admin routes for MVP (admin routes are not exposed publicly — they're localhost-only or behind a simple API key).

---

## 10. Project Folder Structure

```
kisan-awaaz/
├── app/
│   ├── main.py                  # FastAPI app, webhook endpoint
│   ├── router.py                # MessageRouter
│   ├── config.py                # Pydantic settings, env loading
│   │
│   ├── adapters/                # Swappable external service wrappers
│   │   ├── __init__.py
│   │   ├── whatsapp/
│   │   │   ├── base.py
│   │   │   └── twilio_adapter.py
│   │   ├── vision/
│   │   │   ├── base.py
│   │   │   └── qwen_adapter.py
│   │   ├── speech/
│   │   │   ├── base.py
│   │   │   └── whisper_adapter.py
│   │   └── llm/
│   │       ├── base.py
│   │       └── qwen_adapter.py
│   │
│   ├── pipelines/               # Business logic orchestration
│   │   ├── image_pipeline.py
│   │   ├── voice_pipeline.py
│   │   ├── text_pipeline.py
│   │   └── response_validator.py
│   │
│   ├── rag/
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── knowledge_loader.py
│   │   └── chunker.py
│   │
│   ├── models/
│   │   ├── database.py
│   │   ├── conversation.py
│   │   └── schemas.py
│   │
│   └── core/
│       ├── logging.py
│       └── errors.py
│
├── knowledge/                   # Raw source documents
│   ├── raw/                     # PDFs, markdown, text files
│   └── processed/               # Chunked + embedded artifacts
│       ├── index.faiss
│       └── metadata.json
│
├── scripts/
│   ├── ingest_knowledge.py      # Offline script to build FAISS index
│   └── seed_test_data.py        # Load sample conversations for testing
│
├── tests/
│   ├── test_router.py
│   ├── test_pipelines.py
│   ├── test_rag.py
│   ├── test_response_validator.py
│   └── fixtures/
│       ├── sample_crop_image.jpg
│       ├── sample_voice_note.ogg
│       └── sample_queries.json
│
├── notebooks/
│   └── rag_evaluation.ipynb     # Evaluate retrieval quality
│
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile                     # Common commands: make ingest, make dev, make test
└── README.md
```

---

## 11. Environment Variables

```env
# --- App ---
APP_ENV=development
LOG_LEVEL=INFO

# --- WhatsApp (Twilio) ---
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=+14155238886   # Twilio sandbox number
WEBHOOK_BASE_URL=https://your-app.railway.app  # ngrok URL for local dev

# --- Vision Model ---
VISION_PROVIDER=qwen_api              # qwen_api | qwen_local
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx    # for Qwen VL via DashScope
# or
QWEN_VL_MODEL_PATH=/models/qwen2-vl  # for local Ollama/vLLM

# --- LLM ---
LLM_PROVIDER=qwen_api                # qwen_api | together | ollama
TOGETHER_API_KEY=xxxxxxxxxxxxxxxx
# or
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx
# or
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# --- Speech ---
SPEECH_PROVIDER=whisper_api           # whisper_api | whisper_local
OPENAI_API_KEY=sk-xxxxxxxxxxxx       # for Whisper API

# --- RAG ---
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
FAISS_INDEX_PATH=knowledge/processed/index.faiss
METADATA_PATH=knowledge/processed/metadata.json
RAG_TOP_K=5
RAG_CONTEXT_K=3

# --- Database ---
DATABASE_URL=sqlite:///./data/kisan_awaaz.db

# --- Admin ---
ADMIN_API_KEY=your-simple-admin-key
```

---

## 12. Testing Strategy

### 12.1 Unit Tests (fast, no external calls)

- **ResponseValidator:** given a mock LLM response + mock RAG sources, does it correctly flag uncertainty, strip invented dosages, append disclaimers?
- **Router:** given different media types, does it dispatch to the correct pipeline?
- **Chunker:** does it produce correctly-sized chunks without breaking Urdu text?

### 12.2 Integration Tests (with mocked adapters)

- Replace all adapters with mock implementations that return canned responses.
- Test the full pipeline: incoming webhook → router → pipeline → response.
- Verify conversation records are persisted correctly.

### 12.3 RAG Evaluation (offline, most important)

Build a test set of 30–50 question→expected_source pairs. Run retrieval and measure:
- **Recall@5:** is the correct source in the top 5 results?
- **Precision@3:** are the top 3 results relevant?

This is the single highest-value test activity. If RAG fails, everything downstream fails.

### 12.4 End-to-End Smoke Test (manual)

Send a real WhatsApp message with a crop image. Verify the response arrives, is in Urdu, and references a real knowledge source. Do this before every demo.

### 12.5 What NOT to test in MVP

- Load testing
- Concurrent user simulation
- Adversarial input fuzzing

---

## 13. Deployment Strategy

### 13.1 MVP Deployment

```
Railway (or Render)
├── FastAPI app (single service)
├── SQLite database (file on persistent volume)
├── FAISS index (bundled in deployment or on volume)
└── Environment variables (set in dashboard)
```

**Steps:**
1. Dockerfile with Python 3.11, ffmpeg, dependencies.
2. Push to GitHub → connect Railway to repo.
3. Set env vars in Railway dashboard.
4. Railway gives you a public URL → set as Twilio webhook base URL.
5. Done.

### 13.2 Local Development

```bash
# Terminal 1: run the app
uvicorn app.main:app --reload --port 8000

# Terminal 2: expose via ngrok for Twilio webhook
ngrok http 8000

# Update Twilio sandbox webhook URL to ngrok URL
```

### 13.3 Demo Day Backup Plan

- Record a 2-minute video of a successful interaction.
- If live demo fails, play the video and walk through the code.
- Never rely solely on live network + live AI models for a competition demo.

---

## 14. Security Considerations

| Risk | Mitigation |
|---|---|
| Twilio webhook spoofing | Validate `X-Twilio-Signature` on every request |
| API key leaks | `.env` file, never committed; `.gitignore` includes `.env` |
| Malicious image uploads | Validate file type and size before processing |
| LLM hallucinating treatments | ResponseValidator strips ungrounded medical/chemical claims |
| Admin endpoint abuse | API key auth + don't expose admin routes publicly |
| Farmer PII (phone numbers) | Phone numbers stored but never logged; no other PII collected |
| Prompt injection via voice | Transcribed text goes through same RAG grounding; LLM system prompt ignores instructions embedded in user input |

---

## 15. Development Phases (exact order)

### Phase 1 — RAG Foundation (days 1–3)
**Goal:** a working knowledge base that retrieves relevant agricultural passages.

1. Collect 30+ source documents on Pakistani crops and diseases.
2. Build the chunker and embedder.
3. Build the FAISS index.
4. Build the retriever.
5. Write `rag_evaluation.ipynb` — verify retrieval quality with 20+ test queries.
6. Iterate on chunk size and overlap until Recall@5 > 80%.

**Exit criteria:** you can ask "wheat rust treatment" in English and get the correct FAO passage.

### Phase 2 — LLM + Response Pipeline (days 4–5)
**Goal:** grounded Urdu responses from text queries.

1. Integrate Qwen2.5 via API.
2. Build the text pipeline: query → RAG → LLM → Urdu response.
3. Build the ResponseValidator.
4. Test with 10+ queries; verify no hallucinated treatments.

**Exit criteria:** text query in → grounded Urdu response out, with uncertainty when appropriate.

### Phase 3 — Vision Pipeline (days 6–7)
**Goal:** crop image analysis feeding into the RAG pipeline.

1. Integrate Qwen2-VL via API.
2. Build the image pipeline: image → vision model → symptom extraction.
3. Connect vision output to text pipeline as additional context.
4. Test with 10+ crop disease images.

**Exit criteria:** photo of diseased crop → symptom description → RAG-grounded Urdu advice.

### Phase 4 — WhatsApp Integration (days 8–9)
**Goal:** end-to-end WhatsApp conversation.

1. Set up Twilio WhatsApp Sandbox.
2. Build the webhook endpoint and message router.
3. Build the Twilio adapter.
4. Test text messages end-to-end.
5. Test image messages end-to-end.

**Exit criteria:** send a WhatsApp message, receive a grounded Urdu reply.

### Phase 5 — Voice Pipeline (days 10–11)
**Goal:** voice note → Urdu text → response.

1. Build audio download + format conversion.
2. Integrate Whisper API.
3. Connect transcription output to text pipeline.
4. Test with 5+ Urdu voice notes.

**Exit criteria:** send a voice note in Urdu, receive a text response.

### Phase 6 — Polish + Demo Prep (days 12–14)
**Goal:** competition-ready demonstration.

1. Add admin endpoints for viewing conversation history.
2. Prepare 3–5 demo scenarios (different crops, diseases, voice/text).
3. Record backup demo video.
4. Write presentation slides with architecture diagram.
5. Rehearse the demo flow.

---

## 16. Components to Mock During Early Development

| Component | Mock With | When to Replace |
|---|---|---|
| Twilio WhatsApp | Print response to console | Phase 4 |
| Qwen2-VL (vision) | Hardcoded JSON: `{crop: "wheat", symptoms: ["brown rust spots"], confidence: "high"}` | Phase 3 |
| Whisper (STT) | Hardcoded Urdu text string | Phase 5 |
| Qwen2.5 (LLM) | Canned Urdu response string | Phase 2 |
| FAISS | Hardcoded list of 3 passages | Phase 1 (replace immediately — this IS Phase 1) |

**Why mock aggressively:** you can build and test the router, pipelines, and response validator before any AI API is integrated. This decouples architecture work from model tuning.

---

## 17. MVP Essential vs. Postpone

### Essential for Competition MVP

| Component | Priority |
|---|---|
| RAG knowledge base (30+ documents, Urdu+English) | P0 |
| Qwen LLM integration with grounded prompting | P0 |
| ResponseValidator (uncertainty + hallucination guard) | P0 |
| Qwen-VL image analysis | P0 |
| WhatsApp text + image flow | P0 |
| Whisper voice-to-text | P1 |
| Conversation persistence (SQLite) | P1 |
| Admin conversation viewer | P1 |
| Architecture diagram for slides | P1 |

### Postpone (after competition)

| Component | Why |
|---|---|
| Text-to-speech reply | Demo value is low; architecture is clear without it |
| Punjabi language support | Adds linguistic complexity; Urdu first |
| Multi-turn conversation memory | Single-turn works for demo; memory adds state complexity |
| User authentication / farmer profiles | No security need for demo |
| Analytics dashboard | Post-competition feature |
| SMS fallback (non-WhatsApp) | Out of MVP scope |
| Offline / edge deployment | Competition is live-demo or video |
| GPU self-hosting of models | API is cheaper and simpler for MVP |
| Rate limiting / abuse prevention | Demo scale doesn't need it |
| CI/CD pipeline | Push-to-deploy on Railway is sufficient |

---

## Key Architectural Decisions Summary

1. **Synchronous pipeline.** No task queue. Simple, debuggable, sufficient for demo load.
2. **Adapter pattern for all external services.** Every AI provider is swappable by changing one file.
3. **Vision model identifies symptoms, never recommends treatments.** Treatments come only from the RAG knowledge base. This is the core safety guarantee.
4. **ResponseValidator as a mandatory gate.** Every response is checked before being sent. This is the single feature that wins or loses a competition judge's trust.
5. **SQLite + FAISS as file-based stores.** Zero infrastructure. Bundle in Docker. Migrate later.
6. **Twilio Sandbox for WhatsApp.** 5-minute setup, free, identical user experience to production WhatsApp.
7. **Aggressive mocking in early phases.** Build the skeleton end-to-end before integrating any AI service.
