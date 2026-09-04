"""Admin API routes (ARCHITECTURE.md §9): conversation history + knowledge reload.

Admin routes are localhost-only or behind a simple API key; they are never
exposed publicly (§14). Requests must carry the configured key in the
X-API-Key header. When no key is configured the endpoints fail closed.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.models.schemas import (
    ConversationDetail,
    ConversationSummary,
    KnowledgeReloadResponse,
)

admin_router = APIRouter(prefix="/admin")


def require_admin_api_key(request: Request) -> None:
    api_key = settings.admin_api_key
    if not api_key:
        raise HTTPException(status_code=503, detail="Admin API key not configured")
    provided = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(api_key.encode("utf-8"), provided.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")


def _conversation_store(request: Request):
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Conversation store not available")
    return store


def _build_rag_service():
    from app.rag.embeddings import SentenceTransformerEmbeddings
    from app.rag.service import RAGService

    return RAGService(
        raw_dir=settings.rag_raw_dir,
        processed_dir=settings.rag_processed_dir,
        embedding_provider=SentenceTransformerEmbeddings(settings.embedding_model),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


@admin_router.get(
    "/conversations",
    response_model=list[ConversationSummary],
    dependencies=[Depends(require_admin_api_key)],
)
async def list_conversations(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    store = _conversation_store(request)
    conversations = store.list_conversations(limit=limit, offset=offset)
    return [ConversationSummary.model_validate(c) for c in conversations]


@admin_router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    dependencies=[Depends(require_admin_api_key)],
)
async def get_conversation(conversation_id: str, request: Request):
    store = _conversation_store(request)
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail.model_validate(conversation)


@admin_router.post(
    "/knowledge/reload",
    response_model=KnowledgeReloadResponse,
    dependencies=[Depends(require_admin_api_key)],
)
async def reload_knowledge():
    service = _build_rag_service()
    try:
        summary = await run_in_threadpool(service.index_documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge reload failed: {e}") from e
    return KnowledgeReloadResponse(status="reloaded", **summary)
