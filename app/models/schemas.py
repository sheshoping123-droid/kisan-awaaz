"""API response schemas for admin endpoints (ARCHITECTURE.md §9, §10)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    farmer_phone: str
    message_in: str | None = None
    media_type: str | None = None
    created_at: str | None = None
    responded_at: str | None = None


class ConversationDetail(ConversationSummary):
    media_url: str | None = None
    vision_result: str | None = None
    rag_sources: str | None = None
    response_out: str | None = None
    confidence_flag: str | None = None


class KnowledgeReloadResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    dimension: int
    index_path: str
