"""Conversation record persisted for every farmer interaction (ARCHITECTURE.md §8.1)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Mirrors the `conversations` table schema."""

    farmer_phone: str
    message_in: str | None = None
    media_type: str | None = None
    media_url: str | None = None
    vision_result: str | None = None
    rag_sources: str | None = None
    response_out: str | None = None
    confidence_flag: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str | None = None
    responded_at: str | None = None
