"""SQLite database access: schema management and conversation persistence.

ARCHITECTURE.md §8.1 specifies SQLite with SQLAlchemy, but SQLAlchemy is not
among the project's declared dependencies, so the stdlib sqlite3 module is
used directly. The schema matches the architecture exactly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.logging import get_logger
from app.models.conversation import Conversation

logger = get_logger(__name__)

SQLITE_URL_PREFIX = "sqlite:///"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    farmer_phone    TEXT NOT NULL,
    message_in      TEXT,
    media_type      TEXT,
    media_url       TEXT,
    vision_result   TEXT,
    rag_sources     TEXT,
    response_out    TEXT,
    confidence_flag TEXT,
    created_at      DATETIME,
    responded_at    DATETIME
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id              TEXT PRIMARY KEY,
    source_title    TEXT,
    source_url      TEXT,
    crop            TEXT,
    section         TEXT,
    content         TEXT,
    embedding_id    INTEGER
);
"""

INSERT_CONVERSATION_SQL = """
INSERT INTO conversations (
    id, farmer_phone, message_in, media_type, media_url,
    vision_result, rag_sources, response_out, confidence_flag,
    created_at, responded_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

LIST_CONVERSATIONS_SQL = """
SELECT id, farmer_phone, message_in, media_type, media_url,
       vision_result, rag_sources, response_out, confidence_flag,
       created_at, responded_at
FROM conversations
ORDER BY created_at DESC
LIMIT ? OFFSET ?
"""

GET_CONVERSATION_SQL = """
SELECT id, farmer_phone, message_in, media_type, media_url,
       vision_result, rag_sources, response_out, confidence_flag,
       created_at, responded_at
FROM conversations
WHERE id = ?
"""


def database_path(database_url: str) -> Path:
    """Extract the filesystem path from a sqlite:/// URL."""
    if not database_url.startswith(SQLITE_URL_PREFIX):
        raise ValueError(f"Unsupported database URL (only SQLite is supported): {database_url}")
    return Path(database_url[len(SQLITE_URL_PREFIX):])


class ConversationStore:
    """Persists Conversation records to SQLite.

    Opens a short-lived connection per operation; each write commits and
    closes immediately, so no connection state is shared across requests.
    """

    def __init__(self, database_url: str):
        self._path = database_path(database_url)

    def init_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_SQL)
        finally:
            conn.close()
        logger.info("Database schema initialised at %s", self._path)

    def save(self, conversation: Conversation) -> None:
        conn = self._connect()
        try:
            conn.execute(
                INSERT_CONVERSATION_SQL,
                (
                    conversation.id,
                    conversation.farmer_phone,
                    conversation.message_in,
                    conversation.media_type,
                    conversation.media_url,
                    conversation.vision_result,
                    conversation.rag_sources,
                    conversation.response_out,
                    conversation.confidence_flag,
                    conversation.created_at,
                    conversation.responded_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[Conversation]:
        """Most recent conversations first (created_at descending)."""
        conn = self._connect()
        try:
            rows = conn.execute(LIST_CONVERSATIONS_SQL, (limit, offset)).fetchall()
        finally:
            conn.close()
        return [self._row_to_conversation(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        conn = self._connect()
        try:
            row = conn.execute(GET_CONVERSATION_SQL, (conversation_id,)).fetchone()
        finally:
            conn.close()
        return self._row_to_conversation(row) if row is not None else None

    @staticmethod
    def _row_to_conversation(row) -> Conversation:
        return Conversation(
            id=row[0],
            farmer_phone=row[1],
            message_in=row[2],
            media_type=row[3],
            media_url=row[4],
            vision_result=row[5],
            rag_sources=row[6],
            response_out=row[7],
            confidence_flag=row[8],
            created_at=row[9],
            responded_at=row[10],
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)
