"""Tests for SQLite conversation persistence (app/models). Each test uses an
isolated temporary database file; no real external services are touched."""

import sqlite3

import pytest

from app.models.conversation import Conversation
from app.models.database import ConversationStore, database_path


def make_conversation(**overrides) -> Conversation:
    fields = dict(
        farmer_phone="whatsapp:+923001234567",
        message_in="میری گندم میں زنگ ہے",
        media_type="text",
        media_url=None,
        response_out="یہ جواب ہے۔",
        created_at="2026-09-04T10:00:00+00:00",
        responded_at="2026-09-04T10:00:05+00:00",
    )
    fields.update(overrides)
    return Conversation(**fields)


def fetch_rows(db_path, table: str = "conversations") -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
    finally:
        conn.close()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "kisan.db"


@pytest.fixture
def store(db_path):
    store = ConversationStore(f"sqlite:///{db_path}")
    store.init_schema()
    return store


def test_database_path_parses_sqlite_url(tmp_path):
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    assert database_path(url) == tmp_path / "db.sqlite"


def test_non_sqlite_url_rejected():
    with pytest.raises(ValueError, match="SQLite"):
        ConversationStore("postgresql://localhost/kisan_awaaz")


def test_init_schema_creates_parent_directories(tmp_path):
    db_path = tmp_path / "nested" / "sub" / "kisan.db"
    assert not db_path.exists()
    ConversationStore(f"sqlite:///{db_path}").init_schema()
    assert db_path.exists()


def test_init_schema_creates_both_tables(store, db_path):
    conn = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {"conversations", "knowledge_chunks"} <= tables


def test_conversations_columns_match_architecture(store, db_path):
    conn = sqlite3.connect(db_path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(conversations)")]
    finally:
        conn.close()
    assert columns == [
        "id",
        "farmer_phone",
        "message_in",
        "media_type",
        "media_url",
        "vision_result",
        "rag_sources",
        "response_out",
        "confidence_flag",
        "created_at",
        "responded_at",
    ]


def test_save_round_trips_all_fields(store, db_path):
    conversation = make_conversation()
    store.save(conversation)

    rows = fetch_rows(db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == conversation.id
    assert row["farmer_phone"] == "whatsapp:+923001234567"
    assert row["message_in"] == "میری گندم میں زنگ ہے"
    assert row["media_type"] == "text"
    assert row["media_url"] is None
    assert row["response_out"] == "یہ جواب ہے۔"
    assert row["created_at"] == "2026-09-04T10:00:00+00:00"
    assert row["responded_at"] == "2026-09-04T10:00:05+00:00"
    assert row["vision_result"] is None
    assert row["rag_sources"] is None
    assert row["confidence_flag"] is None


def test_save_persists_across_store_instances(store, db_path):
    store.save(make_conversation())
    ConversationStore(f"sqlite:///{db_path}").save(
        make_conversation(farmer_phone="whatsapp:+923009998887")
    )

    rows = fetch_rows(db_path)
    assert len(rows) == 2
    assert {row["farmer_phone"] for row in rows} == {
        "whatsapp:+923001234567",
        "whatsapp:+923009998887",
    }


def test_conversation_generates_unique_ids():
    first = Conversation(farmer_phone="whatsapp:+923001234567")
    second = Conversation(farmer_phone="whatsapp:+923001234567")
    assert first.id != second.id
    assert len(first.id) == 36
