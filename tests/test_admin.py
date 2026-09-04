"""Tests for admin endpoints: API-key auth, conversation listing/detail, knowledge reload.

Uses an isolated temporary database and fake RAG services; no real external calls."""

import pytest

from app.core.config import settings
from app.main import app
from app.models.conversation import Conversation
from app.models.database import ConversationStore

ADMIN_KEY = "test-admin-key"


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


@pytest.fixture
def admin_store(tmp_path):
    store = ConversationStore(f"sqlite:///{tmp_path / 'admin.db'}")
    store.init_schema()
    app.state.conversation_store = store
    yield store
    if hasattr(app.state, "conversation_store"):
        delattr(app.state, "conversation_store")


@pytest.fixture
def auth(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)
    return {"X-API-Key": ADMIN_KEY}


def seed_three(store):
    store.save(make_conversation(id="conv-oldest", created_at="2026-09-04T09:00:00+00:00"))
    store.save(make_conversation(id="conv-middle", created_at="2026-09-04T10:00:00+00:00"))
    store.save(make_conversation(id="conv-newest", created_at="2026-09-04T11:00:00+00:00"))


class FakeRAGService:
    def __init__(self, summary=None, error=None):
        self.calls = 0
        self._summary = summary or {
            "documents": 3,
            "chunks": 12,
            "dimension": 384,
            "index_path": "knowledge/processed",
        }
        self._error = error

    def index_documents(self):
        self.calls += 1
        if self._error:
            raise self._error
        return self._summary


# --- authentication ---


@pytest.mark.asyncio
async def test_admin_missing_api_key_returns_401(client, admin_store, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)

    response = await client.get("/admin/conversations")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_wrong_api_key_returns_401(client, admin_store, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)

    response = await client.get("/admin/conversations", headers={"X-API-Key": "wrong"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_unconfigured_key_fails_closed(client, admin_store, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "")

    response = await client.get("/admin/conversations", headers={"X-API-Key": "anything"})

    assert response.status_code == 503


# --- GET /admin/conversations ---


@pytest.mark.asyncio
async def test_list_returns_most_recent_first(client, admin_store, auth):
    seed_three(admin_store)

    response = await client.get("/admin/conversations", headers=auth)

    assert response.status_code == 200
    data = response.json()
    assert [c["id"] for c in data] == ["conv-newest", "conv-middle", "conv-oldest"]
    first = data[0]
    assert first["farmer_phone"] == "whatsapp:+923001234567"
    assert first["message_in"] == "میری گندم میں زنگ ہے"
    assert first["media_type"] == "text"
    assert "response_out" not in first


@pytest.mark.asyncio
async def test_list_respects_limit_and_offset(client, admin_store, auth):
    seed_three(admin_store)

    response = await client.get("/admin/conversations", headers=auth, params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    assert [c["id"] for c in response.json()] == ["conv-middle", "conv-oldest"]


@pytest.mark.asyncio
async def test_list_empty_returns_empty_array(client, admin_store, auth):
    response = await client.get("/admin/conversations", headers=auth)

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_store_not_wired_returns_503(client, auth, monkeypatch):
    monkeypatch.delattr(app.state, "conversation_store", raising=False)

    response = await client.get("/admin/conversations", headers=auth)

    assert response.status_code == 503


# --- GET /admin/conversations/{id} ---


@pytest.mark.asyncio
async def test_detail_returns_all_fields(client, admin_store, auth):
    admin_store.save(make_conversation(id="conv-1", media_type="image", media_url="https://example.com/m.jpg"))

    response = await client.get("/admin/conversations/conv-1", headers=auth)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "conv-1"
    assert data["media_type"] == "image"
    assert data["media_url"] == "https://example.com/m.jpg"
    assert data["response_out"] == "یہ جواب ہے۔"
    assert data["created_at"] == "2026-09-04T10:00:00+00:00"
    assert data["vision_result"] is None
    assert data["rag_sources"] is None
    assert data["confidence_flag"] is None


@pytest.mark.asyncio
async def test_detail_not_found_returns_404(client, admin_store, auth):
    response = await client.get("/admin/conversations/missing", headers=auth)

    assert response.status_code == 404


# --- POST /admin/knowledge/reload ---


@pytest.mark.asyncio
async def test_reload_reingests_knowledge_base(client, auth, monkeypatch):
    fake = FakeRAGService()
    monkeypatch.setattr("app.admin._build_rag_service", lambda: fake)

    response = await client.post("/admin/knowledge/reload", headers=auth)

    assert response.status_code == 200
    assert response.json() == {
        "status": "reloaded",
        "documents": 3,
        "chunks": 12,
        "dimension": 384,
        "index_path": "knowledge/processed",
    }
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_reload_failure_returns_500(client, auth, monkeypatch):
    fake = FakeRAGService(error=RuntimeError("embedding model failed"))
    monkeypatch.setattr("app.admin._build_rag_service", lambda: fake)

    response = await client.post("/admin/knowledge/reload", headers=auth)

    assert response.status_code == 500
    assert "embedding model failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_reload_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)
    fake = FakeRAGService()
    monkeypatch.setattr("app.admin._build_rag_service", lambda: fake)

    response = await client.post("/admin/knowledge/reload")

    assert response.status_code == 401
    assert fake.calls == 0
