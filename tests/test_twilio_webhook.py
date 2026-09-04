"""Tests for Twilio webhook validation, parsing, and the /webhook/whatsapp endpoint."""

import base64
import hashlib
import hmac
import sqlite3

import pytest

from app.adapters.whatsapp.base import WhatsAppAdapter
from app.adapters.whatsapp.webhook import (
    parse_incoming_message,
    validate_twilio_signature,
)
from app.core.config import settings
from app.core.errors import ValidationError
from app.main import app
from app.models.database import ConversationStore
from app.router import MessageRouter

TEST_TOKEN = "test-auth-token"
WEBHOOK_URL = "http://test/webhook/whatsapp"


def compute_signature(url: str, params: dict, token: str) -> str:
    concatenated = url + "".join(f"{k}{v}" for k, v in sorted(params.items()))
    digest = hmac.new(
        token.encode("utf-8"),
        concatenated.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("ascii")


# --- signature validation ---


def test_valid_signature_accepted():
    params = {"From": "whatsapp:+923001234567", "Body": "hello"}
    sig = compute_signature(WEBHOOK_URL, params, TEST_TOKEN)
    assert validate_twilio_signature(WEBHOOK_URL, params, sig, TEST_TOKEN) is True


def test_tampered_params_rejected():
    params = {"From": "whatsapp:+923001234567", "Body": "hello"}
    sig = compute_signature(WEBHOOK_URL, {"From": "whatsapp:+923001234567", "Body": "hacked"}, TEST_TOKEN)
    assert validate_twilio_signature(WEBHOOK_URL, params, sig, TEST_TOKEN) is False


def test_wrong_token_rejected():
    params = {"From": "whatsapp:+923001234567"}
    sig = compute_signature(WEBHOOK_URL, params, "other-token")
    assert validate_twilio_signature(WEBHOOK_URL, params, sig, TEST_TOKEN) is False


def test_empty_signature_rejected():
    params = {"From": "whatsapp:+923001234567"}
    assert validate_twilio_signature(WEBHOOK_URL, params, "", TEST_TOKEN) is False


# --- message parsing ---


def test_parse_text_message():
    msg = parse_incoming_message({"From": "whatsapp:+923001234567", "Body": "میری فصل میں زنگ ہے"})
    assert msg.from_number == "whatsapp:+923001234567"
    assert msg.body == "میری فصل میں زنگ ہے"
    assert msg.media_type is None
    assert msg.media_url is None


def test_parse_image_message():
    form = {
        "From": "whatsapp:+923001234567",
        "Body": "",
        "NumMedia": "1",
        "MediaUrl0": "https://api.twilio.com/2010-04-01/Accounts/ACxx/Media/MExx",
        "MediaContentType0": "image/jpeg",
    }
    msg = parse_incoming_message(form)
    assert msg.media_type == "image"
    assert msg.media_url == form["MediaUrl0"]


def test_parse_audio_message():
    form = {
        "From": "whatsapp:+923001234567",
        "NumMedia": "1",
        "MediaUrl0": "https://api.twilio.com/media/xx",
        "MediaContentType0": "audio/ogg; codecs=opus",
    }
    msg = parse_incoming_message(form)
    assert msg.media_type == "audio"
    assert msg.media_url == form["MediaUrl0"]


def test_parse_missing_from_raises():
    with pytest.raises(ValidationError, match="'From' is required"):
        parse_incoming_message({"Body": "hello"})


# --- endpoint ---


class FakeWhatsAppAdapter(WhatsAppAdapter):
    def __init__(self):
        self.sent = []

    async def send_text(self, to: str, body: str) -> None:
        self.sent.append((to, body))

    async def send_audio(self, to: str, audio_url: str) -> None:
        self.sent.append((to, audio_url))

    async def download_media(self, media_url: str) -> bytes:
        return b"fake-media"


class RecordingRouter(MessageRouter):
    def __init__(self, reply: str = "یہ جواب ہے۔"):
        super().__init__()
        self.messages = []
        self.reply = reply

    async def route(self, message):
        self.messages.append(message)
        return self.reply


@pytest.fixture
def wired():
    whatsapp = FakeWhatsAppAdapter()
    router = RecordingRouter()
    app.state.router = router
    app.state.whatsapp = whatsapp
    yield router, whatsapp
    for attr in ("router", "whatsapp"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.mark.asyncio
async def test_webhook_text_message_routes_and_replies(client, wired, monkeypatch):
    router, whatsapp = wired
    monkeypatch.setattr(settings, "twilio_auth_token", "")

    response = await client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+923001234567", "Body": "میری گندم میں زنگ ہے"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(router.messages) == 1
    assert router.messages[0].body == "میری گندم میں زنگ ہے"
    assert router.messages[0].media_type is None
    assert whatsapp.sent == [("whatsapp:+923001234567", "یہ جواب ہے۔")]


@pytest.mark.asyncio
async def test_webhook_valid_signature_processes_message(client, wired, monkeypatch):
    router, whatsapp = wired
    monkeypatch.setattr(settings, "twilio_auth_token", TEST_TOKEN)
    form = {"From": "whatsapp:+923001234567", "Body": "hello"}
    sig = compute_signature(WEBHOOK_URL, form, TEST_TOKEN)

    response = await client.post(
        "/webhook/whatsapp",
        data=form,
        headers={"X-Twilio-Signature": sig},
    )

    assert response.status_code == 200
    assert len(router.messages) == 1
    assert whatsapp.sent[0][0] == "whatsapp:+923001234567"


@pytest.mark.asyncio
async def test_webhook_invalid_signature_returns_403(client, wired, monkeypatch):
    router, whatsapp = wired
    monkeypatch.setattr(settings, "twilio_auth_token", TEST_TOKEN)

    response = await client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+923001234567", "Body": "hello"},
        headers={"X-Twilio-Signature": "deadbeef"},
    )

    assert response.status_code == 403
    assert router.messages == []
    assert whatsapp.sent == []


@pytest.mark.asyncio
async def test_webhook_image_message_routed_to_image_handler(client, wired, monkeypatch):
    router, whatsapp = wired
    monkeypatch.setattr(settings, "twilio_auth_token", "")

    response = await client.post(
        "/webhook/whatsapp",
        data={
            "From": "whatsapp:+923001234567",
            "NumMedia": "1",
            "MediaUrl0": "https://api.twilio.com/media/xx",
            "MediaContentType0": "image/jpeg",
        },
    )

    assert response.status_code == 200
    assert router.messages[0].media_type == "image"
    assert router.messages[0].media_url == "https://api.twilio.com/media/xx"
    assert len(whatsapp.sent) == 1


@pytest.mark.asyncio
async def test_webhook_audio_message_routed_to_audio_handler(client, wired, monkeypatch):
    router, whatsapp = wired
    monkeypatch.setattr(settings, "twilio_auth_token", "")

    response = await client.post(
        "/webhook/whatsapp",
        data={
            "From": "whatsapp:+923001234567",
            "NumMedia": "1",
            "MediaUrl0": "https://api.twilio.com/media/yy",
            "MediaContentType0": "audio/ogg; codecs=opus",
        },
    )

    assert response.status_code == 200
    assert router.messages[0].media_type == "audio"
    assert len(whatsapp.sent) == 1


@pytest.mark.asyncio
async def test_webhook_missing_from_returns_422(client, monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", "")

    response = await client.post("/webhook/whatsapp", data={"Body": "hello"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_webhook_router_not_wired_returns_503(client, monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", "")
    monkeypatch.delattr(app.state, "router", raising=False)
    monkeypatch.delattr(app.state, "whatsapp", raising=False)

    response = await client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+923001234567", "Body": "hello"},
    )

    assert response.status_code == 503


# --- conversation persistence ---


@pytest.fixture
def persisted(tmp_path):
    db_path = tmp_path / "webhook.db"
    store = ConversationStore(f"sqlite:///{db_path}")
    store.init_schema()
    app.state.router = RecordingRouter()
    app.state.whatsapp = FakeWhatsAppAdapter()
    app.state.conversation_store = store
    yield store
    for attr in ("router", "whatsapp", "conversation_store"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def fetch_rows(db_path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM conversations")]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_webhook_persists_text_conversation(client, persisted, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", "")

    response = await client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+923001234567", "Body": "میری گندم میں زنگ ہے"},
    )

    assert response.status_code == 200
    rows = fetch_rows(tmp_path / "webhook.db")
    assert len(rows) == 1
    row = rows[0]
    assert row["farmer_phone"] == "whatsapp:+923001234567"
    assert row["message_in"] == "میری گندم میں زنگ ہے"
    assert row["media_type"] == "text"
    assert row["media_url"] is None
    assert row["response_out"] == "یہ جواب ہے۔"
    assert row["created_at"]
    assert row["responded_at"]


@pytest.mark.asyncio
async def test_webhook_persists_image_conversation(client, persisted, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", "")

    response = await client.post(
        "/webhook/whatsapp",
        data={
            "From": "whatsapp:+923001234567",
            "NumMedia": "1",
            "MediaUrl0": "https://api.twilio.com/media/xx",
            "MediaContentType0": "image/jpeg",
        },
    )

    assert response.status_code == 200
    rows = fetch_rows(tmp_path / "webhook.db")
    assert len(rows) == 1
    assert rows[0]["media_type"] == "image"
    assert rows[0]["media_url"] == "https://api.twilio.com/media/xx"
    assert rows[0]["message_in"] is None


@pytest.mark.asyncio
async def test_webhook_persists_voice_conversation(client, persisted, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", "")

    response = await client.post(
        "/webhook/whatsapp",
        data={
            "From": "whatsapp:+923001234567",
            "NumMedia": "1",
            "MediaUrl0": "https://api.twilio.com/media/yy",
            "MediaContentType0": "audio/ogg; codecs=opus",
        },
    )

    assert response.status_code == 200
    rows = fetch_rows(tmp_path / "webhook.db")
    assert len(rows) == 1
    assert rows[0]["media_type"] == "voice"
    assert rows[0]["media_url"] == "https://api.twilio.com/media/yy"


class BrokenStore:
    def save(self, conversation):
        raise RuntimeError("database unavailable")


@pytest.mark.asyncio
async def test_webhook_persistence_failure_still_replies(client, wired, monkeypatch):
    router, whatsapp = wired
    monkeypatch.setattr(settings, "twilio_auth_token", "")
    monkeypatch.setattr(app.state, "conversation_store", BrokenStore(), raising=False)

    response = await client.post(
        "/webhook/whatsapp",
        data={"From": "whatsapp:+923001234567", "Body": "hello"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert whatsapp.sent == [("whatsapp:+923001234567", "یہ جواب ہے۔")]
