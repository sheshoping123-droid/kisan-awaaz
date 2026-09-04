"""Tests for app.adapters.whatsapp.twilio_adapter (no network needed)."""

import httpx
import pytest

from app.adapters.whatsapp.twilio_adapter import TwilioAdapter
from app.core.errors import AdapterError


@pytest.fixture
def adapter():
    return TwilioAdapter(
        account_sid="ACtest123",
        auth_token="test-auth-token",
        whatsapp_number="+14155238886",
    )


class FakeResponse:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data or {}
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


def make_fake_client(monkeypatch, post_response=None, get_response=None, post_error=None, get_error=None):
    requests = []

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, data=None, auth=None, **kwargs):
            requests.append(
                {"method": "POST", "url": url, "data": data, "auth": auth, "init": self.init_kwargs}
            )
            if post_error:
                raise post_error
            return post_response

        async def get(self, url, auth=None, **kwargs):
            requests.append(
                {"method": "GET", "url": url, "auth": auth, "init": self.init_kwargs}
            )
            if get_error:
                raise get_error
            return get_response

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    return requests


def test_normalize_from_adds_prefix(adapter):
    assert adapter._normalize_from("+14155238886") == "whatsapp:+14155238886"


def test_normalize_from_keeps_existing_prefix(adapter):
    assert adapter._normalize_from("whatsapp:+14155238886") == "whatsapp:+14155238886"


def test_parse_response_success(adapter):
    assert adapter._parse_response({"sid": "SM123"}) == "SM123"


def test_parse_response_missing_sid_raises(adapter):
    with pytest.raises(AdapterError, match="missing 'sid'"):
        adapter._parse_response({"status": "queued"})


def test_messages_url_contains_account_sid(adapter):
    assert adapter.messages_url == "https://api.twilio.com/2010-04-01/Accounts/ACtest123/Messages.json"


@pytest.mark.asyncio
async def test_send_text_posts_expected_payload(adapter, monkeypatch):
    requests = make_fake_client(
        monkeypatch,
        post_response=FakeResponse(json_data={"sid": "SM123"}),
    )

    await adapter.send_text("whatsapp:+923001234567", "سلام")

    assert len(requests) == 1
    req = requests[0]
    assert req["url"] == "https://api.twilio.com/2010-04-01/Accounts/ACtest123/Messages.json"
    assert req["auth"] == ("ACtest123", "test-auth-token")
    assert req["data"]["From"] == "whatsapp:+14155238886"
    assert req["data"]["To"] == "whatsapp:+923001234567"
    assert req["data"]["Body"] == "سلام"
    assert "MediaUrl" not in req["data"]


@pytest.mark.asyncio
async def test_send_audio_posts_expected_payload(adapter, monkeypatch):
    requests = make_fake_client(
        monkeypatch,
        post_response=FakeResponse(json_data={"sid": "SM456"}),
    )

    await adapter.send_audio("whatsapp:+923001234567", "https://example.com/audio.ogg")

    assert len(requests) == 1
    req = requests[0]
    assert req["data"]["From"] == "whatsapp:+14155238886"
    assert req["data"]["To"] == "whatsapp:+923001234567"
    assert req["data"]["MediaUrl"] == "https://example.com/audio.ogg"
    assert "Body" not in req["data"]


@pytest.mark.asyncio
async def test_send_text_http_error_raises_adapter_error(adapter, monkeypatch):
    request = httpx.Request("POST", "https://api.twilio.com/x")
    response = httpx.Response(500, request=request)
    status_error = httpx.HTTPStatusError("Server error", request=request, response=response)

    make_fake_client(monkeypatch, post_error=status_error)

    with pytest.raises(AdapterError, match="API returned 500"):
        await adapter.send_text("whatsapp:+923001234567", "سلام")


@pytest.mark.asyncio
async def test_download_media_returns_bytes_with_redirects(adapter, monkeypatch):
    requests = make_fake_client(
        monkeypatch,
        get_response=FakeResponse(content=b"media-bytes"),
    )

    result = await adapter.download_media("https://api.twilio.com/media/ME123")

    assert result == b"media-bytes"
    assert requests[0]["method"] == "GET"
    assert requests[0]["url"] == "https://api.twilio.com/media/ME123"
    assert requests[0]["auth"] == ("ACtest123", "test-auth-token")
    assert requests[0]["init"]["follow_redirects"] is True


@pytest.mark.asyncio
async def test_download_media_request_error_raises_adapter_error(adapter, monkeypatch):
    make_fake_client(monkeypatch, get_error=httpx.ConnectError("connection refused"))

    with pytest.raises(AdapterError, match="Media download failed"):
        await adapter.download_media("https://api.twilio.com/media/ME123")
