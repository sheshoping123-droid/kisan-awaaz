"""Tests for app.adapters.speech.whisper_adapter (no network needed)."""

import pytest

from app.adapters.speech.whisper_adapter import WhisperAdapter
from app.core.errors import AdapterError

OGG_BYTES = b"OggS" + b"\x00" * 100


@pytest.fixture
def adapter():
    return WhisperAdapter(api_key="fake-key")


def test_parse_response_successful(adapter):
    data = {"text": "میری گندم کی فصل میں زنگ لگ رہی ہے"}
    result = adapter._parse_response(data)
    assert result == "میری گندم کی فصل میں زنگ لگ رہی ہے"


def test_parse_response_strips_whitespace(adapter):
    data = {"text": "  کیا علاج ہے  "}
    result = adapter._parse_response(data)
    assert result == "کیا علاج ہے"


def test_parse_response_missing_text_key_raises(adapter):
    with pytest.raises(AdapterError, match="missing 'text' key"):
        adapter._parse_response({"unexpected": "no text"})


def test_parse_response_empty_text_raises(adapter):
    with pytest.raises(AdapterError, match="Empty transcription"):
        adapter._parse_response({"text": "   "})


def test_synthesize_not_implemented(adapter):
    with pytest.raises(NotImplementedError):
        import asyncio
        asyncio.run(adapter.synthesize("متن"))


def test_default_config():
    adapter = WhisperAdapter(api_key="key")
    assert adapter._model == "whisper-1"
    assert adapter._timeout == 60
    assert adapter._max_audio_bytes == 25 * 1024 * 1024
