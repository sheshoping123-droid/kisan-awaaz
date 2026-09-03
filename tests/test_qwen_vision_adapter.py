"""Tests for app.adapters.vision.qwen_adapter._parse_response (no network needed)."""

import pytest

from app.adapters.vision.base import VisionResult
from app.adapters.vision.qwen_adapter import QwenVisionAdapter
from app.core.errors import AdapterError


@pytest.fixture
def adapter():
    return QwenVisionAdapter(api_key="fake-key")


def _make_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_parse_response_successful_json(adapter):
    data = _make_response(
        '{"crop": "wheat", "condition": "diseased", '
        '"symptoms": ["leaf rust"], "confidence": "high"}'
    )
    result = adapter._parse_response(data)
    assert isinstance(result, VisionResult)
    assert result.crop == "wheat"
    assert result.condition == "diseased"
    assert result.symptoms == ["leaf rust"]
    assert result.confidence == "high"


def test_parse_response_code_fenced_json(adapter):
    fenced = '```json\n{"crop": "rice", "condition": "healthy", ' \
             '"symptoms": [], "confidence": "high"}\n```'
    result = adapter._parse_response(_make_response(fenced))
    assert result.crop == "rice"
    assert result.condition == "healthy"


def test_parse_response_malformed_json_returns_low_confidence(adapter):
    result = adapter._parse_response(_make_response("this is not json at all"))
    assert result.crop == "unknown"
    assert result.condition == "unknown"
    assert result.confidence == "low"


def test_parse_response_missing_fields_uses_defaults(adapter):
    data = _make_response('{"crop": "maize"}')
    result = adapter._parse_response(data)
    assert result.crop == "maize"
    assert result.condition == "unknown"
    assert result.symptoms == []
    assert result.confidence == "low"


def test_parse_response_unexpected_structure_raises(adapter):
    with pytest.raises(AdapterError, match="Unexpected response structure"):
        adapter._parse_response({"unexpected_key": "no choices here"})
