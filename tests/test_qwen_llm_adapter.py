"""Tests for app.adapters.llm.qwen_adapter._parse_response (no network needed)."""

import pytest

from app.adapters.llm.qwen_adapter import QwenLLMAdapter
from app.core.errors import AdapterError


@pytest.fixture
def adapter():
    return QwenLLMAdapter(api_key="fake-key")


def _make_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_parse_response_successful(adapter):
    data = _make_response("گندم کی زنگ کے لیے یہ علاج تجویز کیا جاتا ہے۔")
    result = adapter._parse_response(data)
    assert isinstance(result, str)
    assert "گندم" in result


def test_parse_response_code_fenced(adapter):
    fenced = "```urdu\nگندم کی بیماری کا علاج۔\n```"
    result = adapter._parse_response(_make_response(fenced))
    assert "گندم" in result
    assert "```" not in result


def test_parse_response_empty_raises(adapter):
    with pytest.raises(AdapterError, match="Empty response"):
        adapter._parse_response(_make_response(""))


def test_parse_response_unexpected_structure_raises(adapter):
    with pytest.raises(AdapterError, match="Unexpected response structure"):
        adapter._parse_response({"unexpected_key": "no choices"})


def test_system_prompt_included_in_init():
    custom_prompt = "Custom system prompt for testing."
    adapter = QwenLLMAdapter(api_key="fake", system_prompt=custom_prompt)
    assert adapter._system_prompt == custom_prompt


def test_default_system_prompt(adapter):
    assert "Kisan Awaaz" in adapter._system_prompt
    assert "agricultural advisor" in adapter._system_prompt
