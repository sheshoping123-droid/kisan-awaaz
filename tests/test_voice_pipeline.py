"""Tests for app.pipelines.voice_pipeline."""

import pytest

from app.adapters.speech.base import SpeechAdapter
from app.core.errors import PipelineError, ValidationError
from app.pipelines.text_pipeline import TextPipeline
from app.pipelines.voice_pipeline import VoicePipeline

OGG_BYTES = b"OggS" + b"\x00" * 100


class FakeSpeechAdapter(SpeechAdapter):
    def __init__(self, transcription: str = "میری گندم کی فصل میں زنگ لگ رہی ہے"):
        self._transcription = transcription
        self.last_audio = b""
        self.last_language = ""

    async def transcribe(self, audio_bytes: bytes, language: str = "ur") -> str:
        self.last_audio = audio_bytes
        self.last_language = language
        return self._transcription

    async def synthesize(self, text: str, language: str = "ur") -> bytes:
        raise NotImplementedError


class FailingSpeechAdapter(SpeechAdapter):
    async def transcribe(self, audio_bytes: bytes, language: str = "ur") -> str:
        raise RuntimeError("whisper service down")

    async def synthesize(self, text: str, language: str = "ur") -> bytes:
        raise NotImplementedError


class FakeTextPipeline:
    def __init__(self, response: str = "گندم کی زنگ کے لیے منظور شدہ علاج یہ ہے۔"):
        self._response = response
        self.last_text = ""

    async def process(self, text: str, vision_context: dict | None = None) -> str:
        self.last_text = text
        return self._response


class FailingTextPipeline:
    async def process(self, text: str, vision_context: dict | None = None) -> str:
        raise RuntimeError("text pipeline failure")


@pytest.fixture
def pipeline():
    return VoicePipeline(
        speech_adapter=FakeSpeechAdapter(),
        text_pipeline=FakeTextPipeline(),
    )


async def test_validation_rejects_empty(pipeline):
    with pytest.raises(ValidationError, match="empty"):
        await pipeline.process(b"")


async def test_validation_rejects_invalid_format(pipeline):
    with pytest.raises(ValidationError, match="Unsupported"):
        await pipeline.process(b"FLAC" + b"\x00" * 100)


async def test_validation_rejects_oversized(pipeline):
    pipeline._max_audio_bytes = 50
    with pytest.raises(ValidationError, match="too large"):
        await pipeline.process(OGG_BYTES)


async def test_transcription_passed_to_text_pipeline(pipeline):
    await pipeline.process(OGG_BYTES)
    assert pipeline._text_pipeline.last_text == "میری گندم کی فصل میں زنگ لگ رہی ہے"


async def test_full_flow_returns_text_pipeline_response(pipeline):
    result = await pipeline.process(OGG_BYTES)
    assert result == "گندم کی زنگ کے لیے منظور شدہ علاج یہ ہے۔"


async def test_audio_bytes_forwarded_to_speech_adapter(pipeline):
    await pipeline.process(OGG_BYTES)
    assert pipeline._speech.last_audio == OGG_BYTES


async def test_language_defaults_to_ur(pipeline):
    await pipeline.process(OGG_BYTES)
    assert pipeline._speech.last_language == "ur"


async def test_empty_transcription_raises_pipeline_error(pipeline):
    pipeline._speech = FakeSpeechAdapter(transcription="")
    with pytest.raises(PipelineError, match="empty"):
        await pipeline.process(OGG_BYTES)


async def test_whitespace_transcription_raises_pipeline_error(pipeline):
    pipeline._speech = FakeSpeechAdapter(transcription="   ")
    with pytest.raises(PipelineError, match="empty"):
        await pipeline.process(OGG_BYTES)


async def test_speech_failure_raises_pipeline_error(pipeline):
    pipeline._speech = FailingSpeechAdapter()
    with pytest.raises(PipelineError, match="Speech transcription failed"):
        await pipeline.process(OGG_BYTES)


async def test_text_pipeline_failure_propagates(pipeline):
    pipeline._text_pipeline = FailingTextPipeline()
    with pytest.raises(RuntimeError, match="text pipeline failure"):
        await pipeline.process(OGG_BYTES)


async def test_custom_language_forwarded():
    speech = FakeSpeechAdapter()
    pipeline = VoicePipeline(
        speech_adapter=speech,
        text_pipeline=FakeTextPipeline(),
        language="en",
    )
    await pipeline.process(OGG_BYTES)
    assert speech.last_language == "en"
