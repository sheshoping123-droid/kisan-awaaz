import pytest
from app.router import MessageRouter
from app.adapters.whatsapp.base import WhatsAppMessage


@pytest.fixture
def router():
    return MessageRouter()


@pytest.mark.asyncio
async def test_text_message_routing(router):
    msg = WhatsAppMessage(from_number="+923001234567", body="میری فصل میں بیماری ہے")
    response = await router.route(msg)
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_image_message_routing(router):
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="",
        media_type="image",
        media_url="https://example.com/crop.jpg",
    )
    response = await router.route(msg)
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_voice_message_routing(router):
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="",
        media_type="audio",
        media_url="https://example.com/voice.ogg",
    )
    response = await router.route(msg)
    assert isinstance(response, str)


# --- Phase 3: wired image pipeline tests ---

from app.adapters.vision.base import VisionAdapter, VisionResult
from app.adapters.whatsapp.base import WhatsAppAdapter
from app.pipelines.image_pipeline import ImagePipeline

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100


class FakeVisionAdapter(VisionAdapter):
    async def analyze_image(self, image_bytes: bytes) -> VisionResult:
        return VisionResult(
            crop="wheat",
            condition="diseased",
            symptoms=["leaf rust"],
            confidence="high",
        )


class FakeWhatsAppAdapter(WhatsAppAdapter):
    async def send_text(self, to: str, body: str) -> None:
        pass

    async def send_audio(self, to: str, audio_url: str) -> None:
        pass

    async def download_media(self, media_url: str) -> bytes:
        return JPEG_BYTES


@pytest.fixture
def wired_router():
    pipeline = ImagePipeline(vision_adapter=FakeVisionAdapter())
    return MessageRouter(image_pipeline=pipeline, whatsapp_adapter=FakeWhatsAppAdapter())


@pytest.mark.asyncio
async def test_image_routing_with_pipeline(wired_router):
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="",
        media_type="image",
        media_url="https://example.com/crop.jpg",
    )
    response = await wired_router.route(msg)
    assert "wheat" in response


@pytest.mark.asyncio
async def test_image_routing_no_media_url(wired_router):
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="",
        media_type="image",
        media_url=None,
    )
    response = await wired_router.route(msg)
    assert "براہ کرم" in response


@pytest.mark.asyncio
async def test_image_routing_without_pipeline_returns_placeholder():
    router = MessageRouter()
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="",
        media_type="image",
        media_url="https://example.com/crop.jpg",
    )
    response = await router.route(msg)
    assert "دستیاب نہیں" in response


# --- Phase 4: wired text pipeline tests ---

from app.adapters.llm.base import LLMAdapter
from app.pipelines.text_pipeline import TextPipeline


class FakeTextLLMAdapter(LLMAdapter):
    async def generate(self, prompt: str, context: str = "") -> str:
        return "گندم کی زنگ کے لیے منظور شدہ پھپھوند کش استعمال کریں۔"


class FailingTextPipeline(TextPipeline):
    async def process(self, text: str, vision_context=None) -> str:
        raise RuntimeError("pipeline error")


@pytest.fixture
def text_wired_router():
    pipeline = TextPipeline(llm_adapter=FakeTextLLMAdapter())
    return MessageRouter(text_pipeline=pipeline)


@pytest.mark.asyncio
async def test_text_routing_with_pipeline(text_wired_router):
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="میری گندم کی فصل میں زنگ لگ رہی ہے",
    )
    response = await text_wired_router.route(msg)
    assert "گندم" in response


@pytest.mark.asyncio
async def test_text_routing_without_pipeline_returns_placeholder():
    router = MessageRouter()
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="میری فصل میں بیماری ہے",
    )
    response = await router.route(msg)
    assert "دستیاب نہیں" in response


@pytest.mark.asyncio
async def test_text_routing_pipeline_error_returns_graceful_message():
    pipeline = FailingTextPipeline(llm_adapter=FakeTextLLMAdapter())
    router = MessageRouter(text_pipeline=pipeline)
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="میری فصل میں بیماری ہے",
    )
    response = await router.route(msg)
    assert "معذرت" in response


# --- Phase 5: wired voice pipeline tests ---

OGG_BYTES = b"OggS" + b"\x00" * 100


class FakeVoiceWhatsAppAdapter(WhatsAppAdapter):
    async def send_text(self, to: str, body: str) -> None:
        pass

    async def send_audio(self, to: str, audio_url: str) -> None:
        pass

    async def download_media(self, media_url: str) -> bytes:
        return OGG_BYTES


class FakeVoicePipeline:
    async def process(self, audio_bytes: bytes) -> str:
        return "صوتی پیغام کا تجزیہ مکمل ہوا: گندم کی زنگ کا علاج۔"


class FailingVoicePipeline:
    async def process(self, audio_bytes: bytes) -> str:
        raise RuntimeError("voice pipeline failure")


@pytest.fixture
def voice_wired_router():
    return MessageRouter(
        voice_pipeline=FakeVoicePipeline(),
        whatsapp_adapter=FakeVoiceWhatsAppAdapter(),
    )


@pytest.mark.asyncio
async def test_voice_routing_with_pipeline(voice_wired_router):
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="",
        media_type="audio",
        media_url="https://example.com/voice.ogg",
    )
    response = await voice_wired_router.route(msg)
    assert "گندم" in response


@pytest.mark.asyncio
async def test_voice_routing_no_media_url(voice_wired_router):
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="",
        media_type="audio",
        media_url=None,
    )
    response = await voice_wired_router.route(msg)
    assert "براہ کرم" in response


@pytest.mark.asyncio
async def test_voice_routing_pipeline_error_returns_graceful_message():
    router = MessageRouter(
        voice_pipeline=FailingVoicePipeline(),
        whatsapp_adapter=FakeVoiceWhatsAppAdapter(),
    )
    msg = WhatsAppMessage(
        from_number="+923001234567",
        body="",
        media_type="audio",
        media_url="https://example.com/voice.ogg",
    )
    response = await router.route(msg)
    assert "معذرت" in response
