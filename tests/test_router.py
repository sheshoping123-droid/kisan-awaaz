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
