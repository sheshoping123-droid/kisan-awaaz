from __future__ import annotations

from app.adapters.whatsapp.base import WhatsAppAdapter, WhatsAppMessage
from app.core.logging import get_logger
from app.pipelines.image_pipeline import ImagePipeline
from app.pipelines.text_pipeline import TextPipeline

logger = get_logger(__name__)


class MessageRouter:
    """Routes incoming WhatsApp messages to the appropriate pipeline.

    Dispatches based on media_type: image -> ImagePipeline,
    audio -> VoicePipeline, text -> TextPipeline.
    """

    def __init__(
        self,
        image_pipeline: ImagePipeline | None = None,
        whatsapp_adapter: WhatsAppAdapter | None = None,
        text_pipeline: TextPipeline | None = None,
    ):
        self._image_pipeline = image_pipeline
        self._whatsapp = whatsapp_adapter
        self._text_pipeline = text_pipeline

    async def route(self, message: WhatsAppMessage) -> str:
        logger.info(
            "MessageRouter: from=%s, media_type=%s, body_len=%d",
            message.from_number,
            message.media_type,
            len(message.body) if message.body else 0,
        )

        if message.media_type == "image":
            return await self._handle_image(message)
        elif message.media_type == "audio":
            return await self._handle_voice(message)
        else:
            return await self._handle_text(message)

    async def _handle_image(self, message: WhatsAppMessage) -> str:
        if self._image_pipeline is None or self._whatsapp is None:
            logger.info("MessageRouter: image pipeline not configured")
            return "تصویر کا تجزیہ ابھی دستیاب نہیں ہے۔"

        if not message.media_url:
            return "براہ کرم تصویر دوبارہ بھیجیں۔"

        try:
            image_bytes = await self._whatsapp.download_media(message.media_url)
            result = await self._image_pipeline.process(image_bytes)
            return result.response_text
        except Exception as e:
            logger.error("MessageRouter: image processing failed: %s", e)
            return "معذرت، تصویر کا تجزیہ نہیں ہو سکا۔ براہ کرم دوبارہ کوشش کریں۔"

    async def _handle_voice(self, message: WhatsAppMessage) -> str:
        # Phase 5: VoicePipeline + TextPipeline
        logger.info("MessageRouter: voice handling not yet implemented")
        return "صوتی پیغامات کا جواب ابھی دستیاب نہیں ہے۔"

    async def _handle_text(self, message: WhatsAppMessage) -> str:
        if self._text_pipeline is None:
            logger.info("MessageRouter: text pipeline not configured")
            return "ٹیکسٹ کا جواب ابھی دستیاب نہیں ہے۔"

        if not message.body or not message.body.strip():
            return "براہ کرم اپنا سوال دوبارہ بھیجیں۔"

        try:
            return await self._text_pipeline.process(message.body)
        except Exception as e:
            logger.error("MessageRouter: text processing failed: %s", e)
            return "معذرت، آپ کے سوال کا جواب نہیں دیا جا سکا۔ براہ کرم دوبارہ کوشش کریں۔"
