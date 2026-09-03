from app.adapters.speech.base import SpeechAdapter
from app.core.logging import get_logger
from app.core.errors import PipelineError

logger = get_logger(__name__)


class VoicePipeline:
    """Converts voice notes to text via speech-to-text.

    Phase 1 placeholder — real implementation in Phase 5.
    """

    def __init__(self, speech_adapter: SpeechAdapter):
        self.speech = speech_adapter

    async def process(self, audio_bytes: bytes, language: str = "ur") -> str:
        logger.info("VoicePipeline: transcribing audio (%d bytes), lang=%s", len(audio_bytes), language)
        try:
            transcript = await self.speech.transcribe(audio_bytes, language=language)
            logger.info("VoicePipeline: transcript length=%d chars", len(transcript))
            return transcript
        except Exception as e:
            raise PipelineError("VoicePipeline", f"Transcription failed: {e}") from e
