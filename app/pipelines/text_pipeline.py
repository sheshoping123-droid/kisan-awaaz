from app.core.logging import get_logger
from app.core.errors import PipelineError

logger = get_logger(__name__)


class TextPipeline:
    """Processes text queries through RAG retrieval and LLM generation.

    Phase 1 placeholder — real implementation in Phase 2.
    """

    async def process(self, text: str, vision_context: dict | None = None) -> str:
        logger.info("TextPipeline: processing query (%d chars)", len(text))
        if vision_context:
            logger.info("TextPipeline: received vision context for crop=%s", vision_context.get("crop", "unknown"))
        # Phase 2: RAG retrieval → LLM generation → Urdu response
        raise PipelineError("TextPipeline", "Not yet implemented — Phase 2")
