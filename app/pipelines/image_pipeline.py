"""Image analysis pipeline: validate -> vision -> RAG retrieval -> formatted response."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.adapters.vision.base import VisionAdapter, VisionResult
from app.adapters.vision.image_validator import validate_image
from app.core.errors import PipelineError
from app.core.logging import get_logger
from app.rag.models import RetrievalResult

logger = get_logger(__name__)


@dataclass
class ImageAnalysisResult:
    """Enriched result combining vision analysis with RAG context."""

    vision: VisionResult
    rag_results: list[RetrievalResult] = field(default_factory=list)
    query_text: str = ""
    response_text: str = ""


class ImagePipeline:
    """Processes crop images through validation, vision analysis, and RAG retrieval.

    Flow: validate bytes -> vision model -> build query -> RAG retrieve -> format response
    """

    URDU_UNCERTAINTY_DISCLAIMER = (
        "ہم مکمل طور پر یقینی نہیں ہیں۔ براہ کرم اپنے قریبی زرعی توسیع کے دفتر سے رجوع کریں۔"
    )

    def __init__(
        self,
        vision_adapter: VisionAdapter,
        rag_retrieve: Callable[..., list[RetrievalResult]] | None = None,
        max_image_bytes: int = 10 * 1024 * 1024,
        rag_top_k: int = 5,
    ):
        self.vision = vision_adapter
        self._rag_retrieve = rag_retrieve
        self._max_image_bytes = max_image_bytes
        self._rag_top_k = rag_top_k

    async def process(self, image_bytes: bytes) -> ImageAnalysisResult:
        """Run the full image analysis pipeline."""
        fmt = validate_image(image_bytes, max_bytes=self._max_image_bytes)
        logger.info(
            "ImagePipeline: validated image format=%s, size=%d bytes",
            fmt,
            len(image_bytes),
        )

        try:
            vision_result = await self.vision.analyze_image(image_bytes)
        except Exception as e:
            raise PipelineError(
                "ImagePipeline", f"Vision analysis failed: {e}"
            ) from e

        logger.info(
            "ImagePipeline: crop=%s, condition=%s, confidence=%s, symptoms=%d",
            vision_result.crop,
            vision_result.condition,
            vision_result.confidence,
            len(vision_result.symptoms),
        )

        query = self._build_query(vision_result)

        rag_results: list[RetrievalResult] = []
        if self._rag_retrieve is not None:
            try:
                rag_results = self._rag_retrieve(query, top_k=self._rag_top_k)
                logger.info("ImagePipeline: retrieved %d RAG results", len(rag_results))
            except Exception as e:
                logger.warning("ImagePipeline: RAG retrieval failed: %s", e)

        response_text = self._format_response(vision_result, rag_results)

        return ImageAnalysisResult(
            vision=vision_result,
            rag_results=rag_results,
            query_text=query,
            response_text=response_text,
        )

    def _build_query(self, result: VisionResult) -> str:
        """Convert VisionResult into a natural-language retrieval query."""
        parts = [result.crop]
        if result.symptoms:
            parts.append(", ".join(result.symptoms))
        if result.condition and result.condition != "unknown":
            parts.append(result.condition)
        return " ".join(parts)

    def _format_response(
        self,
        vision: VisionResult,
        rag_results: list[RetrievalResult],
    ) -> str:
        """Build an Urdu response from vision + RAG context.

        Phase 3: template-based. Phase 4 will replace this with LLM generation.
        """
        symptoms_text = (
            "، ".join(vision.symptoms) if vision.symptoms else "کوئی واضح علامات نہیں"
        )

        lines = [
            f"فصل: {vision.crop}",
            f"حالت: {vision.condition}",
            f"علامات: {symptoms_text}",
        ]

        if vision.confidence == "low":
            lines.append("")
            lines.append(self.URDU_UNCERTAINTY_DISCLAIMER)

        if rag_results:
            lines.append("")
            lines.append("تجویز کردہ معلومات:")
            for r in rag_results[:3]:
                text = r.text[:200].strip()
                lines.append(f"• {text}")

        return "\n".join(lines)
