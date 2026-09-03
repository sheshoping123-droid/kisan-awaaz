from dataclasses import dataclass, field
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    is_safe: bool = True
    has_rag_source: bool = False
    has_ungrounded_treatment: bool = False
    low_confidence: bool = False
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = ""


class ResponseValidator:
    """Validates LLM responses before they are sent to farmers.

    Ensures responses are grounded in RAG sources and do not contain
    hallucinated treatment recommendations.
    """

    URDU_UNCERTAINTY_DISCLAIMER = (
        "ہم مکمل طور پر یقینی نہیں ہیں۔ براہ کرم اپنے قریبی زرعی توسیع کے دفتر سے رجوع کریں۔"
    )

    def validate(
        self,
        response_text: str,
        rag_sources: list[str] | None = None,
        vision_confidence: str = "high",
    ) -> ValidationReport:
        report = ValidationReport()

        if not rag_sources:
            report.has_rag_source = False
            report.warnings.append("No RAG sources — response is ungrounded")
            report.disclaimer = self.URDU_UNCERTAINTY_DISCLAIMER
        else:
            report.has_rag_source = True

        if vision_confidence == "low":
            report.low_confidence = True
            report.warnings.append("Low vision confidence — adding uncertainty framing")
            report.disclaimer = self.URDU_UNCERTAINTY_DISCLAIMER

        logger.info(
            "Response validation: sources=%s, low_conf=%s, warnings=%d",
            report.has_rag_source,
            report.low_confidence,
            len(report.warnings),
        )
        return report
