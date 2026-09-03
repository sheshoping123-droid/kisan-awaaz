import re
from dataclasses import dataclass, field
from app.core.logging import get_logger

logger = get_logger(__name__)

DOSAGE_PATTERN = re.compile(
    r'\d+\.?\d*\s*(?:ml|ltr|liters?|grams?|kg|g\b|oz|tbsp|tsp|ppm|%|hectare)',
    re.IGNORECASE,
)


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
        rag_texts: list[str] | None = None,
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

        if rag_texts is not None and rag_texts:
            if self._has_ungrounded_dosage(response_text, rag_texts):
                report.has_ungrounded_treatment = True
                report.warnings.append("Response contains dosage not found in RAG texts")

        logger.info(
            "Response validation: sources=%s, low_conf=%s, ungrounded=%s, warnings=%d",
            report.has_rag_source,
            report.low_confidence,
            report.has_ungrounded_treatment,
            len(report.warnings),
        )
        return report

    @staticmethod
    def _has_ungrounded_dosage(response_text: str, rag_texts: list[str]) -> bool:
        dosages = DOSAGE_PATTERN.findall(response_text)
        if not dosages:
            return False

        rag_combined = " ".join(rag_texts)
        for dosage in dosages:
            if dosage.strip() not in rag_combined:
                return True
        return False
