"""Tests for app.pipelines.image_pipeline."""

import pytest

from app.adapters.vision.base import VisionAdapter, VisionResult
from app.adapters.vision.image_validator import MAX_IMAGE_BYTES
from app.core.errors import PipelineError, ValidationError
from app.pipelines.image_pipeline import ImageAnalysisResult, ImagePipeline
from app.rag.models import RetrievalResult

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100


class FakeVisionAdapter(VisionAdapter):
    def __init__(self, result: VisionResult | None = None):
        self._result = result or VisionResult(
            crop="wheat",
            condition="diseased",
            symptoms=["leaf rust", "yellow spots"],
            confidence="high",
        )

    async def analyze_image(self, image_bytes: bytes) -> VisionResult:
        return self._result


class FailingVisionAdapter(VisionAdapter):
    async def analyze_image(self, image_bytes: bytes) -> VisionResult:
        raise RuntimeError("vision service down")


def _make_rag_results(n: int = 2) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"chunk_{i}",
            document_id="doc_1",
            text=f"RAG advice passage number {i} for wheat rust treatment.",
            score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


@pytest.fixture
def pipeline():
    return ImagePipeline(vision_adapter=FakeVisionAdapter())


@pytest.fixture
def pipeline_with_rag():
    def fake_retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]:
        return _make_rag_results(min(top_k, 2))

    return ImagePipeline(vision_adapter=FakeVisionAdapter(), rag_retrieve=fake_retrieve)


async def test_validation_rejects_empty(pipeline):
    with pytest.raises(ValidationError, match="empty"):
        await pipeline.process(b"")


async def test_validation_rejects_invalid_format(pipeline):
    with pytest.raises(ValidationError, match="Unsupported"):
        await pipeline.process(b"GIF89a" + b"\x00" * 100)


async def test_validation_rejects_oversized(pipeline):
    big = b"\xff\xd8\xff" + b"\x00" * 200
    pipeline._max_image_bytes = 100
    with pytest.raises(ValidationError, match="too large"):
        await pipeline.process(big)


async def test_returns_image_analysis_result(pipeline):
    result = await pipeline.process(JPEG_BYTES)
    assert isinstance(result, ImageAnalysisResult)
    assert result.vision.crop == "wheat"
    assert result.vision.condition == "diseased"
    assert result.vision.symptoms == ["leaf rust", "yellow spots"]
    assert result.vision.confidence == "high"


async def test_query_text_built_from_vision(pipeline):
    result = await pipeline.process(JPEG_BYTES)
    assert "wheat" in result.query_text
    assert "leaf rust" in result.query_text
    assert "diseased" in result.query_text


async def test_response_text_contains_crop_info(pipeline):
    result = await pipeline.process(JPEG_BYTES)
    assert "wheat" in result.response_text
    assert "diseased" in result.response_text


async def test_low_confidence_adds_urdu_disclaimer():
    low_conf_result = VisionResult(
        crop="unknown", condition="unknown", symptoms=[], confidence="low"
    )
    pipeline = ImagePipeline(vision_adapter=FakeVisionAdapter(result=low_conf_result))
    result = await pipeline.process(JPEG_BYTES)
    assert pipeline.URDU_UNCERTAINTY_DISCLAIMER in result.response_text


async def test_rag_retrieval_called_with_correct_query(pipeline_with_rag):
    calls = []

    def tracking_retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]:
        calls.append((query, top_k))
        return _make_rag_results(1)

    pipeline_with_rag._rag_retrieve = tracking_retrieve
    await pipeline_with_rag.process(JPEG_BYTES)
    assert len(calls) == 1
    query, top_k = calls[0]
    assert "wheat" in query
    assert top_k == 5


async def test_rag_failure_degrades_gracefully(pipeline):
    def failing_retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]:
        raise RuntimeError("index unavailable")

    pipeline._rag_retrieve = failing_retrieve
    result = await pipeline.process(JPEG_BYTES)
    assert result.rag_results == []
    assert "wheat" in result.response_text


async def test_no_rag_retriever_works(pipeline):
    assert pipeline._rag_retrieve is None
    result = await pipeline.process(JPEG_BYTES)
    assert result.rag_results == []
    assert isinstance(result.response_text, str)
    assert len(result.response_text) > 0


async def test_vision_failure_raises_pipeline_error():
    pipeline = ImagePipeline(vision_adapter=FailingVisionAdapter())
    with pytest.raises(PipelineError, match="Vision analysis failed"):
        await pipeline.process(JPEG_BYTES)


# --- Phase 4: LLM integration tests ---

from app.adapters.llm.base import LLMAdapter


class FakeLLMAdapter(LLMAdapter):
    def __init__(self, response: str = "گندم کی زنگ کے لیے منظور شدہ پھپھوند کش استعمال کریں۔"):
        self._response = response
        self.last_prompt = ""
        self.last_context = ""

    async def generate(self, prompt: str, context: str = "") -> str:
        self.last_prompt = prompt
        self.last_context = context
        return self._response


class FailingLLMAdapter(LLMAdapter):
    async def generate(self, prompt: str, context: str = "") -> str:
        raise RuntimeError("LLM service down")


async def test_llm_adapter_produces_llm_response():
    llm = FakeLLMAdapter(response="گندم کی بیماری کا مکمل علاج یہ ہے۔")

    def retrieve(query, top_k=5):
        return _make_rag_results(1)

    pipeline = ImagePipeline(
        vision_adapter=FakeVisionAdapter(),
        rag_retrieve=retrieve,
        llm_adapter=llm,
    )
    result = await pipeline.process(JPEG_BYTES)
    assert "گندم کی بیماری کا مکمل علاج" in result.response_text
    assert "wheat" in llm.last_prompt


async def test_no_llm_adapter_uses_template():
    pipeline = ImagePipeline(vision_adapter=FakeVisionAdapter())
    result = await pipeline.process(JPEG_BYTES)
    assert "فصل:" in result.response_text
    assert "حالت:" in result.response_text


async def test_llm_failure_falls_back_to_template():
    pipeline = ImagePipeline(
        vision_adapter=FakeVisionAdapter(),
        llm_adapter=FailingLLMAdapter(),
    )
    result = await pipeline.process(JPEG_BYTES)
    assert "فصل:" in result.response_text
    assert "wheat" in result.response_text


async def test_llm_response_with_no_rag_adds_disclaimer():
    llm = FakeLLMAdapter(response="فصل کو صاف رکھیں۔")
    pipeline = ImagePipeline(
        vision_adapter=FakeVisionAdapter(),
        llm_adapter=llm,
    )
    result = await pipeline.process(JPEG_BYTES)
    assert "یقینی نہیں" in result.response_text
