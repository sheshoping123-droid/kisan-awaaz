"""Tests for app.pipelines.text_pipeline."""

import pytest

from app.adapters.llm.base import LLMAdapter
from app.core.errors import PipelineError, ValidationError
from app.pipelines.text_pipeline import TextPipeline, SYSTEM_PROMPT
from app.rag.models import RetrievalResult


class FakeLLMAdapter(LLMAdapter):
    def __init__(self, response: str = "یہ گندم کی زنگ کے بارے میں مشورہ ہے۔"):
        self._response = response
        self.last_prompt = ""
        self.last_context = ""

    async def generate(self, prompt: str, context: str = "") -> str:
        self.last_prompt = prompt
        self.last_context = context
        return self._response


class FailingLLMAdapter(LLMAdapter):
    async def generate(self, prompt: str, context: str = "") -> str:
        raise RuntimeError("LLM service unavailable")


def _make_rag_results(n: int = 2) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"chunk_{i}",
            document_id=f"doc_{i}",
            text=f"RAG advice passage {i} about wheat rust treatment.",
            score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


def fake_retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]:
    return _make_rag_results(min(top_k, 2))


@pytest.fixture
def pipeline():
    return TextPipeline(llm_adapter=FakeLLMAdapter())


@pytest.fixture
def pipeline_with_rag():
    return TextPipeline(llm_adapter=FakeLLMAdapter(), rag_retrieve=fake_retrieve)


async def test_full_pipeline_returns_response(pipeline_with_rag):
    result = await pipeline_with_rag.process("میری گندم کی فصل میں زنگ لگ رہی ہے")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_empty_text_raises_validation_error(pipeline):
    with pytest.raises(ValidationError, match="empty"):
        await pipeline.process("")


async def test_whitespace_text_raises_validation_error(pipeline):
    with pytest.raises(ValidationError, match="empty"):
        await pipeline.process("   ")


async def test_llm_called_with_correct_prompt(pipeline_with_rag):
    llm = FakeLLMAdapter()
    pipeline_with_rag._llm = llm
    await pipeline_with_rag.process("کیا علاج ہے")
    assert llm.last_prompt == "کیا علاج ہے"


async def test_rag_context_passed_to_llm(pipeline_with_rag):
    llm = FakeLLMAdapter()
    pipeline_with_rag._llm = llm
    await pipeline_with_rag.process("کیا علاج ہے")
    assert "RAG advice passage" in llm.last_context


async def test_no_rag_retriever_works(pipeline):
    llm = FakeLLMAdapter()
    pipeline._llm = llm
    result = await pipeline.process("کیا علاج ہے")
    assert llm.last_context == ""
    assert isinstance(result, str)


async def test_rag_failure_degrades_gracefully(pipeline):
    def failing_retrieve(query: str, top_k: int = 5):
        raise RuntimeError("index unavailable")

    pipeline._rag_retrieve = failing_retrieve
    result = await pipeline.process("کیا علاج ہے")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_vision_context_enriches_query(pipeline_with_rag):
    llm = FakeLLMAdapter()
    calls = []

    def tracking_retrieve(query: str, top_k: int = 5):
        calls.append(query)
        return _make_rag_results(1)

    pipeline_with_rag._llm = llm
    pipeline_with_rag._rag_retrieve = tracking_retrieve
    vision_ctx = {"crop": "wheat", "symptoms": ["leaf rust"]}
    await pipeline_with_rag.process("کیا علاج ہے", vision_context=vision_ctx)
    assert len(calls) == 1
    assert "wheat" in calls[0]
    assert "leaf rust" in calls[0]


async def test_llm_failure_raises_pipeline_error(pipeline):
    pipeline._llm = FailingLLMAdapter()
    with pytest.raises(PipelineError, match="LLM generation failed"):
        await pipeline.process("کیا علاج ہے")


async def test_no_rag_sources_adds_disclaimer(pipeline):
    response = await pipeline.process("کیا علاج ہے")
    assert "یقینی نہیں" in response


async def test_with_rag_sources_no_disclaimer(pipeline_with_rag):
    response = await pipeline_with_rag.process("کیا علاج ہے")
    assert "یقینی نہیں" not in response


async def test_ungrounded_treatment_returns_fallback():
    llm = FakeLLMAdapter(response="500ml fungicide spray کریں ہر 7 دن")

    def retrieve_with_text(query, top_k=5):
        return [
            RetrievalResult(
                chunk_id="c1",
                document_id="doc_1",
                text="wheat rust can be treated with approved fungicides",
                score=0.8,
            )
        ]

    pipeline = TextPipeline(llm_adapter=llm, rag_retrieve=retrieve_with_text)
    result = await pipeline.process("علاج بتائیں")
    assert "معذرت" in result
    assert "زرعی توسیع" in result


async def test_system_prompt_constant():
    assert "Kisan Awaaz" in SYSTEM_PROMPT
    assert "Urdu" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT
