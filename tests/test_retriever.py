import numpy as np
import pytest

from app.rag.index import FAISSVectorIndex
from app.rag.models import DocumentChunk
from app.rag.retriever import Retriever
from tests.conftest import FakeEmbeddings


def _build_index(dim: int = 32, n: int = 5) -> FAISSVectorIndex:
    index = FAISSVectorIndex(dimension=dim)
    chunks = [
        DocumentChunk(
            chunk_id=f"chunk-{i}",
            document_id="doc1",
            text=f"Agriculture text about crops and diseases number {i}.",
            metadata={"filename": "test.txt", "chunk_index": i},
        )
        for i in range(n)
    ]
    rng = np.random.default_rng(seed=77)
    embeddings = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    index.add(embeddings, chunks)
    return index


def test_retrieve_returns_results():
    emb = FakeEmbeddings(dimension=32)
    index = _build_index()
    retriever = Retriever(emb, index)
    results = retriever.retrieve("wheat disease", top_k=3)
    assert len(results) == 3


def test_retrieve_top_k():
    emb = FakeEmbeddings(dimension=32)
    index = _build_index(n=10)
    retriever = Retriever(emb, index)
    results = retriever.retrieve("query", top_k=5)
    assert len(results) == 5


def test_retrieve_preserves_metadata():
    emb = FakeEmbeddings(dimension=32)
    index = _build_index()
    retriever = Retriever(emb, index)
    results = retriever.retrieve("test query", top_k=2)
    for r in results:
        assert r.document_id == "doc1"
        assert "filename" in r.metadata


def test_retrieve_scores_are_valid():
    emb = FakeEmbeddings(dimension=32)
    index = _build_index()
    retriever = Retriever(emb, index)
    results = retriever.retrieve("test", top_k=3)
    for r in results:
        assert isinstance(r.score, float)
        assert -1.0 <= r.score <= 1.0


def test_empty_query_raises():
    emb = FakeEmbeddings(dimension=32)
    index = _build_index()
    retriever = Retriever(emb, index)
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("", top_k=3)


def test_whitespace_query_raises():
    emb = FakeEmbeddings(dimension=32)
    index = _build_index()
    retriever = Retriever(emb, index)
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("   ", top_k=3)
