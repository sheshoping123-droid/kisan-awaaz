import numpy as np
import pytest

from app.rag.index import FAISSVectorIndex
from app.rag.models import DocumentChunk


def _make_chunks(n: int, doc_id: str = "doc1") -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"chunk-{i}",
            document_id=doc_id,
            text=f"Text for chunk {i} about agriculture.",
            metadata={"chunk_index": i, "filename": "test.txt"},
        )
        for i in range(n)
    ]


def _make_embeddings(n: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed=99)
    vecs = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32)


def test_create_index():
    index = FAISSVectorIndex(dimension=32)
    assert index.dimension == 32
    assert index.size == 0


def test_add_and_size():
    index = FAISSVectorIndex(dimension=32)
    chunks = _make_chunks(5)
    embeddings = _make_embeddings(5)
    index.add(embeddings, chunks)
    assert index.size == 5


def test_search_returns_results():
    index = FAISSVectorIndex(dimension=32)
    chunks = _make_chunks(5)
    embeddings = _make_embeddings(5)
    index.add(embeddings, chunks)

    query = _make_embeddings(1)
    results = index.search(query, top_k=3)
    assert len(results) == 3
    assert all(r.score is not None for r in results)


def test_search_top_k_larger_than_size():
    index = FAISSVectorIndex(dimension=32)
    chunks = _make_chunks(2)
    embeddings = _make_embeddings(2)
    index.add(embeddings, chunks)

    results = index.search(_make_embeddings(1), top_k=10)
    assert len(results) == 2


def test_search_empty_index():
    index = FAISSVectorIndex(dimension=32)
    results = index.search(_make_embeddings(1), top_k=5)
    assert results == []


def test_metadata_preserved():
    index = FAISSVectorIndex(dimension=32)
    chunks = _make_chunks(3)
    embeddings = _make_embeddings(3)
    index.add(embeddings, chunks)

    results = index.search(_make_embeddings(1), top_k=3)
    for r in results:
        assert r.document_id == "doc1"
        assert "filename" in r.metadata
        assert "chunk_index" in r.metadata


def test_dimension_mismatch_on_add():
    index = FAISSVectorIndex(dimension=32)
    bad_embeddings = _make_embeddings(3, dim=16)
    chunks = _make_chunks(3)
    with pytest.raises(ValueError, match="dim"):
        index.add(bad_embeddings, chunks)


def test_count_mismatch_on_add():
    index = FAISSVectorIndex(dimension=32)
    embeddings = _make_embeddings(3)
    chunks = _make_chunks(5)
    with pytest.raises(ValueError, match="count"):
        index.add(embeddings, chunks)


def test_dimension_mismatch_on_search():
    index = FAISSVectorIndex(dimension=32)
    index.add(_make_embeddings(3), _make_chunks(3))
    bad_query = _make_embeddings(1, dim=16)
    with pytest.raises(ValueError, match="dim"):
        index.search(bad_query, top_k=1)


def test_save_and_load(tmp_dir):
    index = FAISSVectorIndex(dimension=32)
    chunks = _make_chunks(5)
    embeddings = _make_embeddings(5)
    index.add(embeddings, chunks)
    index.save(tmp_dir)

    loaded = FAISSVectorIndex.load(tmp_dir)
    assert loaded.size == 5
    assert loaded.dimension == 32

    results = loaded.search(_make_embeddings(1), top_k=3)
    assert len(results) == 3


def test_metadata_survives_persistence(tmp_dir):
    index = FAISSVectorIndex(dimension=32)
    chunks = _make_chunks(3)
    embeddings = _make_embeddings(3)
    index.add(embeddings, chunks)
    index.save(tmp_dir)

    loaded = FAISSVectorIndex.load(tmp_dir)
    results = loaded.search(_make_embeddings(1), top_k=3)
    for r in results:
        assert r.chunk_id.startswith("chunk-")
        assert r.metadata["filename"] == "test.txt"


def test_load_missing_index_raises(tmp_dir):
    with pytest.raises(FileNotFoundError):
        FAISSVectorIndex.load(tmp_dir)


def test_invalid_dimension_raises():
    with pytest.raises(ValueError):
        FAISSVectorIndex(dimension=0)


def test_scores_are_floats():
    index = FAISSVectorIndex(dimension=32)
    index.add(_make_embeddings(5), _make_chunks(5))
    results = index.search(_make_embeddings(1), top_k=3)
    assert all(isinstance(r.score, float) for r in results)
    assert all(-1.0 <= r.score <= 1.0 for r in results)
