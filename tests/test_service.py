import pytest

from app.rag.service import RAGService
from tests.conftest import FakeEmbeddings


def test_index_documents(sample_raw_dir, tmp_dir):
    emb = FakeEmbeddings(dimension=32)
    processed = tmp_dir / "processed"
    service = RAGService(
        raw_dir=sample_raw_dir,
        processed_dir=processed,
        embedding_provider=emb,
        chunk_size=50,
        chunk_overlap=10,
    )
    summary = service.index_documents()
    assert summary["documents"] == 2
    assert summary["chunks"] > 0
    assert summary["dimension"] == 32
    assert (processed / "index.faiss").exists()
    assert (processed / "metadata.json").exists()


def test_retrieve_after_indexing(sample_raw_dir, tmp_dir):
    emb = FakeEmbeddings(dimension=32)
    processed = tmp_dir / "processed"
    service = RAGService(
        raw_dir=sample_raw_dir,
        processed_dir=processed,
        embedding_provider=emb,
        chunk_size=50,
        chunk_overlap=10,
    )
    service.index_documents()
    results = service.retrieve("wheat rust disease", top_k=3)
    assert len(results) > 0
    assert all(r.text for r in results)
    assert all(r.metadata.get("filename") for r in results)


def test_load_index(sample_raw_dir, tmp_dir):
    emb = FakeEmbeddings(dimension=32)
    processed = tmp_dir / "processed"
    service = RAGService(
        raw_dir=sample_raw_dir,
        processed_dir=processed,
        embedding_provider=emb,
        chunk_size=50,
        chunk_overlap=10,
    )
    service.index_documents()

    service2 = RAGService(
        raw_dir=sample_raw_dir,
        processed_dir=processed,
        embedding_provider=emb,
        chunk_size=50,
        chunk_overlap=10,
    )
    service2.load_index()
    results = service2.retrieve("cotton pests", top_k=2)
    assert len(results) == 2


def test_empty_knowledge_base(empty_raw_dir, tmp_dir):
    emb = FakeEmbeddings(dimension=32)
    processed = tmp_dir / "processed"
    service = RAGService(
        raw_dir=empty_raw_dir,
        processed_dir=processed,
        embedding_provider=emb,
    )
    summary = service.index_documents()
    assert summary["documents"] == 0
    assert summary["chunks"] == 0


def test_retrieve_without_index_raises(sample_raw_dir, tmp_dir):
    emb = FakeEmbeddings(dimension=32)
    processed = tmp_dir / "no_index"
    service = RAGService(
        raw_dir=sample_raw_dir,
        processed_dir=processed,
        embedding_provider=emb,
    )
    with pytest.raises(FileNotFoundError):
        service.retrieve("test query")
