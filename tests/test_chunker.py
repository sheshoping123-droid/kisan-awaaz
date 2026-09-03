import pytest
from app.rag.chunker import TextChunker
from app.rag.models import Document


def test_basic_chunking():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    doc = Document(
        document_id="d1",
        source="test.txt",
        text="abcdefghij" * 5,  # 50 chars
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    assert all(c.document_id == "d1" for c in chunks)


def test_deterministic_chunk_ids():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    doc = Document(document_id="d1", source="test.txt", text="abcdefghij" * 5)
    chunks_a = chunker.chunk_document(doc)
    chunks_b = chunker.chunk_document(doc)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_overlap_preserved():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    doc = Document(document_id="d1", source="test.txt", text="A" * 35)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2


def test_chunk_metadata_inherits_document_metadata(sample_document):
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(sample_document)
    for chunk in chunks:
        assert "filename" in chunk.metadata
        assert "chunk_index" in chunk.metadata


def test_chunk_index_sequential(sample_document):
    chunker = TextChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk_document(sample_document)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_single_chunk_for_short_text():
    chunker = TextChunker(chunk_size=1000, chunk_overlap=50)
    doc = Document(document_id="d1", source="t.txt", text="Short text.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1


def test_empty_document_returns_no_chunks():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    doc = Document(document_id="d1", source="t.txt", text="   ")
    chunks = chunker.chunk_document(doc)
    assert chunks == []


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, chunk_overlap=0)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_documents_batch():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    docs = [
        Document(document_id="d1", source="a.txt", text="X" * 50),
        Document(document_id="d2", source="b.txt", text="Y" * 30),
    ]
    chunks = chunker.chunk_documents(docs)
    doc_ids = {c.document_id for c in chunks}
    assert doc_ids == {"d1", "d2"}
