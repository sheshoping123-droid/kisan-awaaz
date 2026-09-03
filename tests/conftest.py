import tempfile
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import Document, DocumentChunk


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class FakeEmbeddings(EmbeddingProvider):
    """Deterministic mock embedding provider for unit tests. No network needed."""

    def __init__(self, dimension: int = 32):
        self._dimension = dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = np.empty((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed=seed)
            embeddings[i] = rng.random(self._dimension, dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (embeddings / norms).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])

    @property
    def dimension(self) -> int:
        return self._dimension


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings(dimension=32)


@pytest.fixture
def sample_document():
    return Document(
        document_id="test-doc-001",
        source="test/wheat.txt",
        text="Wheat rust is a destructive disease. Yellow rust appears as yellow-orange pustules. Brown rust shows orange-brown pustules on leaves.",
        metadata={"filename": "wheat.txt", "path": "test/wheat.txt"},
    )


@pytest.fixture
def sample_chunks(sample_document):
    return [
        DocumentChunk(
            chunk_id="chunk-001",
            document_id=sample_document.document_id,
            text="Wheat rust is a destructive disease.",
            metadata={**sample_document.metadata, "chunk_index": 0},
        ),
        DocumentChunk(
            chunk_id="chunk-002",
            document_id=sample_document.document_id,
            text="Yellow rust appears as yellow-orange pustules.",
            metadata={**sample_document.metadata, "chunk_index": 1},
        ),
        DocumentChunk(
            chunk_id="chunk-003",
            document_id=sample_document.document_id,
            text="Brown rust shows orange-brown pustules on leaves.",
            metadata={**sample_document.metadata, "chunk_index": 2},
        ),
    ]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_raw_dir(tmp_dir):
    raw = tmp_dir / "raw"
    raw.mkdir()
    (raw / "doc1.txt").write_text("Wheat rust is a disease affecting crops in Pakistan.", encoding="utf-8")
    (raw / "doc2.md").write_text("# Cotton Pests\nCotton bollworm damages bolls.", encoding="utf-8")
    return raw


@pytest.fixture
def empty_raw_dir(tmp_dir):
    raw = tmp_dir / "empty_raw"
    raw.mkdir()
    return raw
