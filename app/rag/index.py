from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.core.logging import get_logger
from app.rag.models import DocumentChunk, RetrievalResult

logger = get_logger(__name__)

_METADATA_FILE = "metadata.json"


class FAISSVectorIndex:
    """Local FAISS-backed vector index using cosine similarity (normalized IP)."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[dict] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) != chunk count ({len(chunks)})"
            )
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dim ({embeddings.shape[1]}) != index dim ({self._dimension})"
            )

        vectors = embeddings.astype(np.float32)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)

        for chunk in chunks:
            self._chunks.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })

        logger.info("Added %d vectors to index (total=%d)", len(chunks), self.size)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if self.size == 0:
            return []

        if query_embedding.shape[1] != self._dimension:
            raise ValueError(
                f"Query dim ({query_embedding.shape[1]}) != index dim ({self._dimension})"
            )

        k = min(top_k, self.size)
        query_vec = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_data = self._chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_data["chunk_id"],
                    document_id=chunk_data["document_id"],
                    text=chunk_data["text"],
                    score=float(score),
                    metadata=chunk_data["metadata"],
                )
            )

        logger.info("Search returned %d result(s) for top_k=%d", len(results), top_k)
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        metadata_path = directory / _METADATA_FILE

        faiss.write_index(self._index, str(index_path))

        payload = {
            "dimension": self._dimension,
            "total_vectors": self.size,
            "chunks": self._chunks,
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Saved index (%d vectors) to %s", self.size, directory)

    @classmethod
    def load(cls, directory: str | Path) -> FAISSVectorIndex:
        directory = Path(directory)
        index_path = directory / "index.faiss"
        metadata_path = directory / _METADATA_FILE

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Index metadata not found: {metadata_path}")

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        dimension = payload["dimension"]

        instance = cls(dimension=dimension)
        instance._index = faiss.read_index(str(index_path))
        instance._chunks = payload["chunks"]

        if instance._index.ntotal != len(instance._chunks):
            raise ValueError(
                f"Index vector count ({instance._index.ntotal}) != "
                f"metadata count ({len(instance._chunks)})"
            )

        logger.info("Loaded index (%d vectors) from %s", instance.size, directory)
        return instance
