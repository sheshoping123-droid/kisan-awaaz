from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation. Swap implementations here."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns array of shape (len(texts), dim)."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns array of shape (1, dim)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model: %s (dim=%d)", model_name, self._dimension)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension
