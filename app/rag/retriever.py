from __future__ import annotations

from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingProvider
from app.rag.index import FAISSVectorIndex
from app.rag.models import RetrievalResult

logger = get_logger(__name__)


class Retriever:
    """Embeds a query and retrieves top-k results from the FAISS index."""

    def __init__(self, embedding_provider: EmbeddingProvider, index: FAISSVectorIndex):
        self._embeddings = embedding_provider
        self._index = index

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        logger.info("Retrieving for query (%d chars), top_k=%d", len(query), top_k)

        query_embedding = self._embeddings.embed_query(query)
        results = self._index.search(query_embedding, top_k=top_k)

        logger.info("Retrieved %d result(s)", len(results))
        return results
