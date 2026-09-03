from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger
from app.rag.chunker import TextChunker
from app.rag.embeddings import EmbeddingProvider
from app.rag.index import FAISSVectorIndex
from app.rag.loaders import DocumentLoader
from app.rag.models import RetrievalResult
from app.rag.retriever import Retriever

logger = get_logger(__name__)


class RAGService:
    """Coordinates document indexing and query retrieval.

    Indexing: loader → chunker → embeddings → FAISS index
    Retrieval: query → embeddings → FAISS search → results
    """

    def __init__(
        self,
        raw_dir: str | Path,
        processed_dir: str | Path,
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self._raw_dir = Path(raw_dir)
        self._processed_dir = Path(processed_dir)
        self._embeddings = embedding_provider
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._index: FAISSVectorIndex | None = None
        self._retriever: Retriever | None = None

    def index_documents(self) -> dict:
        loader = DocumentLoader(self._raw_dir)
        documents = loader.load_all()

        if not documents:
            logger.warning("No documents found in %s", self._raw_dir)
            return {"documents": 0, "chunks": 0, "dimension": self._embeddings.dimension}

        chunks = self._chunker.chunk_documents(documents)
        texts = [c.text for c in chunks]

        logger.info("Generating embeddings for %d chunks...", len(chunks))
        embeddings = self._embeddings.embed_documents(texts)

        index = FAISSVectorIndex(dimension=self._embeddings.dimension)
        index.add(embeddings, chunks)
        index.save(self._processed_dir)

        self._index = index
        self._retriever = Retriever(self._embeddings, index)

        summary = {
            "documents": len(documents),
            "chunks": len(chunks),
            "dimension": self._embeddings.dimension,
            "index_path": str(self._processed_dir),
        }
        logger.info("Indexing complete: %s", summary)
        return summary

    def load_index(self) -> None:
        self._index = FAISSVectorIndex.load(self._processed_dir)
        self._retriever = Retriever(self._embeddings, self._index)
        logger.info("Loaded existing index with %d vectors", self._index.size)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._retriever is None:
            self.load_index()
        assert self._retriever is not None
        return self._retriever.retrieve(query, top_k=top_k)
