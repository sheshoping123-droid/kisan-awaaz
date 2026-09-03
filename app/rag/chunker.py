from __future__ import annotations

import hashlib

from app.core.logging import get_logger
from app.rag.models import Document, DocumentChunk

logger = get_logger(__name__)


def _chunk_id(document_id: str, chunk_index: int) -> str:
    raw = f"{document_id}::{chunk_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


class TextChunker:
    """Splits documents into fixed-size overlapping chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        text = document.text
        if not text.strip():
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_chunk_id(document.document_id, index),
                        document_id=document.document_id,
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_index": index,
                        },
                    )
                )
                index += 1

            if end >= len(text):
                break

            start = end - self.chunk_overlap

        logger.info(
            "Chunked document %s into %d chunk(s) (size=%d, overlap=%d)",
            document.document_id,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Total chunks: %d from %d document(s)", len(all_chunks), len(documents))
        return all_chunks
