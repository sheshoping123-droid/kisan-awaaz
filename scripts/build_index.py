"""Build the FAISS index from documents in knowledge/raw/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import setup_logging, get_logger
from app.rag.embeddings import SentenceTransformerEmbeddings
from app.rag.service import RAGService

setup_logging()
logger = get_logger(__name__)

RAW_DIR = Path("knowledge/raw")
PROCESSED_DIR = Path("knowledge/processed")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def main() -> None:
    logger.info("Building RAG index...")
    logger.info("  raw_dir=%s", RAW_DIR)
    logger.info("  processed_dir=%s", PROCESSED_DIR)
    logger.info("  model=%s", EMBEDDING_MODEL)
    logger.info("  chunk_size=%d, overlap=%d", CHUNK_SIZE, CHUNK_OVERLAP)

    embeddings = SentenceTransformerEmbeddings(EMBEDDING_MODEL)
    service = RAGService(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        embedding_provider=embeddings,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    summary = service.index_documents()

    print("\n=== Index Build Summary ===")
    print(f"  Documents processed: {summary['documents']}")
    print(f"  Chunks created:      {summary['chunks']}")
    print(f"  Embedding dimension: {summary['dimension']}")
    print(f"  Index location:      {summary['index_path']}")


if __name__ == "__main__":
    main()
