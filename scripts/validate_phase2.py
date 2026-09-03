"""Phase 2 real-model validation: real embeddings, real index, real queries."""

import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from app.core.logging import setup_logging, get_logger
from app.rag.embeddings import SentenceTransformerEmbeddings
from app.rag.service import RAGService
from app.core.config import settings as _settings_instance

setup_logging()
logger = get_logger(__name__)

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RAW_DIR = Path("knowledge/raw")
PROCESSED_DIR = Path("knowledge/processed")
VALIDATION_PROCESSED = Path("knowledge/processed_validation")

QUERIES = [
    "How can wheat rust be controlled?",
    "What are the symptoms of wheat rust?",
    "How do I manage cotton bollworm?",
    "What are the control strategies for cotton bollworm?",
]


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_embedding_provider() -> SentenceTransformerEmbeddings:
    section("1. Embedding Provider Tests")

    embeddings = SentenceTransformerEmbeddings(MODEL)
    print(f"  Model loaded: {MODEL}")
    print(f"  Dimension: {embeddings.dimension}")

    test_text = "Wheat rust is a fungal disease affecting crops."
    vec = embeddings.embed_query(test_text)
    print(f"  embed_query shape: {vec.shape}")
    print(f"  embed_query dtype: {vec.dtype}")
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"
    assert vec.shape == (1, embeddings.dimension), f"Expected (1, {embeddings.dimension}), got {vec.shape}"

    norm = np.linalg.norm(vec)
    print(f"  L2 norm: {norm:.6f}")
    assert abs(norm - 1.0) < 1e-5, f"Not L2-normalized: norm={norm}"

    texts = ["Wheat rust disease", "Cotton bollworm pest", "Crop management tips"]
    vecs = embeddings.embed_documents(texts)
    print(f"  embed_documents shape: {vecs.shape}")
    assert vecs.shape == (3, embeddings.dimension)
    assert vecs.dtype == np.float32

    norms = np.linalg.norm(vecs, axis=1)
    print(f"  L2 norms: {norms}")
    assert all(abs(n - 1.0) < 1e-5 for n in norms), "Not all vectors normalized"

    vec1 = embeddings.embed_documents(["Wheat rust disease"])
    vec2 = embeddings.embed_documents(["Wheat rust disease"])
    assert np.allclose(vec1, vec2), "Determinism check failed"
    print("  Determinism: PASS")

    print("  [PASS] All embedding provider checks passed.")
    return embeddings


def test_build_index(embeddings: SentenceTransformerEmbeddings) -> RAGService:
    section("2. Build Real FAISS Index")

    if VALIDATION_PROCESSED.exists():
        shutil.rmtree(VALIDATION_PROCESSED)

    service = RAGService(
        raw_dir=RAW_DIR,
        processed_dir=VALIDATION_PROCESSED,
        embedding_provider=embeddings,
        chunk_size=512,
        chunk_overlap=50,
    )
    summary = service.index_documents()

    print(f"  Documents processed: {summary['documents']}")
    print(f"  Chunks created:      {summary['chunks']}")
    print(f"  Embedding dimension: {summary['dimension']}")
    print(f"  Index location:      {summary['index_path']}")

    assert summary["documents"] >= 2, f"Expected >=2 documents, got {summary['documents']}"
    assert summary["chunks"] >= 2, f"Expected >=2 chunks, got {summary['chunks']}"
    assert summary["dimension"] == embeddings.dimension

    index_file = VALIDATION_PROCESSED / "index.faiss"
    meta_file = VALIDATION_PROCESSED / "metadata.json"
    assert index_file.exists(), "index.faiss not found"
    assert meta_file.exists(), "metadata.json not found"
    print(f"  index.faiss size: {index_file.stat().st_size:,} bytes")
    print(f"  metadata.json size: {meta_file.stat().st_size:,} bytes")

    with open(meta_file) as f:
        meta = json.load(f)
    print(f"  Metadata keys: {list(meta.keys())}")
    print(f"  Stored chunks: {meta.get('total_vectors', 'N/A')}")

    print("  [PASS] Index build succeeded.")
    return service


def test_semantic_queries(service: RAGService) -> None:
    section("3. Semantic Retrieval Queries")

    for i, query in enumerate(QUERIES, 1):
        print(f"\n  Query {i}: \"{query}\"")
        results = service.retrieve(query, top_k=3)

        assert len(results) > 0, f"No results for query {i}"
        for j, r in enumerate(results, 1):
            text_preview = r.text[:80].replace("\n", " ")
            print(f"    #{j} (score={r.score:.4f}) [{r.document_id[:8]}..] {text_preview}...")

        assert all(0.0 <= r.score <= 1.0 for r in results), "Score out of [0,1] range"
        assert all(isinstance(r.score, float) for r in results), "Score not float"
        print(f"    [PASS] {len(results)} results, scores in [0,1].")


def test_persistence(embeddings: SentenceTransformerEmbeddings) -> None:
    section("4. Persistence (Save -> Reload -> Retrieve)")

    reloaded_service = RAGService(
        raw_dir=RAW_DIR,
        processed_dir=VALIDATION_PROCESSED,
        embedding_provider=embeddings,
    )
    reloaded_service.load_index()
    print("  Index reloaded from disk.")

    query = "How can wheat rust be controlled?"
    results = reloaded_service.retrieve(query, top_k=3)
    print(f"  Query: \"{query}\"")
    for j, r in enumerate(results, 1):
        text_preview = r.text[:80].replace("\n", " ")
        print(f"    #{j} (score={r.score:.4f}) {text_preview}...")

    assert len(results) > 0, "No results after reload"
    print("  [PASS] Persistence round-trip succeeded.")


def test_config_validation() -> None:
    section("5. Configuration Validation")

    settings = _settings_instance
    print(f"  embedding_model: {settings.embedding_model}")
    print(f"  rag_raw_dir: {settings.rag_raw_dir}")
    print(f"  rag_processed_dir: {settings.rag_processed_dir}")
    print(f"  rag_chunk_size: {settings.rag_chunk_size}")
    print(f"  rag_chunk_overlap: {settings.rag_chunk_overlap}")
    print(f"  rag_top_k: {settings.rag_top_k}")

    assert settings.rag_chunk_overlap < settings.rag_chunk_size, (
        f"chunk_overlap ({settings.rag_chunk_overlap}) >= chunk_size ({settings.rag_chunk_size})"
    )
    print("  chunk_overlap < chunk_size: PASS")
    print("  [PASS] Configuration is valid.")


def main() -> None:
    print("\n" + "=" * 60)
    print("  PHASE 2 — REAL-MODEL VALIDATION")
    print("=" * 60)

    embeddings = test_embedding_provider()
    service = test_build_index(embeddings)
    test_semantic_queries(service)
    test_persistence(embeddings)
    test_config_validation()

    section("VALIDATION COMPLETE")
    print("  All checks passed. Phase 2 is ready.")
    print()

    if VALIDATION_PROCESSED.exists():
        shutil.rmtree(VALIDATION_PROCESSED)
        print(f"  Cleaned up: {VALIDATION_PROCESSED}")


if __name__ == "__main__":
    main()
