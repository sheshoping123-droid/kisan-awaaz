from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.logging import get_logger
from app.rag.models import Document

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def _deterministic_id(source: str) -> str:
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:12]


class DocumentLoader:
    """Discovers and loads text-based documents from a directory."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    def discover(self) -> list[Path]:
        if not self.raw_dir.exists():
            logger.warning("Knowledge raw directory does not exist: %s", self.raw_dir)
            return []

        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(self.raw_dir.rglob(f"*{ext}"))

        files.sort(key=lambda p: str(p.relative_to(self.raw_dir)))
        logger.info("Discovered %d document(s) in %s", len(files), self.raw_dir)
        return files

    def load_file(self, path: Path) -> Document:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported document type: {path.suffix}")

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            raise OSError(f"Cannot read document: {path}") from e

        if not text.strip():
            raise ValueError(f"Document is empty: {path}")

        rel_path = str(path.relative_to(self.raw_dir))
        doc_id = _deterministic_id(rel_path)

        logger.info("Loaded document: %s (id=%s, %d chars)", rel_path, doc_id, len(text))
        return Document(
            document_id=doc_id,
            source=rel_path,
            text=text,
            metadata={
                "filename": path.name,
                "path": rel_path,
                "extension": path.suffix,
            },
        )

    def load_all(self) -> list[Document]:
        documents = []
        for path in self.discover():
            try:
                documents.append(self.load_file(path))
            except (ValueError, OSError) as e:
                logger.error("Skipping document %s: %s", path, e)
        logger.info("Loaded %d document(s) total", len(documents))
        return documents
