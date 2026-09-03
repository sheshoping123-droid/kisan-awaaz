from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    document_id: str
    source: str
    text: str
    metadata: dict[str, str | int | None] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, str | int | None] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, str | int | None] = field(default_factory=dict)
