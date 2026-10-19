"""Data structures shared by ingestion, retrieval and tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    file: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str

    @classmethod
    def from_manifest(cls, entry: dict) -> "DocMeta":
        return cls(
            doc_id=entry["doc_id"],
            file=entry["file"],
            title=entry["title"],
            doc_type=entry["doc_type"].lower(),
            regions=tuple(r.upper() for r in entry["regions"]),
            effective_date=date.fromisoformat(entry["effective_date"]),
            access_level=entry["access_level"].lower(),
        )


@dataclass
class Block:
    """One paragraph or table, in reading order, with its nearest heading."""

    kind: str  # "text" | "table"
    text: str
    heading: str | None
    page: int | None = None
    rows: list[list[str]] | None = None
    ocr: bool = False


@dataclass
class ParsedDoc:
    meta: DocMeta
    blocks: list[Block]
    has_headings: bool
    ocr_pages: int = 0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    locator: str
    heading: str | None
    text: str  # text shown to the LLM (heading line + body)
    index_text: str  # text used for BM25 and embeddings (adds the doc title)
    tokens: int = 0


@dataclass(frozen=True)
class Holiday:
    doc_id: str
    day: date
    name: str


@dataclass
class IngestionReport:
    documents: int = 0
    chunks: int = 0
    ocr_pages: int = 0
    holidays: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    dense_enabled: bool = False
