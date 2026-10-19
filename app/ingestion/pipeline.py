"""Automatic ingestion: manifest + corpus -> chunks, hybrid index and holiday table."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date

from ..config import Settings
from ..logging_setup import log_event
from ..retrieval.index import HybridIndex, load_embedder
from .chunker import build_chunks
from .holidays import extract_holidays
from .models import Chunk, DocMeta, Holiday, IngestionReport, ParsedDoc
from .ocr import repair_spacing
from .parsers import parse_file

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBase:
    docs: dict[str, DocMeta]
    chunks: list[Chunk]
    holidays: dict[str, list[Holiday]]  # calendar doc_id -> closure days
    index: HybridIndex
    report: IngestionReport
    chunk_pos: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunk_pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}


def load_manifest(path) -> list[DocMeta]:
    with open(path, encoding="utf-8") as fh:
        return [DocMeta.from_manifest(entry) for entry in json.load(fh)]


def build_knowledge_base(settings: Settings, embedder=None, use_embeddings: bool = True) -> KnowledgeBase:
    started = time.monotonic()
    report = IngestionReport()
    parsed: list[ParsedDoc] = []
    for meta in load_manifest(settings.manifest_path):
        path = settings.corpus_dir / meta.file
        try:
            blocks, has_headings, ocr_pages = parse_file(path)
        except Exception as exc:
            report.failures.append(meta.doc_id)
            log_event(logger, "ingest_failed", logging.ERROR, doc_id=meta.doc_id, error=type(exc).__name__)
            continue
        parsed.append(ParsedDoc(meta=meta, blocks=blocks, has_headings=has_headings, ocr_pages=ocr_pages))
        report.ocr_pages += ocr_pages

    # OCR spacing repair uses the vocabulary of the documents that had a text layer.
    vocab: set[str] = set()
    for doc in parsed:
        for block in doc.blocks:
            if not block.ocr:
                vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", block.text))
    for doc in parsed:
        for block in doc.blocks:
            if block.ocr:
                block.text = repair_spacing(block.text, vocab)

    chunks: list[Chunk] = []
    holidays: dict[str, list[Holiday]] = {}
    for doc in parsed:
        doc_chunks = build_chunks(doc)
        chunks.extend(doc_chunks)
        found = extract_holidays(doc)
        if doc.meta.doc_type == "calendar":
            holidays[doc.meta.doc_id] = found
            report.holidays[doc.meta.doc_id] = len(found)
        log_event(logger, "ingested_document", doc_id=doc.meta.doc_id, chunks=len(doc_chunks),
                  ocr_pages=doc.ocr_pages, has_headings=doc.has_headings)

    if use_embeddings and embedder is None:
        embedder = load_embedder(settings.embedding_model, settings.model_cache_dir)
    index = HybridIndex([c.index_text for c in chunks], embedder if use_embeddings else None)

    report.documents = len(parsed)
    report.chunks = len(chunks)
    report.dense_enabled = index.dense_enabled
    log_event(logger, "ingestion_complete", documents=report.documents, chunks=report.chunks,
              ocr_pages=report.ocr_pages, holidays=report.holidays, failures=report.failures,
              dense_retrieval=report.dense_enabled, seconds=round(time.monotonic() - started, 1))
    return KnowledgeBase(docs={d.meta.doc_id: d.meta for d in parsed}, chunks=chunks,
                         holidays=holidays, index=index, report=report)


def holiday_dates(kb: KnowledgeBase, doc_id: str) -> set[date]:
    return {h.day for h in kb.holidays.get(doc_id, [])}
