"""Heading-aware chunking.

One chunk per section when the section fits in MAX_CHUNK_TOKENS (most do);
longer sections are split on block boundaries. Tables stay whole, or are split
by rows with the header row repeated, so a value is never separated from its
column names.
"""
from __future__ import annotations

from itertools import groupby

from ..retrieval.tokens import count_tokens
from .models import Block, Chunk, ParsedDoc
from .parsers import table_to_markdown

MAX_CHUNK_TOKENS = 350


def _split_table(block: Block) -> list[str]:
    rows = block.rows or []
    if len(rows) < 3 or count_tokens(block.text) <= MAX_CHUNK_TOKENS:
        return [block.text]
    header, body = rows[0], rows[1:]
    parts, current = [], []
    for row in body:
        candidate = table_to_markdown([header, *current, row])
        if current and count_tokens(candidate) > MAX_CHUNK_TOKENS:
            parts.append(table_to_markdown([header, *current]))
            current = []
        current.append(row)
    if current:
        parts.append(table_to_markdown([header, *current]))
    return parts


def _locator(doc: ParsedDoc, heading: str | None, page: int | None) -> str:
    if heading:
        return heading
    if page is not None:
        return f"p. {page}"
    return doc.meta.title if doc.has_headings else "p. 1"


def build_chunks(doc: ParsedDoc) -> list[Chunk]:
    chunks: list[Chunk] = []
    title = doc.meta.title

    def section_key(block: Block):
        # Documents without headings are cited by page, so group them by page.
        return (block.heading, None if (block.heading or doc.has_headings) else block.page)

    for (heading, page), group in groupby(doc.blocks, key=section_key):
        blocks = list(group)
        first_page = page if page is not None else blocks[0].page
        locator = _locator(doc, heading, first_page)
        prefix = f"## {heading}\n" if heading else ""
        pieces: list[str] = []
        for block in blocks:
            pieces.extend(_split_table(block) if block.kind == "table" else [block.text])

        current: list[str] = []
        for piece in pieces:
            candidate = prefix + "\n".join([*current, piece])
            if current and count_tokens(candidate) > MAX_CHUNK_TOKENS:
                chunks.append(_make_chunk(doc, len(chunks), locator, heading, prefix, current, title))
                current = []
            current.append(piece)
        if current:
            chunks.append(_make_chunk(doc, len(chunks), locator, heading, prefix, current, title))
    return chunks


def _make_chunk(doc, n, locator, heading, prefix, pieces, title) -> Chunk:
    text = prefix + "\n".join(pieces)
    return Chunk(
        chunk_id=f"{doc.meta.doc_id}#{n}",
        doc_id=doc.meta.doc_id,
        locator=locator,
        heading=heading,
        text=text,
        index_text=f"{title}\n{text}",
        tokens=count_tokens(text),
    )
