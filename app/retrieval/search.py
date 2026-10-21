"""Policy search used by the agent: filter, rank, pack within the token budget.

Order of operations is the access-control guarantee: documents the requester
cannot read, or that do not cover their region, are removed *before* scoring,
so their text can never reach the LLM.
"""
from __future__ import annotations

import dataclasses
import html
import re
from dataclasses import dataclass, field
from datetime import date

from ..ingestion.models import Chunk, DocMeta
from ..ingestion.pipeline import KnowledgeBase
from .eligibility import is_eligible, is_readable
from .index import tokenize
from .precedence import blockers_of_upcoming, order_by_precedence, precedence_key
from .tokens import count_tokens

CORPUS_TOKEN_LIMIT = 3000  # Section 4.3: per LLM call
PER_SEARCH_LIMIT = 1800
MAX_CHUNKS_PER_DOC = 3
MAX_UPCOMING_CHUNKS = 2
_DUPLICATE_JACCARD = 0.8


@dataclass
class SearchHit:
    ref: str
    chunk: Chunk
    doc: DocMeta
    upcoming: bool = False


@dataclass
class SearchResult:
    text: str
    tokens: int
    hits: list[SearchHit] = field(default_factory=list)


def section_key(heading: str | None) -> str:
    """'6. Carrying annual leave forward' and '1. Annual leave (Urlaub)' -> comparable section names."""
    if not heading:
        return ""
    text = re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", heading.lower())
    text = re.sub(r"\([^)]*\)", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def _format_chunk(hit: SearchHit, blockers: list[DocMeta]) -> str:
    doc, chunk = hit.doc, hit.chunk
    attrs = [
        f'ref="{hit.ref}"',
        f'doc_id="{doc.doc_id}"',
        f'title="{html.escape(doc.title)}"',
        f'type="{doc.doc_type}"',
        f'regions="{",".join(doc.regions)}"',
        f'effective="{doc.effective_date.isoformat()}"',
        f'section="{html.escape(chunk.locator)}"',
    ]
    if hit.upcoming:
        attrs.append(f'status="UPCOMING - not in effect until {doc.effective_date.isoformat()}"')
        if blockers:
            attrs.append(f'still_outranked_by="{",".join(d.doc_id for d in blockers)}"')
    # Escape markup so document text cannot close the tag or fake a new one.
    body = chunk.text.replace("<", "&lt;").replace(">", "&gt;")
    return f"<chunk {' '.join(attrs)}>\n{body}\n</chunk>"


class Retriever:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self._chunk_tokens = [set(tokenize(c.text)) for c in kb.chunks]
        # Section name -> chunks, for pulling in the matching section of higher-precedence documents.
        self._by_section: dict[str, list[int]] = {}
        for i, c in enumerate(kb.chunks):
            key = section_key(c.heading)
            if key:
                self._by_section.setdefault(key, []).append(i)

    def search(self, query: str, *, role: str, region: str, reference_date: date, today: date,
               token_budget: int, first_ref: int = 1) -> SearchResult:
        docs = self.kb.docs
        eligible_ids = {d.doc_id for d in docs.values() if is_eligible(d, role, region, reference_date)}
        upcoming_ids: set[str] = set()
        if reference_date >= today:
            upcoming_ids = {d.doc_id for d in docs.values()
                            if is_readable(d, role, region) and d.effective_date > today}

        current_idx = [i for i, c in enumerate(self.kb.chunks) if c.doc_id in eligible_ids]
        future_idx = [i for i, c in enumerate(self.kb.chunks) if c.doc_id in upcoming_ids]
        ranked = self.kb.index.search(query, current_idx, top_k=10)
        joint = self.kb.index.search(query, current_idx + future_idx, top_k=6) if future_idx else []
        future_ranked = [i for i, _ in joint if i in set(future_idx)][:MAX_UPCOMING_CHUNKS]

        budget = min(token_budget, PER_SEARCH_LIMIT)
        selected: list[SearchHit] = []

        def numbered(hits: list[SearchHit]) -> list[SearchHit]:
            return [dataclasses.replace(h, ref=f"C{first_ref + k}") for k, h in enumerate(hits)]

        def try_add(i: int, upcoming: bool) -> None:
            chunk = self.kb.chunks[i]
            doc = docs[chunk.doc_id]
            if sum(h.chunk.doc_id == chunk.doc_id for h in selected) >= MAX_CHUNKS_PER_DOC:
                return
            candidate = list(selected)
            for hit in selected:
                if hit.upcoming != upcoming:
                    continue
                if _jaccard(self._chunk_tokens[i], self._chunk_tokens[self.kb.chunk_pos[hit.chunk.chunk_id]]) \
                        < _DUPLICATE_JACCARD:
                    continue
                # Near-duplicates (e.g. the same section in two policy versions) can differ in one
                # value. Keep the higher-precedence copy whatever its rank; keep both when tied,
                # because tied documents that disagree must be reported as a conflict.
                mine, theirs = precedence_key(doc), precedence_key(hit.doc)
                if mine < theirs:
                    return
                if mine > theirs:
                    candidate.remove(hit)
            candidate.append(SearchHit(ref="", chunk=chunk, doc=doc, upcoming=upcoming))
            if count_tokens(self._render(numbered(candidate), reference_date)) > budget:
                return
            selected[:] = candidate

        for i, _score in ranked:
            try_add(i, upcoming=False)
        # A regional addendum usually overrides a global rule in a section with the same name
        # ("Carrying annual leave forward"). When a section is selected, add the same-named section
        # of any higher-precedence eligible document, so the model sees the overriding rule even
        # if it ranked lower.
        for hit in list(selected):
            for j in self._by_section.get(section_key(hit.chunk.heading), []):
                other = self.kb.chunks[j]
                if (other.doc_id in eligible_ids and other.doc_id != hit.doc.doc_id
                        and precedence_key(docs[other.doc_id]) > precedence_key(hit.doc)
                        and all(h.chunk.chunk_id != other.chunk_id for h in selected)):
                    try_add(j, upcoming=False)
        for i in future_ranked:
            try_add(i, upcoming=True)

        if not selected:
            text = ("<search_results>No eligible document matched this query. If no other search finds it, "
                    "the documents do not cover the question.</search_results>")
            return SearchResult(text=text, tokens=count_tokens(text), hits=[])
        hits = numbered(selected)
        text = self._render(hits, reference_date)
        return SearchResult(text=text, tokens=count_tokens(text), hits=hits)

    def _render(self, hits: list[SearchHit], reference_date: date) -> str:
        current = {h.doc.doc_id: h.doc for h in hits if not h.upcoming}
        tiers = order_by_precedence(list(current.values()))
        order = " > ".join(" = ".join(d.doc_id for d in tier) for tier in tiers)
        lines = [f'<search_results reference_date="{reference_date.isoformat()}">',
                 f"Precedence of the in-effect documents below, highest first ('=' means tied): {order or 'n/a'}"]
        for hit in hits:
            blockers = blockers_of_upcoming(hit.doc, list(current.values())) if hit.upcoming else []
            lines.append(_format_chunk(hit, blockers))
        lines.append("</search_results>")
        return "\n".join(lines)
