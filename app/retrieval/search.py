"""Policy search used by the agent: filter, rank, pack within the token budget.

Order of operations is the access-control guarantee: documents the requester
cannot read, or that do not cover their region, are removed *before* scoring,
so their text can never reach the LLM.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import date

from ..ingestion.models import Chunk, DocMeta
from ..ingestion.pipeline import KnowledgeBase
from .eligibility import is_eligible, is_readable
from .index import tokenize
from .precedence import blockers_of_upcoming, order_by_precedence
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
        per_doc: dict[str, int] = {}
        ref_no = first_ref

        def try_add(i: int, upcoming: bool) -> None:
            nonlocal ref_no
            chunk = self.kb.chunks[i]
            if per_doc.get(chunk.doc_id, 0) >= MAX_CHUNKS_PER_DOC:
                return
            if any(_jaccard(self._chunk_tokens[i], self._chunk_tokens[self.kb.chunk_pos[h.chunk.chunk_id]])
                   >= _DUPLICATE_JACCARD for h in selected):
                return  # near-duplicate of something already selected
            hit = SearchHit(ref=f"C{ref_no}", chunk=chunk, doc=docs[chunk.doc_id], upcoming=upcoming)
            trial = self._render([*selected, hit], reference_date)
            if count_tokens(trial) > budget:
                return
            selected.append(hit)
            per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
            ref_no += 1

        for i, _score in ranked:
            try_add(i, upcoming=False)
        for i in future_ranked:
            try_add(i, upcoming=True)

        if not selected:
            text = ("<search_results>No eligible document matched this query. If no other search finds it, "
                    "the documents do not cover the question.</search_results>")
            return SearchResult(text=text, tokens=count_tokens(text), hits=[])
        text = self._render(selected, reference_date)
        return SearchResult(text=text, tokens=count_tokens(text), hits=selected)

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
