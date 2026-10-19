"""Section 5.1 rule 2 (precedence): authority, then scope, then recency.

Code computes the ordering; the LLM only decides which statements describe the
same rule. Documents with equal keys are tied, which is an unresolved conflict
if they give different values.
"""
from __future__ import annotations

from datetime import date
from itertools import groupby

from ..ingestion.models import DocMeta

PRIMARY_TYPES = frozenset({"policy", "addendum", "memo", "handbook", "calendar"})


def authority(doc: DocMeta) -> int:
    """1 for authoritative documents, 0 for faq/wiki (and any unknown type)."""
    return 1 if doc.doc_type in PRIMARY_TYPES else 0


def scope(doc: DocMeta) -> int:
    """1 for documents written for specific regions, 0 for GLOBAL ones."""
    return 0 if "GLOBAL" in doc.regions else 1


def precedence_key(doc: DocMeta) -> tuple[int, int, date]:
    return (authority(doc), scope(doc), doc.effective_date)


def outranks(a: DocMeta, b: DocMeta) -> bool:
    return precedence_key(a) > precedence_key(b)


def order_by_precedence(docs: list[DocMeta]) -> list[list[DocMeta]]:
    """Group documents into tiers, highest precedence first; each tier is a tie."""
    ordered = sorted(docs, key=lambda d: (precedence_key(d), d.doc_id), reverse=True)
    return [list(g) for _, g in groupby(ordered, key=precedence_key)]


def winner(docs: list[DocMeta]) -> list[DocMeta]:
    """The top tier (one document, or several if tied)."""
    tiers = order_by_precedence(docs)
    return tiers[0] if tiers else []


def blockers_of_upcoming(future: DocMeta, current: list[DocMeta]) -> list[DocMeta]:
    """Current documents that would still beat `future` once it takes effect.

    A future document always wins on recency, so it can only lose on authority
    or on scope (Section 5.1 "Upcoming changes").
    """
    out = []
    for doc in current:
        if authority(doc) > authority(future):
            out.append(doc)
        elif authority(doc) == authority(future) and scope(doc) > scope(future):
            out.append(doc)
    return out
