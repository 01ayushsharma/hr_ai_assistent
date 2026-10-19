"""Section 5.1 rule 1 (eligibility) and the document part of Section 3.1 (roles)."""
from __future__ import annotations

from datetime import date

from ..ingestion.models import DocMeta

ROLE_DOC_ACCESS: dict[str, frozenset[str]] = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "hr": frozenset({"all", "managers", "hr"}),
}


def can_read(role: str, access_level: str) -> bool:
    return access_level in ROLE_DOC_ACCESS.get(role, frozenset())


def covers_region(doc: DocMeta, region: str) -> bool:
    return "GLOBAL" in doc.regions or region.upper() in doc.regions


def is_readable(doc: DocMeta, role: str, region: str) -> bool:
    """Readable and for the requester's region, on any date."""
    return can_read(role, doc.access_level) and covers_region(doc, region)


def is_eligible(doc: DocMeta, role: str, region: str, reference_date: date) -> bool:
    return is_readable(doc, role, region) and doc.effective_date <= reference_date
