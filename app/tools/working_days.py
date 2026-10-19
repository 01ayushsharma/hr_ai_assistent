"""Working-day calculation (Section 6.2).

A working day is Monday to Friday and not a public holiday for the
requester's region. For each date, the holiday calendar is the one that
applies to the requester under Section 5 with that date as the reference date:
eligible calendars (readable, for their region, already in effect) ranked by
precedence. A tie between calendars is resolved by taking all their holidays.
"""
from __future__ import annotations

from datetime import date, timedelta

from ..ingestion.pipeline import KnowledgeBase
from ..retrieval.eligibility import is_eligible
from ..retrieval.precedence import winner


def daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def count_working_days(start: date, end: date, holidays: set[date]) -> int:
    return sum(1 for d in daterange(start, end) if d.weekday() < 5 and d not in holidays)


def applicable_calendars(kb: KnowledgeBase, role: str, region: str, on: date) -> list[str]:
    calendars = [d for d in kb.docs.values()
                 if d.doc_type == "calendar" and is_eligible(d, role, region, on)]
    return [d.doc_id for d in winner(calendars)]


def holidays_in_range(kb: KnowledgeBase, role: str, region: str, start: date, end: date) -> dict[date, str]:
    """Public holidays (date -> name) that apply to the requester between start and end."""
    out: dict[date, str] = {}
    cache: dict[tuple[str, ...], dict[date, str]] = {}
    for day in daterange(start, end):
        key = tuple(applicable_calendars(kb, role, region, day))
        if key not in cache:
            cache[key] = {h.day: h.name for doc_id in key for h in kb.holidays.get(doc_id, [])}
        if day in cache[key]:
            out[day] = cache[key][day]
    return out
