"""Extract office-closure dates from calendar documents' tables at ingestion time.

A table counts as a holiday table when it has a column of dates. Tables whose
heading or header row says the days are restricted, optional or floating are
skipped, because the office stays open on those days.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from .models import Holiday, ParsedDoc

_DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%Y-%m-%d",
                 "%d/%m/%Y", "%d.%m.%Y", "%A, %d %B %Y", "%A %d %B %Y")
_NOT_CLOSURE = re.compile(r"\b(restricted|optional|floating)\b", re.IGNORECASE)


def parse_date(text: str) -> date | None:
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def extract_holidays(doc: ParsedDoc) -> list[Holiday]:
    if doc.meta.doc_type != "calendar":
        return []
    found: dict[date, Holiday] = {}
    for block in doc.blocks:
        if block.kind != "table" or not block.rows or len(block.rows) < 2:
            continue
        header = block.rows[0]
        if _NOT_CLOSURE.search(" ".join(header)) or _NOT_CLOSURE.search(block.heading or ""):
            continue
        body = block.rows[1:]
        width = len(header)
        date_col = next((i for i, h in enumerate(header) if "date" in h.lower()), None)
        if date_col is None:
            # No labelled column: use the column where most cells parse as dates.
            scores = [sum(parse_date(r[i]) is not None for r in body if i < len(r)) for i in range(width)]
            if not scores or max(scores) < len(body) / 2:
                continue
            date_col = scores.index(max(scores))
        name_col = next((i for i, h in enumerate(header) if re.search(r"holiday|name|occasion", h, re.I)), width - 1)
        for row in body:
            if date_col >= len(row):
                continue
            day = parse_date(row[date_col])
            if day is None:
                continue
            name = row[name_col] if name_col < len(row) else ""
            found[day] = Holiday(doc_id=doc.meta.doc_id, day=day, name=name)
    return sorted(found.values(), key=lambda h: h.day)
