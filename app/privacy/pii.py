"""Personal-data rules for answers (Section 7.4).

Answers never contain phone numbers, show email addresses only masked
(first character + *** + domain), and show names only of employees the
requester may see. The agent never receives phone numbers or full email
addresses in the first place; this filter is the last line of defence.
"""
from __future__ import annotations

import re
import sqlite3

from ..db.repository import Employee
from ..tools.permissions import visible_employee_ids

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}" if domain else "***"


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


class AnswerScrubber:
    def __init__(self, conn: sqlite3.Connection, requester: Employee) -> None:
        rows = conn.execute("SELECT employee_id, full_name, email, phone FROM employees").fetchall()
        visible = visible_employee_ids(conn, requester)
        self.emails = {r["email"].lower() for r in rows}
        self.phone_digits = {_digits(r["phone"]) for r in rows if len(_digits(r["phone"])) >= 7}
        self.hidden_names = sorted(
            (r["full_name"] for r in rows if visible is not None and r["employee_id"] not in visible),
            key=len, reverse=True)

    def scrub(self, text: str) -> str:
        def email_sub(m: re.Match) -> str:
            return mask_email(m.group(0)) if m.group(0).lower() in self.emails else m.group(0)

        def phone_sub(m: re.Match) -> str:
            digits = _digits(m.group(0))
            if any(digits == p or (len(digits) >= 7 and p.endswith(digits)) for p in self.phone_digits):
                return "[phone number withheld]"
            return m.group(0)

        text = _EMAIL_RE.sub(email_sub, text)
        text = _PHONE_RE.sub(phone_sub, text)
        for name in self.hidden_names:
            text = re.sub(re.escape(name), "[name withheld]", text, flags=re.IGNORECASE)
        return text
