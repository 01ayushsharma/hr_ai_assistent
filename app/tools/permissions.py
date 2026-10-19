"""Leave-data access rules (Section 3.1 / 6.1). Called from inside every tool.

The requester is always the employee loaded from the X-Employee-Id header;
nothing here reads identity from the prompt or from LLM-chosen arguments.
"""
from __future__ import annotations

import sqlite3

from ..db.repository import Employee, direct_report_ids, get_employee
from .errors import AccessDenied


def can_view_leave(conn: sqlite3.Connection, requester: Employee, target_id: str) -> bool:
    if target_id == requester.employee_id:
        return True
    target = get_employee(conn, target_id)
    if target is None:
        return False
    if requester.role == "hr":
        return True
    if requester.role == "manager":
        return target.manager_id == requester.employee_id
    return False


def require_view_leave(conn: sqlite3.Connection, requester: Employee, target_id: str) -> None:
    if not can_view_leave(conn, requester, target_id):
        raise AccessDenied()


def require_manager(requester: Employee) -> None:
    if requester.role != "manager":
        raise AccessDenied("Only managers can view team leave, and only for their own direct reports.")


def visible_employee_ids(conn: sqlite3.Connection, requester: Employee) -> set[str] | None:
    """Employees whose names and leave the requester may see; None means everyone (HR)."""
    if requester.role == "hr":
        return None
    ids = {requester.employee_id}
    if requester.role == "manager":
        ids |= direct_report_ids(conn, requester.employee_id)
    return ids
