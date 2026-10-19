"""SQLite data access. The original tables are never altered; writes run in
explicit `BEGIN IMMEDIATE` transactions so a check and its write are atomic."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Employee:
    employee_id: str
    full_name: str
    email: str
    phone: str
    region: str
    join_date: str
    level: str
    role: str
    manager_id: str | None


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Path | str):
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _employee(row: sqlite3.Row | None) -> Employee | None:
    if row is None:
        return None
    return Employee(**{k: row[k] for k in row.keys()})


def get_employee(conn: sqlite3.Connection, employee_id: str) -> Employee | None:
    row = conn.execute(
        "SELECT employee_id, full_name, email, phone, region, join_date, level, role, manager_id "
        "FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
    return _employee(row)


def direct_report_ids(conn: sqlite3.Connection, manager_id: str) -> set[str]:
    rows = conn.execute("SELECT employee_id FROM employees WHERE manager_id = ?", (manager_id,))
    return {r["employee_id"] for r in rows}


def find_employees_by_name(conn: sqlite3.Connection, name: str, within: set[str] | None) -> list[Employee]:
    terms = [t for t in name.lower().split() if t]
    if not terms:
        return []
    rows = conn.execute(
        "SELECT employee_id, full_name, email, phone, region, join_date, level, role, manager_id FROM employees")
    out = []
    for row in rows:
        if within is not None and row["employee_id"] not in within:
            continue
        full = row["full_name"].lower()
        if all(t in full.split() or t in full for t in terms):
            out.append(_employee(row))
    return out


def get_balances(conn: sqlite3.Connection, employee_id: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = ? ORDER BY leave_type",
        (employee_id,))
    return {r["leave_type"]: r["balance_days"] for r in rows}


def _request(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def list_requests(conn: sqlite3.Connection, employee_id: str, status: str | None = None) -> list[dict]:
    sql = ("SELECT request_id, employee_id, leave_type, start_date, end_date, working_days, status, created_at "
           "FROM leave_requests WHERE employee_id = ?")
    params: list = [employee_id]
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY start_date, request_id"
    return [_request(r) for r in conn.execute(sql, params)]


def get_request(conn: sqlite3.Connection, request_id: str) -> dict | None:
    row = conn.execute(
        "SELECT request_id, employee_id, leave_type, start_date, end_date, working_days, status, created_at "
        "FROM leave_requests WHERE request_id = ?", (request_id,)).fetchone()
    return _request(row) if row else None


def active_overlaps(conn: sqlite3.Connection, employee_id: str, start: str, end: str) -> list[dict]:
    rows = conn.execute(
        "SELECT request_id, employee_id, leave_type, start_date, end_date, working_days, status, created_at "
        "FROM leave_requests WHERE employee_id = ? AND status IN ('pending', 'approved') "
        "AND start_date <= ? AND end_date >= ? ORDER BY start_date", (employee_id, end, start))
    return [_request(r) for r in rows]


def team_requests(conn: sqlite3.Connection, manager_id: str, start: str, end: str) -> list[dict]:
    rows = conn.execute(
        "SELECT r.request_id, r.employee_id, e.full_name, r.leave_type, r.start_date, r.end_date, "
        "r.working_days, r.status FROM leave_requests r JOIN employees e ON e.employee_id = r.employee_id "
        "WHERE e.manager_id = ? AND r.status IN ('pending', 'approved') "
        "AND r.start_date <= ? AND r.end_date >= ? ORDER BY r.start_date, r.employee_id",
        (manager_id, end, start))
    return [_request(r) for r in rows]


def insert_request(conn: sqlite3.Connection, req: dict) -> None:
    conn.execute(
        "INSERT INTO leave_requests (request_id, employee_id, leave_type, start_date, end_date, working_days, "
        "status, created_at) VALUES (:request_id, :employee_id, :leave_type, :start_date, :end_date, "
        ":working_days, :status, :created_at)", req)


def adjust_balance(conn: sqlite3.Connection, employee_id: str, leave_type: str, delta: float) -> None:
    cur = conn.execute(
        "UPDATE leave_balances SET balance_days = balance_days + ? WHERE employee_id = ? AND leave_type = ?",
        (delta, employee_id, leave_type))
    if cur.rowcount != 1:
        raise RuntimeError("balance row missing")


def set_request_status(conn: sqlite3.Connection, request_id: str, status: str) -> None:
    conn.execute("UPDATE leave_requests SET status = ? WHERE request_id = ?", (status, request_id))
