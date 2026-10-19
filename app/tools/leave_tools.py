"""The five leave tools (Section 6).

Read tools return plain dicts. Write tools are split into `check_*` (a dry run
that applies every rule) and `execute_*` (re-checks inside one BEGIN IMMEDIATE
transaction, then writes). The orchestrator only ever calls `check_*` while
proposing; `execute_*` runs after the user confirms.

Every function takes a ToolContext whose requester was loaded from the
X-Employee-Id header. Permission checks happen here, not in the prompt.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from ..db import repository as repo
from ..db.repository import Employee
from ..ingestion.pipeline import KnowledgeBase
from .errors import ToolRejected
from .permissions import require_manager, require_view_leave
from .working_days import daterange, holidays_in_range

LEAVE_TYPES = ("annual", "sick")
REQUEST_STATUSES = ("pending", "approved", "cancelled")
LEAVE_YEAR = 2026


@dataclass
class ToolContext:
    db_path: Path
    requester: Employee
    today: date
    kb: KnowledgeBase


def _parse_date(value, field: str) -> date:
    if not isinstance(value, str):
        raise ToolRejected(f"{field} must be a date in YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ToolRejected(f"{field} must be a date in YYYY-MM-DD format.") from None


def _leave_type(value, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or value.strip().lower() not in LEAVE_TYPES:
        raise ToolRejected("Leave type must be either 'annual' or 'sick'.")
    return value.strip().lower()


def _fmt(d: date) -> str:
    return f"{d.strftime('%a')} {d.isoformat()}"


def _days(n: float) -> str:
    return f"{n:g} day" + ("" if n == 1 else "s")


def _public_request(r: dict) -> dict:
    return {k: r[k] for k in ("request_id", "leave_type", "start_date", "end_date", "working_days", "status",
                              "created_at") if k in r}


# ------------------------------------------------------------------ read tools


def get_leave_balance(ctx: ToolContext, employee_id: str, leave_type: str | None = None) -> dict:
    lt = _leave_type(leave_type, required=False)
    with repo.session(ctx.db_path) as conn:
        require_view_leave(conn, ctx.requester, employee_id)
        target = repo.get_employee(conn, employee_id)
        balances = repo.get_balances(conn, employee_id)
    if lt:
        balances = {lt: balances.get(lt, 0.0)}
    return {"employee_id": employee_id, "region": target.region, "join_date": target.join_date,
            "balances_days": balances, "as_of": ctx.today.isoformat(),
            "note": "Current remaining balance; pending and approved requests are already deducted."}


def list_leave_requests(ctx: ToolContext, employee_id: str, status: str | None = None) -> dict:
    if status is not None and status not in REQUEST_STATUSES:
        raise ToolRejected("Status must be one of: pending, approved, cancelled.")
    with repo.session(ctx.db_path) as conn:
        require_view_leave(conn, ctx.requester, employee_id)
        requests = repo.list_requests(conn, employee_id, status)
    return {"employee_id": employee_id, "requests": [_public_request(r) for r in requests]}


def get_team_leave(ctx: ToolContext, start_date: str, end_date: str) -> dict:
    require_manager(ctx.requester)
    start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
    if end < start:
        raise ToolRejected("end_date must be on or after start_date.")
    with repo.session(ctx.db_path) as conn:
        rows = repo.team_requests(conn, ctx.requester.employee_id, start.isoformat(), end.isoformat())
    # Names are shown because a manager may see their direct reports.
    return {"start_date": start.isoformat(), "end_date": end.isoformat(), "team_leave": rows}


# ----------------------------------------------------------------- apply_leave


@dataclass
class ApplyPlan:
    leave_type: str
    start: date
    end: date
    working_days: int
    holidays: dict[date, str]
    balance_before: float

    @property
    def balance_after(self) -> float:
        return self.balance_before - self.working_days

    def summary(self) -> str:
        lines = [
            f"- Leave type: {self.leave_type}",
            f"- From {_fmt(self.start)} to {_fmt(self.end)} (inclusive)",
            f"- Working days: {self.working_days}",
        ]
        if self.holidays:
            names = ", ".join(f"{_fmt(d)} {n}".strip() for d, n in sorted(self.holidays.items())
                              if d.weekday() < 5)
            if names:
                lines.append(f"- Public holidays not counted: {names}")
        lines.append(f"- {self.leave_type.capitalize()} balance after this request: "
                     f"{_days(self.balance_after)} (currently {_days(self.balance_before)})")
        return "\n".join(lines)


def check_apply(ctx: ToolContext, leave_type, start_date, end_date, conn: sqlite3.Connection | None = None) -> ApplyPlan:
    lt = _leave_type(leave_type, required=True)
    start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
    if start.year != LEAVE_YEAR or end.year != LEAVE_YEAR:
        raise ToolRejected(f"Leave can only be requested for dates in {LEAVE_YEAR}.")
    if start < ctx.today:
        raise ToolRejected(f"The start date {start.isoformat()} is in the past. Leave must start today "
                           f"({ctx.today.isoformat()}) or later.")
    if end < start:
        raise ToolRejected("The end date must be on or after the start date.")
    me = ctx.requester
    holidays = holidays_in_range(ctx.kb, me.role, me.region, start, end)
    working = sum(1 for d in daterange(start, end) if d.weekday() < 5 and d not in holidays)
    if working == 0:
        raise ToolRejected(f"{start.isoformat()} to {end.isoformat()} has no working days: every date is a "
                           "weekend or a public holiday, so there is nothing to request.")

    def checks(c: sqlite3.Connection) -> ApplyPlan:
        overlaps = repo.active_overlaps(c, me.employee_id, start.isoformat(), end.isoformat())
        if overlaps:
            o = overlaps[0]
            raise ToolRejected(f"These dates overlap your existing {o['status']} {o['leave_type']} leave request "
                               f"{o['request_id']} ({o['start_date']} to {o['end_date']}).")
        balance = repo.get_balances(c, me.employee_id).get(lt, 0.0)
        if working > balance + 1e-9:
            raise ToolRejected(f"This request needs {_days(working)} of {lt} leave, but your {lt} leave balance "
                               f"is {_days(balance)}.")
        return ApplyPlan(lt, start, end, working, holidays, balance)

    if conn is not None:
        return checks(conn)
    with repo.session(ctx.db_path) as c:
        return checks(c)


def execute_apply(ctx: ToolContext, leave_type, start_date, end_date) -> tuple[ApplyPlan, str]:
    with repo.session(ctx.db_path) as conn:
        with repo.write_transaction(conn):
            plan = check_apply(ctx, leave_type, start_date, end_date, conn=conn)
            request_id = str(uuid.uuid4())
            repo.insert_request(conn, {
                "request_id": request_id,
                "employee_id": ctx.requester.employee_id,
                "leave_type": plan.leave_type,
                "start_date": plan.start.isoformat(),
                "end_date": plan.end.isoformat(),
                "working_days": plan.working_days,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            repo.adjust_balance(conn, ctx.requester.employee_id, plan.leave_type, -plan.working_days)
    return plan, request_id


# ---------------------------------------------------------------- cancel_leave


@dataclass
class CancelPlan:
    request: dict
    balance_before: float

    def summary(self) -> str:
        r = self.request
        return (f"- Request: {r['request_id']}\n"
                f"- {r['leave_type'].capitalize()} leave from {_fmt(date.fromisoformat(r['start_date']))} to "
                f"{_fmt(date.fromisoformat(r['end_date']))} ({_days(r['working_days'])}, status {r['status']})\n"
                f"- {_days(r['working_days'])} will go back to your {r['leave_type']} balance "
                f"({_days(self.balance_before)} now, {_days(self.balance_before + r['working_days'])} after)")


def check_cancel(ctx: ToolContext, request_id, conn: sqlite3.Connection | None = None) -> CancelPlan:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ToolRejected("A request ID is needed to cancel a leave request.")

    def checks(c: sqlite3.Connection) -> CancelPlan:
        req = repo.get_request(c, request_id.strip())
        # Someone else's request is reported exactly like a missing one.
        if req is None or req["employee_id"] != ctx.requester.employee_id:
            raise ToolRejected("I couldn't find a leave request with that ID among your requests.")
        if req["status"] not in ("pending", "approved"):
            raise ToolRejected(f"Request {req['request_id']} is already {req['status']}, so it can't be cancelled.")
        if date.fromisoformat(req["start_date"]) <= ctx.today:
            raise ToolRejected(f"Request {req['request_id']} starts on {req['start_date']}, which is not after today "
                               f"({ctx.today.isoformat()}). Only requests that start after today can be cancelled.")
        balance = repo.get_balances(c, ctx.requester.employee_id).get(req["leave_type"], 0.0)
        return CancelPlan(request=req, balance_before=balance)

    if conn is not None:
        return checks(conn)
    with repo.session(ctx.db_path) as c:
        return checks(c)


def execute_cancel(ctx: ToolContext, request_id) -> CancelPlan:
    with repo.session(ctx.db_path) as conn:
        with repo.write_transaction(conn):
            plan = check_cancel(ctx, request_id, conn=conn)
            req = plan.request
            repo.set_request_status(conn, req["request_id"], "cancelled")
            repo.adjust_balance(conn, ctx.requester.employee_id, req["leave_type"], req["working_days"])
    return plan
