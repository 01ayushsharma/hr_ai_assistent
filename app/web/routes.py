"""JSON endpoints behind the web UI (the leave application form and leave views).

These sit beside the /chat contract and reuse the same tool functions, so the
same permission checks and leave rules apply. Identity always comes from the
X-Employee-Id header. The form's preview -> submit sequence is the explicit
confirmation for a write: /leave/preview never writes, /leave/apply and
/leave/cancel run only when the user presses the submit or cancel button.
"""
from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..config import Settings
from ..db import repository as repo
from ..logging_setup import log_event
from ..privacy.pii import mask_email
from ..retrieval.eligibility import is_readable
from ..tools import leave_tools as lt
from ..tools.errors import AccessDenied, ToolRejected
from ..tools.working_days import holidays_in_range

logger = logging.getLogger(__name__)


class LeaveForm(BaseModel):
    leave_type: str
    start_date: str
    end_date: str


class CancelForm(BaseModel):
    request_id: str


def _plan_json(plan: lt.ApplyPlan) -> dict:
    return {
        "leave_type": plan.leave_type,
        "start_date": plan.start.isoformat(),
        "end_date": plan.end.isoformat(),
        "working_days": plan.working_days,
        "calendar_days": (plan.end - plan.start).days + 1,
        "holidays": [{"date": d.isoformat(), "name": n} for d, n in sorted(plan.holidays.items())
                     if d.weekday() < 5],
        "balance_before": plan.balance_before,
        "balance_after": plan.balance_after,
    }


def build_web_router(settings: Settings, state) -> APIRouter:
    router = APIRouter(prefix="/api")

    def context(request: Request) -> lt.ToolContext:
        employee_id = (request.headers.get("X-Employee-Id") or "").strip()
        if not employee_id:
            raise HTTPException(status_code=401, detail="X-Employee-Id header is required.")
        with repo.session(settings.db_path) as conn:
            me = repo.get_employee(conn, employee_id)
        if me is None:
            raise HTTPException(status_code=401, detail="Unknown employee.")
        if not state.ready.is_set() or state.service is None:
            raise HTTPException(status_code=503, detail="The service is still starting.")
        return lt.ToolContext(db_path=settings.db_path, requester=me, today=settings.today(),
                              kb=state.service.kb)

    @router.get("/me")
    def me(request: Request):
        c = context(request)
        e = c.requester
        return {"employee_id": e.employee_id, "full_name": e.full_name, "email": mask_email(e.email),
                "role": e.role, "region": e.region, "level": e.level, "join_date": e.join_date,
                "today": c.today.isoformat(), "leave_year": lt.LEAVE_YEAR}

    @router.get("/documents")
    def documents(request: Request):
        """Titles of documents this requester may read (used to label citations)."""
        c = context(request)
        return {d.doc_id: d.title for d in c.kb.docs.values()
                if is_readable(d, c.requester.role, c.requester.region)}

    @router.get("/leave/balances")
    def balances(request: Request):
        c = context(request)
        return lt.get_leave_balance(c, c.requester.employee_id)["balances_days"]

    @router.get("/leave/requests")
    def requests(request: Request):
        c = context(request)
        rows = lt.list_leave_requests(c, c.requester.employee_id)["requests"]
        for r in rows:
            r["cancellable"] = (r["status"] in ("pending", "approved")
                                and date.fromisoformat(r["start_date"]) > c.today)
        return sorted(rows, key=lambda r: r["start_date"], reverse=True)

    @router.get("/holidays")
    def holidays(request: Request):
        c = context(request)
        year = lt.LEAVE_YEAR
        found = holidays_in_range(c.kb, c.requester.role, c.requester.region, date(year, 1, 1), date(year, 12, 31))
        return [{"date": d.isoformat(), "name": n} for d, n in sorted(found.items())]

    @router.post("/leave/preview")
    def preview(form: LeaveForm, request: Request):
        c = context(request)
        try:
            return {"ok": True, **_plan_json(lt.check_apply(c, form.leave_type, form.start_date, form.end_date))}
        except ToolRejected as exc:
            return {"ok": False, "reason": exc.reason}

    @router.post("/leave/apply")
    def apply(form: LeaveForm, request: Request):
        c = context(request)
        try:
            plan, request_id = lt.execute_apply(c, form.leave_type, form.start_date, form.end_date)
        except ToolRejected as exc:
            raise HTTPException(status_code=422, detail=exc.reason) from None
        log_event(logger, "web_leave_applied", employee_id=c.requester.employee_id, request_id=request_id,
                  working_days=plan.working_days)
        return {"request_id": request_id, "status": "pending", **_plan_json(plan)}

    @router.post("/leave/cancel")
    def cancel(form: CancelForm, request: Request):
        c = context(request)
        try:
            plan = lt.execute_cancel(c, form.request_id)
        except ToolRejected as exc:
            raise HTTPException(status_code=422, detail=exc.reason) from None
        r = plan.request
        log_event(logger, "web_leave_cancelled", employee_id=c.requester.employee_id, request_id=r["request_id"])
        return {"request_id": r["request_id"], "status": "cancelled", "working_days_returned": r["working_days"],
                "balance_after": plan.balance_before + r["working_days"]}

    @router.get("/team/leave")
    def team_leave(request: Request, start_date: str, end_date: str):
        c = context(request)
        try:
            return lt.get_team_leave(c, start_date, end_date)["team_leave"]
        except AccessDenied as exc:
            raise HTTPException(status_code=403, detail=exc.message) from None
        except ToolRejected as exc:
            raise HTTPException(status_code=422, detail=exc.reason) from None

    return router
