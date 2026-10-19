"""Rules for apply_leave and cancel_leave (Sections 6.2 and 6.3)."""
import uuid
from datetime import date

import pytest

from app.db import repository as repo
from app.tools import leave_tools as lt
from app.tools.errors import ToolRejected

from .conftest import TODAY


def ctx(settings, kb, employee_id, today=TODAY):
    with repo.session(settings.db_path) as conn:
        me = repo.get_employee(conn, employee_id)
    return lt.ToolContext(db_path=settings.db_path, requester=me, today=today, kb=kb)


def balance(settings, employee_id, leave_type):
    with repo.session(settings.db_path) as conn:
        return repo.get_balances(conn, employee_id)[leave_type]


class TestApplyRules:
    @pytest.mark.parametrize("args,reason", [
        (("vacation", "2026-11-02", "2026-11-03"), "annual' or 'sick"),
        (("annual", "2027-01-04", "2027-01-05"), "dates in 2026"),
        (("annual", "2026-12-30", "2027-01-02"), "dates in 2026"),
        (("annual", "2026-10-14", "2026-10-16"), "in the past"),
        (("annual", "2026-11-06", "2026-11-02"), "on or after the start date"),
        (("annual", "2026-11-07", "2026-11-08"), "no working days"),
        (("annual", "02/11/2026", "2026-11-03"), "YYYY-MM-DD"),
    ])
    def test_rejections(self, settings, kb, args, reason):
        with pytest.raises(ToolRejected) as exc:
            lt.check_apply(ctx(settings, kb, "E3004"), *args)
        assert reason in exc.value.reason

    def test_starting_today_is_allowed(self, settings, kb):
        plan = lt.check_apply(ctx(settings, kb, "E3004"), "annual", TODAY.isoformat(), TODAY.isoformat())
        assert plan.working_days == 1

    def test_only_holidays_and_weekend_is_zero_days(self, settings, kb):
        # US: Thanksgiving Thu + Fri, then the weekend.
        with pytest.raises(ToolRejected, match="no working days"):
            lt.check_apply(ctx(settings, kb, "E3004"), "annual", "2026-11-26", "2026-11-29")

    def test_exceeding_balance(self, settings, kb):
        with pytest.raises(ToolRejected) as exc:
            lt.check_apply(ctx(settings, kb, "E2004"), "annual", "2026-11-16", "2026-11-20")
        assert "5 days" in exc.value.reason and "3.5 days" in exc.value.reason

    def test_overlap_with_pending_request(self, settings, kb):
        with pytest.raises(ToolRejected, match="overlap"):
            lt.check_apply(ctx(settings, kb, "E3004"), "sick", "2026-11-25", "2026-11-25")

    def test_cancelled_requests_do_not_block(self, settings, kb):
        # E4009 has a *cancelled* request for exactly these dates.
        plan = lt.check_apply(ctx(settings, kb, "E4009"), "sick", "2026-12-07", "2026-12-11")
        assert plan.working_days == 5

    def test_check_does_not_write(self, settings, kb):
        before = balance(settings, "E3004", "annual")
        lt.check_apply(ctx(settings, kb, "E3004"), "annual", "2026-11-02", "2026-11-06")
        assert balance(settings, "E3004", "annual") == before
        with repo.session(settings.db_path) as conn:
            assert not repo.active_overlaps(conn, "E3004", "2026-11-02", "2026-11-06")

    def test_execute_creates_pending_request_and_deducts(self, settings, kb):
        before = balance(settings, "E3004", "annual")
        plan, request_id = lt.execute_apply(ctx(settings, kb, "E3004"), "annual", "2026-11-02", "2026-11-06")
        assert uuid.UUID(request_id).version == 4
        assert plan.working_days == 5
        assert balance(settings, "E3004", "annual") == before - 5
        with repo.session(settings.db_path) as conn:
            row = repo.get_request(conn, request_id)
        assert row["status"] == "pending" and row["working_days"] == 5 and row["employee_id"] == "E3004"

    def test_execute_rechecks_rules(self, settings, kb):
        c = ctx(settings, kb, "E3004")
        lt.check_apply(c, "annual", "2026-12-07", "2026-12-11")  # proposal looks fine...
        lt.execute_apply(c, "annual", "2026-11-30", "2026-12-04")  # ...but the balance changes meanwhile
        with pytest.raises(ToolRejected, match="balance"):
            lt.execute_apply(c, "annual", "2026-12-07", "2026-12-11")
        assert balance(settings, "E3004", "annual") == 3.5  # nothing partially written

    def test_holidays_are_not_charged(self, settings, kb):
        plan = lt.check_apply(ctx(settings, kb, "E4007"), "sick", "2026-12-21", "2026-12-31")
        assert plan.working_days == 6
        assert date(2026, 12, 24) in plan.holidays


class TestCancelRules:
    PENDING = "33288e16-ebe8-4fa9-9c24-c1ae7d510557"    # E3004, 2026-11-23..25, 3 days, pending
    STARTED = "f5ee8c72-98e6-45b7-a1cd-7be8a7d83351"    # E3004, 2026-07-06, approved, in the past
    CANCELLED = "2edc49c1-f71b-4135-ac06-43b173f29c6a"  # E3008, already cancelled
    OTHERS = "e842997d-a9f5-407c-8ece-78b06f29a9f1"     # belongs to E3005

    def test_cancel_returns_days(self, settings, kb):
        before = balance(settings, "E3004", "annual")
        lt.execute_cancel(ctx(settings, kb, "E3004"), self.PENDING)
        assert balance(settings, "E3004", "annual") == before + 3
        with repo.session(settings.db_path) as conn:
            assert repo.get_request(conn, self.PENDING)["status"] == "cancelled"

    def test_cannot_cancel_started_request(self, settings, kb):
        with pytest.raises(ToolRejected, match="not after today"):
            lt.check_cancel(ctx(settings, kb, "E3004"), self.STARTED)

    def test_cannot_cancel_request_starting_today(self, settings, kb):
        with pytest.raises(ToolRejected, match="not after today"):
            lt.check_cancel(ctx(settings, kb, "E3004", today=date(2026, 11, 23)), self.PENDING)

    def test_cannot_cancel_twice(self, settings, kb):
        with pytest.raises(ToolRejected, match="already cancelled"):
            lt.check_cancel(ctx(settings, kb, "E3008"), self.CANCELLED)

    def test_cannot_cancel_someone_elses_request(self, settings, kb):
        for requester in ("E3004", "E3002", "E1002"):  # colleague, their manager, HR
            with pytest.raises(ToolRejected, match="couldn't find"):
                lt.check_cancel(ctx(settings, kb, requester), self.OTHERS)

    def test_unknown_request(self, settings, kb):
        with pytest.raises(ToolRejected, match="couldn't find"):
            lt.check_cancel(ctx(settings, kb, "E3004"), str(uuid.uuid4()))
