"""Access control for each role, for documents and for leave data."""
import pytest

from app.db import repository as repo
from app.tools import leave_tools as lt
from app.tools.errors import NOT_PERMITTED_MESSAGE, AccessDenied
from app.retrieval.search import Retriever

from .conftest import TODAY

QUERIES = [
    "salary band compensation for level L5",
    "performance improvement plan length",
    "how should managers approve leave requests",
    "sick leave allowance",
    "probation review",
    "confidential information",
]
RESTRICTED_FOR = {
    "employee": {"HB-MANAGER", "HR-COMP-BANDS-2026", "HR-PIP-GUIDE"},
    "manager": {"HR-COMP-BANDS-2026", "HR-PIP-GUIDE"},
    "hr": set(),
}


def ctx(settings, kb, employee_id, today=TODAY):
    with repo.session(settings.db_path) as conn:
        me = repo.get_employee(conn, employee_id)
    return lt.ToolContext(db_path=settings.db_path, requester=me, today=today, kb=kb)


class TestDocuments:
    @pytest.mark.parametrize("role", ["employee", "manager", "hr"])
    def test_restricted_documents_never_retrieved(self, kb, role):
        retriever = Retriever(kb)
        forbidden = RESTRICTED_FOR[role]
        for query in QUERIES:
            result = retriever.search(query, role=role, region="IN", reference_date=TODAY, today=TODAY,
                                      token_budget=3000)
            returned = {h.doc.doc_id for h in result.hits}
            assert not returned & forbidden, (role, query)
            for doc_id in forbidden:
                assert doc_id not in result.text
                assert kb.docs[doc_id].title not in result.text

    def test_permitted_restricted_documents_are_retrievable(self, kb):
        retriever = Retriever(kb)
        hr = retriever.search("salary band compensation for level L5", role="hr", region="IN",
                              reference_date=TODAY, today=TODAY, token_budget=3000)
        assert "HR-COMP-BANDS-2026" in {h.doc.doc_id for h in hr.hits}
        mgr = retriever.search("how should managers approve leave requests", role="manager", region="US",
                               reference_date=TODAY, today=TODAY, token_budget=3000)
        assert "HB-MANAGER" in {h.doc.doc_id for h in mgr.hits}

    def test_other_regions_documents_excluded(self, kb):
        result = Retriever(kb).search("public holidays in December", role="employee", region="IN",
                                      reference_date=TODAY, today=TODAY, token_budget=3000)
        regions = {r for h in result.hits for r in h.doc.regions}
        assert regions <= {"GLOBAL", "IN"}


class TestLeaveData:
    def test_employee_sees_own_balance(self, settings, kb):
        out = lt.get_leave_balance(ctx(settings, kb, "E2004"), "E2004")
        assert out["balances_days"]["annual"] == 3.5

    @pytest.mark.parametrize("target", ["E2005", "E2002", "E1001", "E9999"])
    def test_employee_cannot_see_others(self, settings, kb, target):
        with pytest.raises(AccessDenied) as exc:
            lt.get_leave_balance(ctx(settings, kb, "E2004"), target)
        assert exc.value.message == NOT_PERMITTED_MESSAGE  # same message whether or not E9999 exists
        with pytest.raises(AccessDenied):
            lt.list_leave_requests(ctx(settings, kb, "E2004"), target)

    def test_manager_sees_direct_reports_only(self, settings, kb):
        manager = ctx(settings, kb, "E2002")
        assert lt.get_leave_balance(manager, "E2004")["employee_id"] == "E2004"
        assert lt.list_leave_requests(manager, "E2007")["requests"]
        for target in ("E2008", "E2001", "E9999"):  # another team, own manager, missing
            with pytest.raises(AccessDenied) as exc:
                lt.get_leave_balance(manager, target)
            assert exc.value.message == NOT_PERMITTED_MESSAGE

    def test_hr_sees_any_single_employee(self, settings, kb):
        hr = ctx(settings, kb, "E1002")
        for target in ("E2004", "E3007", "E4010"):
            assert lt.get_leave_balance(hr, target)["employee_id"] == target
        with pytest.raises(AccessDenied):
            lt.get_leave_balance(hr, "E9999")

    @pytest.mark.parametrize("employee_id", ["E2004", "E1002"])
    def test_team_leave_is_managers_only(self, settings, kb, employee_id):
        with pytest.raises(AccessDenied):
            lt.get_team_leave(ctx(settings, kb, employee_id), "2026-11-01", "2026-11-30")

    def test_team_leave_returns_only_direct_reports(self, settings, kb):
        out = lt.get_team_leave(ctx(settings, kb, "E2002"), "2026-11-01", "2026-12-31")
        ids = {r["employee_id"] for r in out["team_leave"]}
        assert ids and ids <= {"E2004", "E2005", "E2006", "E2007"}
        assert all(r["status"] in ("pending", "approved") for r in out["team_leave"])
        assert "phone" not in str(out) and "email" not in str(out)

    def test_balance_result_contains_no_contact_details(self, settings, kb):
        out = lt.get_leave_balance(ctx(settings, kb, "E1002"), "E2004")
        assert set(out) == {"employee_id", "region", "join_date", "balances_days", "as_of", "note"}
