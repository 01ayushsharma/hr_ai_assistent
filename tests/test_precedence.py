"""Section 5 rules, using the worked example from the task and the real corpus."""
from datetime import date

from app.ingestion.models import DocMeta
from app.retrieval.eligibility import is_eligible, is_readable
from app.retrieval.precedence import blockers_of_upcoming, order_by_precedence, winner
from app.retrieval.search import Retriever

from .conftest import TODAY


def doc(doc_id, doc_type, regions, effective, access="all"):
    return DocMeta(doc_id=doc_id, file=f"{doc_id}.md", title=doc_id, doc_type=doc_type,
                   regions=tuple(regions), effective_date=date.fromisoformat(effective), access_level=access)


V1 = doc("V1", "policy", ["GLOBAL"], "2025-01-01")
V2 = doc("V2", "policy", ["GLOBAL"], "2026-04-01")
FAQ = doc("FAQ", "faq", ["GLOBAL"], "2026-05-01")
DE_ADD = doc("DE-ADD", "addendum", ["DE"], "2025-06-01")
V3 = doc("V3", "policy", ["GLOBAL"], "2027-01-01")
ALL = [V1, V2, FAQ, DE_ADD, V3]
REF = date(2026, 9, 15)


def eligible(region, ref):
    return [d for d in ALL if is_eligible(d, "employee", region, ref)]


class TestWorkedExample:
    def test_india_employee_gets_v2_and_upcoming_v3(self):
        docs = eligible("IN", REF)
        assert {d.doc_id for d in docs} == {"V1", "V2", "FAQ"}  # DE addendum and V3 excluded
        assert [d.doc_id for d in winner(docs)] == ["V2"]
        tiers = [[d.doc_id for d in t] for t in order_by_precedence(docs)]
        assert tiers == [["V2"], ["V1"], ["FAQ"]]  # FAQ loses on authority despite being newest
        assert blockers_of_upcoming(V3, docs) == []  # V3 would win -> mention it

    def test_germany_employee_addendum_wins_and_v3_is_not_mentioned(self):
        docs = eligible("DE", REF)
        assert [d.doc_id for d in winner(docs)] == ["DE-ADD"]
        assert [d.doc_id for d in blockers_of_upcoming(V3, docs)] == ["DE-ADD"]

    def test_india_employee_past_date_only_v1(self):
        docs = eligible("IN", date(2025, 8, 1))
        assert [d.doc_id for d in docs] == ["V1"]


class TestRules:
    def test_authority_beats_recency(self):
        assert [d.doc_id for d in winner([V1, FAQ])] == ["V1"]

    def test_scope_beats_recency(self):
        newer_global = doc("G", "policy", ["GLOBAL"], "2026-06-01")
        older_regional = doc("R", "addendum", ["IN"], "2025-01-01")
        assert [d.doc_id for d in winner([newer_global, older_regional])] == ["R"]

    def test_tie_is_reported_as_tie(self):
        a = doc("A", "policy", ["GLOBAL"], "2026-04-01")
        b = doc("B", "handbook", ["GLOBAL"], "2026-04-01")
        assert {d.doc_id for d in winner([a, b])} == {"A", "B"}

    def test_future_document_is_not_eligible(self):
        assert not is_eligible(V3, "employee", "IN", REF)
        assert is_readable(V3, "employee", "IN")

    def test_multi_region_document(self):
        memo = doc("M", "memo", ["IN", "US"], "2026-07-01")
        assert is_eligible(memo, "employee", "US", REF)
        assert not is_eligible(memo, "employee", "DE", REF)


class TestOnCorpus:
    def test_regional_addendum_ranked_first_for_germany(self, kb):
        result = Retriever(kb).search("sick leave days per year", role="employee", region="DE",
                                      reference_date=TODAY, today=TODAY, token_budget=3000)
        order_line = result.text.splitlines()[1]
        assert order_line.split(": ", 1)[1].startswith("ADD-LEAVE-DE")
        # Authoritative global v2 outranks the out-of-date FAQ.
        assert order_line.index("POL-LEAVE-GLOBAL-V2") < order_line.index("FAQ-LEAVE")

    def test_past_reference_date_excludes_later_documents(self, kb):
        ref = date(2025, 8, 1)
        result = Retriever(kb).search("sick leave allowance", role="employee", region="IN",
                                      reference_date=ref, today=TODAY, token_budget=3000)
        assert result.hits
        assert all(h.doc.effective_date <= ref and not h.upcoming for h in result.hits)

    def test_upcoming_v3_is_labelled(self, kb):
        result = Retriever(kb).search("annual leave carry forward limit", role="employee", region="IN",
                                      reference_date=TODAY, today=TODAY, token_budget=3000)
        upcoming = [h for h in result.hits if h.upcoming]
        assert upcoming and all(h.doc.doc_id == "POL-LEAVE-GLOBAL-V3" for h in upcoming)
        assert 'status="UPCOMING' in result.text
