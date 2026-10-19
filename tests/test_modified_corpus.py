"""The graders use a modified corpus: answers must follow the changed documents.

Copies a few documents, edits them (a new holiday row, a changed number, a
tighter access level in the manifest), re-ingests, and checks that holidays,
working days, retrieved text and access all reflect the edits. Nothing about
the corpus is hard-coded, so no code change is needed.
"""
import dataclasses
import json
import shutil
from datetime import date

import pytest

from app.ingestion.pipeline import build_knowledge_base
from app.retrieval.search import Retriever
from app.tools.working_days import count_working_days, holidays_in_range

from .conftest import TODAY

FILES = {
    "CAL-HOL-US-2026": "holiday_calendar_us_2026.html",
    "POL-HOLIDAY-WORK": "working_on_public_holidays.md",
    "FAQ-LEAVE": "employee_leave_faq.html",
}


@pytest.fixture(scope="module")
def modified_kb(base_settings, data_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("modified")
    corpus = root / "corpus"
    corpus.mkdir()
    for name in FILES.values():
        shutil.copy(data_dir / "corpus" / name, corpus / name)

    cal = corpus / FILES["CAL-HOL-US-2026"]
    cal.write_text(cal.read_text(encoding="utf-8").replace(
        "<tr><td>7 September 2026</td>",
        "<tr><td>2 November 2026</td><td>Monday</td><td>Founders Day</td></tr><tr><td>7 September 2026</td>"),
        encoding="utf-8")
    md = corpus / FILES["POL-HOLIDAY-WORK"]
    text = md.read_text(encoding="utf-8")
    assert "within 60 days" in text
    md.write_text(text.replace("within 60 days", "within 45 days"), encoding="utf-8")

    manifest = [e for e in json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
                if e["doc_id"] in FILES]
    for entry in manifest:
        if entry["doc_id"] == "FAQ-LEAVE":
            entry["access_level"] = "managers"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    settings = dataclasses.replace(base_settings, corpus_dir=corpus, manifest_path=root / "manifest.json")
    return build_knowledge_base(settings, use_embeddings=False)


def test_new_holiday_row_changes_working_days(modified_kb):
    holidays = holidays_in_range(modified_kb, "employee", "US", date(2026, 11, 2), date(2026, 11, 6))
    assert holidays == {date(2026, 11, 2): "Founders Day"}
    assert count_working_days(date(2026, 11, 2), date(2026, 11, 6), set(holidays)) == 4


def test_changed_value_is_what_gets_retrieved(modified_kb):
    result = Retriever(modified_kb).search("compensatory time off for working a public holiday", role="employee",
                                           region="US", reference_date=TODAY, today=TODAY, token_budget=3000)
    assert "within 45 days" in result.text and "within 60 days" not in result.text


def test_manifest_access_change_is_enforced(modified_kb):
    retriever = Retriever(modified_kb)
    for role, visible in (("employee", False), ("manager", True)):
        result = retriever.search("how many sick days do I get", role=role, region="US",
                                  reference_date=TODAY, today=TODAY, token_budget=3000)
        assert ("FAQ-LEAVE" in {h.doc.doc_id for h in result.hits}) is visible, role
