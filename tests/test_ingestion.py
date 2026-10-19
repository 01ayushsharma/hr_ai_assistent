"""Ingestion of every format, tables and the scanned document."""
from app.ingestion.chunker import MAX_CHUNK_TOKENS
from app.ingestion.pipeline import load_manifest


def chunks_of(kb, doc_id):
    return [c for c in kb.chunks if c.doc_id == doc_id]


def test_every_manifest_document_is_ingested(kb, base_settings):
    manifest = load_manifest(base_settings.manifest_path)
    assert kb.report.failures == []
    for meta in manifest:
        assert chunks_of(kb, meta.doc_id), meta.doc_id


def test_scanned_memo_is_read_with_ocr(kb):
    assert kb.report.ocr_pages >= 1
    text = " ".join(c.text for c in chunks_of(kb, "MEMO-HYBRID-2026"))
    assert "work from home for up to 3 days per week" in text
    assert "1 July 2026" in text
    assert all(c.locator.startswith("p. ") for c in chunks_of(kb, "MEMO-HYBRID-2026"))


def test_pdf_table_kept_as_markdown_with_header(kb):
    table = next(c for c in chunks_of(kb, "TBL-LEAVE-ACCRUAL") if "|" in c.text)
    assert "| Length of service | Days earned per month | Days earned per full year |" in table.text
    assert "| 2 years to less than 5 years | 1.75 | 21 |" in table.text


def test_docx_and_markdown_tables(kb):
    v3 = " ".join(c.text for c in chunks_of(kb, "POL-LEAVE-GLOBAL-V3"))
    assert "| Carry-forward limit | 10 days | 12 days |" in v3
    learning = " ".join(c.text for c in chunks_of(kb, "POL-LEARNING"))
    assert "| Berlin, Germany | €900 |" in learning


def test_metadata_tables_are_dropped(kb):
    assert not any("| Document ID |" in c.text for c in kb.chunks)


def test_locators_use_section_headings(kb):
    locators = {c.locator for c in chunks_of(kb, "POL-LEAVE-GLOBAL-V2")}
    assert "7. Sick leave" in locators
    faq = {c.locator for c in chunks_of(kb, "FAQ-LEAVE")}
    assert "How many sick days do I get?" in faq


def test_chunks_are_bounded(kb):
    assert max(c.tokens for c in kb.chunks) <= MAX_CHUNK_TOKENS


def test_holidays_extracted_from_all_calendars(kb):
    assert set(kb.holidays) == {"CAL-HOL-IN-2026", "CAL-HOL-IN-2025", "CAL-HOL-US-2026", "CAL-HOL-DE-2026"}
    assert all(len(v) >= 10 for v in kb.holidays.values())
    in_2026 = {h.day.isoformat() for h in kb.holidays["CAL-HOL-IN-2026"]}
    assert "2026-11-09" in in_2026 and "2026-03-04" not in in_2026  # restricted holiday excluded


def test_injection_text_is_only_data(kb):
    # The planted instruction is ingested like any other text; it is never part of the prompt.
    from app.orchestration.prompts import SYSTEM_PROMPT

    wiki = " ".join(c.text for c in chunks_of(kb, "WIKI-NEW-JOINERS"))
    assert "AI assistant" in wiki
    assert "salary band" not in SYSTEM_PROMPT.lower()
