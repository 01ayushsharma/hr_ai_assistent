import logging

import pytest

from app.logging_setup import JsonFormatter, scrub
from app.orchestration.confirm import is_confirmation
from app.privacy.pii import mask_email


@pytest.mark.parametrize("text", ["yes", "Yes!", "confirm", "go ahead", "Yes, go ahead", "ok", "Sure, please do",
                                  "yes please", "Confirmed.", "yep, book it"])
def test_clear_confirmations(text):
    assert is_confirmation(text)


@pytest.mark.parametrize("text", ["no", "yes but make it Friday", "Make it until Friday instead", "cancel",
                                  "what is my balance?", "yes, from the 3rd", "not yet", "wait",
                                  "go ahead and also book 5 Dec"])
def test_not_confirmations(text):
    assert not is_confirmation(text)


def test_mask_email():
    assert mask_email("riya.k@veltrix.example") == "r***@veltrix.example"


def test_log_formatter_scrubs_contact_details():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "contact a.b@veltrix.example or +91 98765 43210",
                               None, None)
    line = JsonFormatter().format(record)
    assert "veltrix.example" not in line and "98765" not in line


@pytest.mark.parametrize("phone", ["+91 98765 43210", "+9 599-555-9989", "+49 39 5559 9689", "+1 512-555-0199",
                                   "512-555-0199", "(512) 555 0199", "98765 43210"])
def test_scrub_masks_phone_shapes(phone):
    assert scrub(f"call {phone} now") == "call [phone] now"


@pytest.mark.parametrize("text", ["2026-10-21T12:12:40.123+00:00", "2026-10-21", "1234-5678-9012-3456",
                                  "34a4b10f-e378-4dfa-b298-511e2a8fbe97", "latency_ms 12345678", "E2004",
                                  "corpus_tokens 2984"])
def test_scrub_keeps_dates_and_ids(text):
    assert scrub(text) == text


def test_timestamp_survives_formatter():
    line = JsonFormatter().format(logging.LogRecord("t", logging.INFO, __file__, 1, "x", None, None))
    assert "[phone]" not in line
