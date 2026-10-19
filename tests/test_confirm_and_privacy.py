import logging

import pytest

from app.logging_setup import JsonFormatter
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
