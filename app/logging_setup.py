"""Structured JSON logging with a personal-data scrubber.

Application code logs employee IDs and counters only, never message text or
names. The scrubber is a second line of defence: it masks anything that looks
like an email address or a phone number before a record is written.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Phone-number shapes only, so ISO dates, timestamps and request IDs pass through intact.
_PHONE_RE = re.compile(
    r"(?<![\w.:-])(?:"
    r"\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,5}){2,4}"  # +91 98765 43210, +49 39 5559 9689
    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"                    # 512-555-0199, (512) 555 0199
    r"|\d{5}\s\d{5}"                                          # 98765 43210
    r")(?![\w:-])"
)


def scrub(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    return _PHONE_RE.sub("[phone]", text)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            # Exception type only: messages and tracebacks may quote user data.
            payload["exception"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return scrub(json.dumps(payload, default=str, ensure_ascii=False))


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    # Third-party loggers can echo request bodies at DEBUG; keep them quiet.
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic", "pdfminer", "fastembed",
                  "huggingface_hub", "RapidOCR", "rapidocr_onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra={"fields": fields})
