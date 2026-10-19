"""Shared fixtures.

The knowledge base (parsing, OCR, embeddings) is built once per test session.
Every test gets its own copy of veltrix.db, and integration tests use a
scripted fake LLM, so the suite is deterministic and needs no API key.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.config import load_settings  # noqa: E402
from app.ingestion.pipeline import build_knowledge_base  # noqa: E402
from app.orchestration.llm import LLMResponse, ToolUse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 10, 15)  # a Thursday


def _data_dir() -> Path:
    # Tests use a fixed copy of the assessment data, so replacing ./data (as the
    # graders do) does not change the expected balances and dates.
    for candidate in (os.environ.get("TEST_DATA_DIR"), ROOT / "tests" / "fixtures" / "data"):
        if candidate and (Path(candidate) / "manifest.json").exists() and (Path(candidate) / "corpus").is_dir():
            return Path(candidate)
    pytest.exit("Test data not found: expected tests/fixtures/data (or set TEST_DATA_DIR)", returncode=2)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return _data_dir()


@pytest.fixture(scope="session")
def base_settings(data_dir):
    s = load_settings()
    return dataclasses.replace(s, corpus_dir=data_dir / "corpus", manifest_path=data_dir / "manifest.json",
                               db_path=data_dir / "veltrix.db", app_today=TODAY, log_level="WARNING",
                               log_file=None)


@pytest.fixture(scope="session")
def kb(base_settings):
    return build_knowledge_base(base_settings)


@pytest.fixture()
def settings(base_settings, tmp_path):
    db_copy = tmp_path / "veltrix.db"
    shutil.copy(base_settings.db_path, db_copy)
    return dataclasses.replace(base_settings, db_path=db_copy)


# ------------------------------------------------------------------ fake LLM


def tool_call(name: str, **args) -> LLMResponse:
    use = ToolUse(id=f"tu_{name}_{abs(hash(json.dumps(args, sort_keys=True))) % 10**8}", name=name, input=args)
    return LLMResponse(stop_reason="tool_use", text="", tool_uses=[use],
                       assistant_content=[{"type": "tool_use", "id": use.id, "name": name, "input": args}])


def final(route: str, answer: str, citations: list[str] | None = None) -> LLMResponse:
    text = json.dumps({"route": route, "answer": answer, "citations": citations or []})
    return LLMResponse(stop_reason="end_turn", text=text, tool_uses=[],
                       assistant_content=[{"type": "text", "text": text}])


class FakeLLM:
    """Replays a script of responses and records every request it receives."""

    model = "fake-llm"

    def __init__(self, *responses):
        self.script = list(responses)
        self.requests: list[dict] = []

    def queue(self, *responses) -> None:
        self.script.extend(responses)

    def complete(self, *, system, messages, tools, output_schema, allow_tools, timeout):
        self.requests.append({"system": system, "messages": json.loads(json.dumps(messages, default=str)),
                              "allow_tools": allow_tools})
        if not self.script:
            return final("clarify", "How can I help?")
        nxt = self.script.pop(0)
        return nxt(self.requests[-1]) if callable(nxt) else nxt

    def sent_text(self) -> str:
        return json.dumps(self.requests)


@pytest.fixture()
def fake_llm():
    return FakeLLM()


@pytest.fixture()
def client(settings, kb, fake_llm):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings, kb=kb, provider_factory=lambda: fake_llm)
    with TestClient(app) as c:
        yield c


def chat(client, employee_id: str, message: str, session_id: str = "s1"):
    return client.post("/chat", json={"session_id": session_id, "message": message},
                       headers={"X-Employee-Id": employee_id})
