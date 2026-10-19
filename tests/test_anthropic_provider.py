"""The real AnthropicProvider against a local mock of the Messages API.

Checks the request shape we send (model, tools, structured output, fallback
beta) and that tool results are replayed correctly - without network access
or an API key.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.orchestration.agent import ChatService
from app.orchestration.llm import AnthropicProvider
from app.orchestration.sessions import Session
from app.db import repository as repo


def _message(content, stop_reason):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


RESPONSES = [
    _message([{"type": "thinking", "thinking": "", "signature": "sig"},
              {"type": "tool_use", "id": "toolu_1", "name": "search_policies",
               "input": {"query": "sick leave days per year"}}], "tool_use"),
    _message([{"type": "text", "text": json.dumps({"route": "rag", "answer": "Twelve.", "citations": ["C1"]})}],
             "end_turn"),
]


@pytest.fixture()
def mock_api(monkeypatch):
    captured: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append({"path": self.path, "headers": dict(self.headers), "body": body})
            payload = json.dumps(RESPONSES[min(len(captured) - 1, len(RESPONSES) - 1)]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    yield captured
    server.shutdown()


def test_agent_loop_over_real_sdk(mock_api, settings, kb):
    service = ChatService(settings, kb, lambda: AnthropicProvider(settings))
    with repo.session(settings.db_path) as conn:
        me = repo.get_employee(conn, "E2004")
    out = service.handle(request_id="r1", session_id="s", session=Session(owner_id="E2004"), requester=me,
                         message="How many sick days do I get?")
    assert out["route"] == "rag" and out["citations"]

    first, second = mock_api[0], mock_api[1]
    assert first["path"].startswith("/v1/messages")
    body = first["body"]
    assert body["model"] == settings.anthropic_model
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == settings.llm_effort
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in first["headers"].get("anthropic-beta", "")
    assert {t["name"] for t in body["tools"]} >= {"search_policies", "get_leave_balance", "apply_leave"}
    assert "temperature" not in body
    # Personal data is not in the prompt.
    assert me.full_name not in json.dumps(body) and me.email not in json.dumps(body)

    # The assistant turn (thinking + tool_use) is echoed back, followed by the tool result.
    replay = second["body"]["messages"]
    assert replay[-2]["role"] == "assistant"
    assert [b["type"] for b in replay[-2]["content"]] == ["thinking", "tool_use"]
    assert replay[-1]["content"][0]["type"] == "tool_result"
    assert replay[-1]["content"][0]["tool_use_id"] == "toolu_1"
    assert replay[-1]["content"][0]["content"].startswith("<search_results")
