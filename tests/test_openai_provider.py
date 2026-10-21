"""The real OpenAICompatibleProvider against a local mock of Chat Completions.

Checks the request shape (tools, tool_choice, temperature, structured output)
and that the assistant tool call and its result are replayed in Chat
Completions form - without network access or an API key.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.db import repository as repo
from app.orchestration.agent import ChatService
from app.orchestration.llm import LLMUnavailable, OpenAICompatibleProvider, build_provider
from app.orchestration.sessions import Session


def _completion(message, finish_reason):
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


RESPONSES = [
    _completion({"content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
        "name": "search_policies", "arguments": json.dumps({"query": "sick leave days per year"})}}]},
        "tool_calls"),
    _completion({"content": "Here you go: " + json.dumps({"route": "rag", "answer": "Twelve.", "citations": ["C1"]})},
                "stop"),
]


@pytest.fixture()
def mock_api():
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
    yield f"http://127.0.0.1:{server.server_port}/v1", captured
    server.shutdown()


@pytest.mark.parametrize("structured", [True, False])
def test_agent_loop_over_openai_sdk(mock_api, settings, kb, structured):
    base_url, captured = mock_api
    provider = OpenAICompatibleProvider(api_key="test-key", base_url=base_url, model="test-model",
                                        structured_output=structured)
    service = ChatService(settings, kb, lambda: provider)
    with repo.session(settings.db_path) as conn:
        me = repo.get_employee(conn, "E2004")
    out = service.handle(request_id="r1", session_id="s", session=Session(owner_id="E2004"), requester=me,
                         message="How many sick days do I get?")
    # The JSON object is extracted even when the model wraps it in prose (no schema on Groq).
    assert out["route"] == "rag" and out["citations"]

    first, second = captured[0], captured[1]
    assert first["path"].endswith("/chat/completions")
    assert {k.lower(): v for k, v in first["headers"].items()}["authorization"] == "Bearer test-key"
    body = first["body"]
    assert body["model"] == "test-model" and body["temperature"] == 0 and body["tool_choice"] == "auto"
    assert body["messages"][0]["role"] == "system" and "Veltrix PeopleOps Assistant" in body["messages"][0]["content"]
    assert {t["function"]["name"] for t in body["tools"]} >= {"search_policies", "get_leave_balance", "apply_leave"}
    assert ("response_format" in body) is structured
    assert me.full_name not in json.dumps(body) and me.email not in json.dumps(body)

    replay = second["body"]["messages"]
    assert replay[-2]["role"] == "assistant"
    assert replay[-2]["tool_calls"][0]["id"] == "call_1"
    assert json.loads(replay[-2]["tool_calls"][0]["function"]["arguments"]) == {"query": "sick leave days per year"}
    assert replay[-1]["role"] == "tool" and replay[-1]["tool_call_id"] == "call_1"
    assert replay[-1]["content"].startswith("<search_results")


def test_build_provider_groq(settings, monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    provider = build_provider(replace(settings, llm_provider="groq"))
    assert isinstance(provider, OpenAICompatibleProvider)
    assert str(provider.client.base_url).startswith("https://api.groq.com/openai/v1")
    assert provider.model == settings.groq_model and provider.structured_output is False


def test_missing_key_is_reported_not_raised(settings, monkeypatch):
    from dataclasses import replace

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "")  # an empty value from .env must not break the client
    provider = build_provider(replace(settings, llm_provider="openai"))
    with pytest.raises(LLMUnavailable):
        provider.complete(system=[{"type": "text", "text": "x"}], messages=[], tools=[], output_schema={},
                          allow_tools=True, timeout=5)


def test_optional_tool_arguments_accept_null():
    from app.orchestration.llm import _nullable_optionals
    from app.orchestration.prompts import TOOLS

    schemas = {t["name"]: _nullable_optionals(t["input_schema"]) for t in TOOLS}
    search = schemas["search_policies"]["properties"]
    assert search["query"]["type"] == "string"  # required stays strict
    assert search["reference_date"]["type"] == ["string", "null"]
    balance = schemas["get_leave_balance"]["properties"]["leave_type"]
    assert balance["type"] == ["string", "null"] and balance["enum"] == ["annual", "sick", None]


def test_final_answer_sent_as_pseudo_tool_call_is_salvaged():
    import httpx
    import openai

    from app.orchestration.llm import _salvage_final_answer

    def rejected(generation):
        body = {"message": "Tool call validation failed", "type": "invalid_request_error",
                "code": "tool_use_failed", "failed_generation": generation}
        response = httpx.Response(400, request=httpx.Request("POST", "http://x/chat/completions"))
        return openai.BadRequestError("bad", response=response, body=body)

    final = {"route": "rag", "answer": "Twelve days.", "citations": ["C1"]}
    tools = {"search_policies", "get_leave_balance"}
    assert json.loads(_salvage_final_answer(rejected(json.dumps({"name": "json", "arguments": final})), tools)) == final
    # A malformed call to a real tool is not an answer; it is retried instead.
    real = json.dumps({"name": "search_policies", "arguments": {"query": None}})
    assert _salvage_final_answer(rejected(real), tools) is None
    assert _salvage_final_answer(rejected("not json"), tools) is None
