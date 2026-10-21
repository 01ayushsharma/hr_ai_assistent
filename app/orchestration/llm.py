"""LLM provider interface and its live implementations.

The agent talks to a small `LLMProvider` protocol so tests can substitute a
scripted fake. Live providers: `anthropic`, `openai`, and `groq` (Groq serves
an OpenAI-compatible API, so it shares `OpenAICompatibleProvider`).

The agent builds requests in the Anthropic Messages shape (system blocks,
tool_use / tool_result content blocks). `OpenAICompatibleProvider` translates
that shape to Chat Completions on the way in and back on the way out, so the
agent loop is the same for every provider.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import Settings
from ..logging_setup import log_event

logger = logging.getLogger(__name__)


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    stop_reason: str
    text: str
    tool_uses: list[ToolUse]
    assistant_content: Any  # appended back to `messages` unchanged
    usage: dict = field(default_factory=dict)


class LLMProvider(Protocol):
    model: str

    def complete(self, *, system: list[dict], messages: list[dict], tools: list[dict],
                 output_schema: dict, allow_tools: bool, timeout: float) -> LLMResponse: ...


class LLMUnavailable(Exception):
    pass


class AnthropicProvider:
    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=1)
        self.model = settings.anthropic_model
        self.effort = settings.llm_effort
        self.fallbacks = settings.llm_fallbacks

    def complete(self, *, system, messages, tools, output_schema, allow_tools, timeout) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8000,
            "system": system,
            "messages": messages,
            "tools": tools,
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": output_schema}},
        }
        if not allow_tools:
            kwargs["tool_choice"] = {"type": "none"}
        if self.fallbacks == "default":
            # On a safety-classifier refusal, the API re-runs the request on a fallback model.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        try:
            resp = self.client.with_options(timeout=timeout).beta.messages.create(**kwargs)
        except self._anthropic.APIError as exc:
            raise LLMUnavailable(type(exc).__name__) from exc
        except TypeError as exc:  # raised by the SDK when no credentials are configured
            raise LLMUnavailable("missing_credentials") from exc
        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_uses = [ToolUse(id=b.id, name=b.name, input=dict(b.input or {}))
                     for b in resp.content if b.type == "tool_use"]
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                 "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", None)}
        # Echo every block back unchanged (thinking blocks included) on the next request.
        content = [b.to_dict() for b in resp.content]
        return LLMResponse(stop_reason=resp.stop_reason or "", text=text, tool_uses=tool_uses,
                           assistant_content=content, usage=usage)


OPENAI_BASE_URL = "https://api.openai.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_FINISH_REASONS = {"tool_calls": "tool_use", "function_call": "tool_use", "stop": "end_turn",
                   "length": "max_tokens", "content_filter": "refusal"}


def _to_chat_messages(system: list[dict], messages: list[dict]) -> list[dict]:
    """Anthropic-shaped system + messages -> Chat Completions messages."""
    out: list[dict] = [{"role": "system", "content": "\n\n".join(b["text"] for b in system)}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in content if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # a user turn carrying tool results
            for b in content:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def _nullable_optionals(schema: dict) -> dict:
    """Let optional tool arguments be null. Some models (gpt-oss on Groq) send `null` for an
    argument they mean to omit, and Groq rejects the whole call if the schema forbids it.
    The agent already treats a null argument as not given."""
    required = set(schema.get("required", []))
    props = {}
    for name, prop in schema.get("properties", {}).items():
        if name not in required and isinstance(prop.get("type"), str):
            prop = {**prop, "type": [prop["type"], "null"]}
            if "enum" in prop:
                prop["enum"] = [*prop["enum"], None]
        props[name] = prop
    return {**schema, "properties": props}


FINAL_ANSWER_NOTE = ("Write the final JSON object as the text of your message. It is not a tool: never call a "
                     "tool named json, final_answer or similar.")


def _salvage_final_answer(exc: Exception, tool_names: set[str]) -> str | None:
    """gpt-oss on Groq sometimes emits its final JSON answer as a call to a non-existent tool
    (e.g. `json`), which Groq rejects with tool_use_failed. If the rejected generation is a
    final answer, return it as JSON text; the agent parses and validates it as usual."""
    body = getattr(exc, "body", None)
    err = body.get("error", body) if isinstance(body, dict) else None
    generation = err.get("failed_generation") if isinstance(err, dict) else None
    if not isinstance(generation, str):
        return None
    try:
        data = json.loads(generation)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("name") not in tool_names and isinstance(data.get("arguments"), dict):
        data = data["arguments"]
    if isinstance(data, dict) and isinstance(data.get("route"), str) and isinstance(data.get("answer"), str):
        return json.dumps(data)
    return None


def _error_code(exc: Exception) -> str | None:
    """The API's machine-readable error code (e.g. rate_limit_exceeded); never the message text."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and isinstance(err.get("code"), str):
            return err["code"][:64]
    return None


def _retry_after(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    try:
        return float(response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return 60.0  # unknown: treat as too long to wait


class OpenAICompatibleProvider:
    """OpenAI Chat Completions, or any server that speaks it (Groq)."""

    def __init__(self, *, api_key: str | None, base_url: str | None, model: str,
                 structured_output: bool) -> None:
        import openai

        self._openai = openai
        # Retries are handled in _create so a long Retry-After cannot push a request past its deadline.
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0) if api_key else None
        self.model = model
        # Groq rejects a JSON schema combined with tools; there the system prompt alone asks for
        # JSON, and the agent extracts the object from the text.
        self.structured_output = structured_output

    def _create(self, kwargs: dict, timeout: float):
        """One call, retried once for transient failures that fit inside the timeout:
        a malformed tool call from the model (Groq's `tool_use_failed`), a short rate-limit
        wait, or a connection error."""
        deadline = time.monotonic() + timeout
        for attempt in (1, 2):
            remaining = deadline - time.monotonic()
            try:
                return self.client.with_options(timeout=max(1.0, remaining)).chat.completions.create(**kwargs)
            except self._openai.APIError as exc:
                code = _error_code(exc)
                if code == "tool_use_failed":
                    salvaged = _salvage_final_answer(exc, {t["function"]["name"] for t in kwargs["tools"]})
                    if salvaged is not None:
                        log_event(logger, "llm_final_answer_salvaged", logging.WARNING)
                        return salvaged
                wait =_retry_after(exc) if isinstance(exc, self._openai.RateLimitError) else 0.0
                retryable = (code == "tool_use_failed" or isinstance(exc, self._openai.APIConnectionError)
                             or (isinstance(exc, self._openai.RateLimitError) and wait <= 5))
                if attempt == 2 or not retryable or deadline - time.monotonic() - wait < 3:
                    raise LLMUnavailable(f"{type(exc).__name__}:{code}" if code else type(exc).__name__) from exc
                log_event(logger, "llm_retry", logging.WARNING, error=type(exc).__name__, code=code,
                          wait_s=round(wait, 1))
                time.sleep(wait)
        raise AssertionError("unreachable")

    def complete(self, *, system, messages, tools, output_schema, allow_tools, timeout) -> LLMResponse:
        if self.client is None:
            raise LLMUnavailable("missing_credentials")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _to_chat_messages([*system, {"type": "text", "text": FINAL_ANSWER_NOTE}], messages),
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": _nullable_optionals(t["input_schema"])}}
                      for t in tools],
            "tool_choice": "auto" if allow_tools else "none",
            "temperature": 0,
            "max_completion_tokens": 4096,
        }
        if self.structured_output:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "final_answer", "schema": output_schema, "strict": True}}
        resp = self._create(kwargs, timeout)
        if isinstance(resp, str):  # a final answer recovered from a rejected pseudo tool call
            return LLMResponse(stop_reason="end_turn", text=resp, tool_uses=[],
                               assistant_content=[{"type": "text", "text": resp}])
        choice = resp.choices[0]
        msg = choice.message
        if getattr(msg, "refusal", None):
            return LLMResponse(stop_reason="refusal", text="", tool_uses=[], assistant_content=[])
        tool_uses = []
        for call in msg.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_uses.append(ToolUse(id=call.id, name=call.function.name,
                                     input=args if isinstance(args, dict) else {}))
        text = msg.content or ""
        content: list[dict] = [{"type": "text", "text": text}] if text else []
        content += [{"type": "tool_use", "id": t.id, "name": t.name, "input": t.input} for t in tool_uses]
        usage = {}
        if resp.usage is not None:
            usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        stop = "tool_use" if tool_uses else _FINISH_REASONS.get(choice.finish_reason or "", "end_turn")
        return LLMResponse(stop_reason=stop, text=text, tool_uses=tool_uses, assistant_content=content,
                           usage=usage)


def _warn_missing(key_name: str) -> None:
    log_event(logger, "llm_credentials_missing", logging.WARNING,
              detail=f"Set {key_name}; chat requests will fail until it is set.")


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            _warn_missing("ANTHROPIC_API_KEY")
        return AnthropicProvider(settings)
    if settings.llm_provider in ("openai", "groq"):
        key_name = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "GROQ_API_KEY"
        api_key = (os.environ.get(key_name) or "").strip() or None
        if api_key is None:
            _warn_missing(key_name)
        if settings.llm_provider == "openai":
            # Pass the default explicitly: the SDK would otherwise read an empty OPENAI_BASE_URL from the env.
            return OpenAICompatibleProvider(api_key=api_key, base_url=settings.openai_base_url or OPENAI_BASE_URL,
                                            model=settings.openai_model, structured_output=True)
        return OpenAICompatibleProvider(api_key=api_key, base_url=GROQ_BASE_URL, model=settings.groq_model,
                                        structured_output=False)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r} (supported: anthropic, openai, groq)")
