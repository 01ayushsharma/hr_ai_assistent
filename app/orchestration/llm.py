"""LLM provider interface and the Anthropic implementation.

The agent talks to a small `LLMProvider` protocol so tests can substitute a
scripted fake. `LLM_PROVIDER=anthropic` is the supported live provider.
"""
from __future__ import annotations

import logging
import os
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


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            log_event(logger, "llm_credentials_missing", logging.WARNING,
                      detail="Set ANTHROPIC_API_KEY; chat requests will fail until it is set.")
        return AnthropicProvider(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r} (supported: anthropic)")
