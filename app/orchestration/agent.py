"""Single-agent orchestration for one /chat message.

Flow:
1. If the session has a proposal waiting and this message is a clear
   confirmation, execute it deterministically (no LLM call).
2. Otherwise run one bounded tool-use loop with the LLM. The LLM chooses which
   tools to call; code enforces everything else:
   - search_policies filters documents by role/region/date before ranking and
     tracks corpus tokens so no LLM call carries more than 3,000;
   - leave tools check permissions with the header identity;
   - write tools only *propose*; the turn ends with a code-written summary.
3. The route, citations and tool_calls are derived from what actually happened,
   then the answer passes through the personal-data scrubber.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date

from ..config import Settings
from ..db import repository as repo
from ..db.repository import Employee
from ..ingestion.pipeline import KnowledgeBase
from ..logging_setup import log_event
from ..privacy.pii import AnswerScrubber
from ..retrieval.search import CORPUS_TOKEN_LIMIT, Retriever, SearchHit
from ..tools import leave_tools as lt
from ..tools.errors import BULK_MESSAGE, AccessDenied, ToolRejected
from ..tools.permissions import visible_employee_ids
from .confirm import is_confirmation
from .llm import LLMProvider, LLMUnavailable, ToolUse
from .prompts import FINAL_SCHEMA, SYSTEM_PROMPT, TOOLS, requester_context
from .sessions import Proposal, Session

logger = logging.getLogger(__name__)

MAX_SEARCHES = 2
MAX_LLM_CALLS = 6
MIN_SEARCH_BUDGET = 150

ROUTES = {"rag", "tool", "hybrid", "clarify", "no_answer", "refuse"}
CONTRACT_TOOLS = {
    "get_leave_balance": ("employee_id", "leave_type"),
    "list_leave_requests": ("employee_id", "status"),
    "get_team_leave": ("start_date", "end_date"),
    "apply_leave": ("leave_type", "start_date", "end_date"),
    "cancel_leave": ("request_id",),
}
LEAVE_TYPE_WORDS = {
    "annual": re.compile(r"\b(annual|vacation|vacations|holiday|holidays|pto|paid time off|earned leave|"
                         r"privilege leave|urlaub)\b", re.IGNORECASE),
    "sick": re.compile(r"\b(sick|sickness|ill|illness|unwell|medical|doctor|flu|fever|surgery|hospital)\b",
                       re.IGNORECASE),
}

REFUSE_TEXT = ("I can't help with that. I can answer questions about Veltrix HR policies and holidays, "
               "and help you view or manage your leave.")
NO_ANSWER_TEXT = "The HR documents don't cover this question. Please contact the HR team for help."
UNAVAILABLE_TEXT = "I'm having trouble reaching the language model right now. Please try again in a moment."
CONFIRM_PROMPT = '\n\nReply "yes" to confirm, or tell me what to change.'


@dataclass
class TurnState:
    corpus_tokens: int = 0
    searches: int = 0
    next_ref: int = 1
    hits: dict[str, SearchHit] = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    other_targets: set[str] = field(default_factory=set)
    proposal: Proposal | None = None
    llm_calls: int = 0


@dataclass
class Outcome:
    route: str
    answer: str
    cited_refs: list[str] = field(default_factory=list)
    kind: str = "llm"  # llm | access_denied | proposal | write_rejected | executed | error


def _record_args(name: str, args: dict) -> dict:
    return {k: str(args[k]).strip() for k in CONTRACT_TOOLS[name] if args.get(k) not in (None, "")}


def _day(value: str) -> str:
    d = date.fromisoformat(value)
    return f"{d.strftime('%a')} {d.isoformat()}"


class ChatService:
    def __init__(self, settings: Settings, kb: KnowledgeBase, provider_factory) -> None:
        self.settings = settings
        self.kb = kb
        self.retriever = Retriever(kb)
        self._provider_factory = provider_factory
        self._provider: LLMProvider | None = None

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = self._provider_factory()
        return self._provider

    # ------------------------------------------------------------------ entry

    def handle(self, *, request_id: str, session_id: str, session: Session, requester: Employee,
               message: str) -> dict:
        started = time.monotonic()
        today = self.settings.today()
        ctx = lt.ToolContext(db_path=self.settings.db_path, requester=requester, today=today, kb=self.kb)
        state = TurnState()
        with session.lock:
            proposal, session.pending = session.pending, None  # at most one, and only for the next message
            try:
                if proposal is not None and is_confirmation(message):
                    outcome = self._execute_proposal(ctx, proposal, state)
                else:
                    outcome = self._run_agent(request_id, ctx, session, message, state, started)
            except LLMUnavailable as exc:
                log_event(logger, "llm_unavailable", logging.ERROR, request_id=request_id, error=str(exc))
                outcome = Outcome("no_answer", UNAVAILABLE_TEXT, kind="error")
            except Exception as exc:  # never leak internals; the log records the type only
                log_event(logger, "turn_failed", logging.ERROR, request_id=request_id, error=type(exc).__name__)
                state.tool_calls = [c for c in state.tool_calls if c["status"] != "pending_confirmation"]
                outcome = Outcome("no_answer", "Sorry, something went wrong. Please try again.", kind="error")
            response = self._finalise(request_id, ctx, outcome, state)
            if outcome.kind == "proposal":
                session.pending = state.proposal
            session.remember(message, response["answer"])
        log_event(logger, "chat_response", request_id=request_id, session_id=session_id,
                  employee_id=requester.employee_id, route=response["route"],
                  citations=len(response["citations"]),
                  tool_calls=[(t["name"], t["status"]) for t in response["tool_calls"]],
                  pending_confirmation=response["pending_confirmation"], llm_calls=state.llm_calls,
                  corpus_tokens=state.corpus_tokens, latency_ms=int((time.monotonic() - started) * 1000))
        return response

    # -------------------------------------------------------------- the agent

    def _run_agent(self, request_id: str, ctx: lt.ToolContext, session: Session, message: str,
                   state: TurnState, started: float) -> Outcome:
        system = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": requester_context(ctx.requester, ctx.today)},
        ]
        messages: list[dict] = [*session.history, {"role": "user", "content": message}]
        deadline = started + self.settings.request_deadline_s
        for call_no in range(1, MAX_LLM_CALLS + 1):
            remaining = deadline - time.monotonic()
            if remaining < 4:
                break
            if state.corpus_tokens > CORPUS_TOKEN_LIMIT:  # cannot happen: search enforces it
                raise RuntimeError("corpus token budget exceeded")
            t0 = time.monotonic()
            resp = self.provider.complete(system=system, messages=messages, tools=TOOLS,
                                          output_schema=FINAL_SCHEMA, allow_tools=call_no < MAX_LLM_CALLS,
                                          timeout=max(3.0, min(self.settings.llm_timeout_s, remaining - 1)))
            state.llm_calls += 1
            log_event(logger, "llm_call", request_id=request_id, call=call_no,
                      model=getattr(self.provider, "model", "unknown"), corpus_tokens=state.corpus_tokens,
                      stop_reason=resp.stop_reason, latency_ms=int((time.monotonic() - t0) * 1000), **resp.usage)
            if resp.stop_reason == "refusal":
                return Outcome("refuse", REFUSE_TEXT)
            if resp.tool_uses:
                messages.append({"role": "assistant", "content": resp.assistant_content})
                results = []
                for tool_use in resp.tool_uses:
                    result = self._dispatch(request_id, ctx, session, message, tool_use, state)
                    if isinstance(result, Outcome):
                        return result
                    content, is_error = result
                    block = {"type": "tool_result", "tool_use_id": tool_use.id, "content": content}
                    if is_error:
                        block["is_error"] = True
                    results.append(block)
                messages.append({"role": "user", "content": results})
                continue
            return self._parse_final(resp.text)
        return Outcome("no_answer", "Sorry, I couldn't finish answering in time. Please try again.", kind="error")

    @staticmethod
    def _parse_final(text: str) -> Outcome:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", text or "", re.DOTALL)
            try:
                data = json.loads(match.group(0)) if match else None
            except json.JSONDecodeError:
                data = None
        if not isinstance(data, dict) or data.get("route") not in ROUTES or not isinstance(data.get("answer"), str):
            return Outcome("no_answer", NO_ANSWER_TEXT)
        refs = [r for r in data.get("citations") or [] if isinstance(r, str)]
        return Outcome(data["route"], data["answer"].strip(), refs)

    # ------------------------------------------------------------------ tools

    def _dispatch(self, request_id: str, ctx: lt.ToolContext, session: Session, message: str,
                  tool_use: ToolUse, state: TurnState):
        name = tool_use.name
        args = tool_use.input if isinstance(tool_use.input, dict) else {}
        if name == "search_policies":
            return self._search(ctx, args, state)
        if name == "find_employee":
            return self._find_employee(ctx, args)
        if name not in CONTRACT_TOOLS:
            return f"Unknown tool: {name}", True

        record = {"name": name, "args": _record_args(name, args)}
        me = ctx.requester
        try:
            if name in ("get_leave_balance", "list_leave_requests"):
                target = str(args.get("employee_id") or "").strip()
                if not target:
                    raise ToolRejected("employee_id is required.")
                if target != me.employee_id and state.other_targets and target not in state.other_targets:
                    raise AccessDenied(BULK_MESSAGE)  # one other employee per message
                if name == "get_leave_balance":
                    result = lt.get_leave_balance(ctx, target, args.get("leave_type"))
                else:
                    result = lt.list_leave_requests(ctx, target, args.get("status"))
                if target != me.employee_id:
                    state.other_targets.add(target)
                self._record(request_id, state, record, "success", target)
                return json.dumps(result), False

            if name == "get_team_leave":
                result = lt.get_team_leave(ctx, args.get("start_date"), args.get("end_date"))
                self._record(request_id, state, record, "success")
                return json.dumps(result), False

            if name == "apply_leave":
                leave_type = str(args.get("leave_type") or "").lower()
                if not self._leave_type_stated(leave_type, session, message):
                    # Internal guard, not a rejection: the model must ask instead of assuming.
                    return ("The user has not said whether this is annual or sick leave. Ask them which one; "
                            "do not assume."), True
                plan = lt.check_apply(ctx, leave_type, args.get("start_date"), args.get("end_date"))
                self._record(request_id, state, record, "pending_confirmation")
                state.proposal = Proposal("apply_leave", record["args"])
                return Outcome("tool", "Here is the leave request I'm ready to submit:\n" + plan.summary()
                               + CONFIRM_PROMPT, kind="proposal")

            # cancel_leave
            plan = lt.check_cancel(ctx, args.get("request_id"))
            self._record(request_id, state, record, "pending_confirmation")
            state.proposal = Proposal("cancel_leave", record["args"])
            return Outcome("tool", "I'm ready to cancel this leave request:\n" + plan.summary() + CONFIRM_PROMPT,
                           kind="proposal")
        except AccessDenied as exc:
            log_event(logger, "tool_access_denied", request_id=request_id, tool=name,
                      employee_id=me.employee_id)
            return Outcome("refuse", exc.message, kind="access_denied")
        except ToolRejected as exc:
            self._record(request_id, state, record, "error")
            if name == "apply_leave":
                return Outcome("tool", f"I can't submit that leave request. {exc.reason}", kind="write_rejected")
            if name == "cancel_leave":
                return Outcome("tool", f"I can't cancel that leave request. {exc.reason}", kind="write_rejected")
            return f"Error: {exc.reason}", True

    def _record(self, request_id: str, state: TurnState, record: dict, status: str, target: str | None = None):
        state.tool_calls.append({**record, "status": status})
        log_event(logger, "tool_call", request_id=request_id, tool=record["name"], status=status,
                  target_employee_id=target)

    @staticmethod
    def _leave_type_stated(leave_type: str, session: Session, message: str) -> bool:
        pattern = LEAVE_TYPE_WORDS.get(leave_type)
        if pattern is None:
            return True  # invalid types are rejected by the tool itself
        texts = [message, *session.recent_user_messages(3)]
        return any(pattern.search(t) for t in texts)

    def _search(self, ctx: lt.ToolContext, args: dict, state: TurnState):
        if state.searches >= MAX_SEARCHES:
            return "Search limit reached for this message. Answer from the results you already have.", True
        remaining = CORPUS_TOKEN_LIMIT - state.corpus_tokens
        if remaining < MIN_SEARCH_BUDGET:
            return "The document budget for this message is used up. Answer from the results you have.", True
        query = str(args.get("query") or "").strip()[:500]
        if not query:
            return "query is required.", True
        reference = ctx.today
        raw = args.get("reference_date")
        if isinstance(raw, str) and raw.strip():
            try:
                reference = min(date.fromisoformat(raw.strip()), ctx.today)
            except ValueError:
                pass
        result = self.retriever.search(query, role=ctx.requester.role, region=ctx.requester.region,
                                       reference_date=reference, today=ctx.today, token_budget=remaining,
                                       first_ref=state.next_ref)
        state.searches += 1
        state.corpus_tokens += result.tokens
        state.next_ref += len(result.hits)
        state.hits.update({h.ref: h for h in result.hits})
        return result.text, False

    def _find_employee(self, ctx: lt.ToolContext, args: dict):
        name = str(args.get("name") or "").strip()
        if not name:
            return "name is required.", True
        with repo.session(ctx.db_path) as conn:
            visible = visible_employee_ids(conn, ctx.requester)
            matches = repo.find_employees_by_name(conn, name, visible)
        payload = [{"employee_id": m.employee_id, "full_name": m.full_name, "region": m.region, "role": m.role}
                   for m in matches[:5]]
        return json.dumps({"matches": payload}), False

    # ------------------------------------------------------- confirmed writes

    def _execute_proposal(self, ctx: lt.ToolContext, proposal: Proposal, state: TurnState) -> Outcome:
        record = {"name": proposal.name, "args": dict(proposal.args)}
        try:
            if proposal.name == "apply_leave":
                plan, request_id = lt.execute_apply(ctx, proposal.args.get("leave_type"),
                                                    proposal.args.get("start_date"), proposal.args.get("end_date"))
                answer = (f"Done. Your {plan.leave_type} leave request from {_day(plan.start.isoformat())} to "
                          f"{_day(plan.end.isoformat())} ({plan.working_days} working day"
                          f"{'' if plan.working_days == 1 else 's'}) has been submitted and is pending approval. "
                          f"Request ID: {request_id}. Your {plan.leave_type} balance is now "
                          f"{plan.balance_after:g} days.")
            else:
                plan = lt.execute_cancel(ctx, proposal.args.get("request_id"))
                r = plan.request
                answer = (f"Done. Request {r['request_id']} ({r['leave_type']} leave, {r['start_date']} to "
                          f"{r['end_date']}) is cancelled, and {r['working_days']:g} working days went back to "
                          f"your {r['leave_type']} balance, which is now "
                          f"{plan.balance_before + r['working_days']:g} days.")
            state.tool_calls.append({**record, "status": "success"})
        except ToolRejected as exc:
            state.tool_calls.append({**record, "status": "error"})
            verb = "submit that leave request" if proposal.name == "apply_leave" else "cancel that leave request"
            answer = f"I couldn't {verb}. {exc.reason}"
        return Outcome("tool", answer, kind="executed")

    # --------------------------------------------------------------- response

    def _finalise(self, request_id: str, ctx: lt.ToolContext, outcome: Outcome, state: TurnState) -> dict:
        citations: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for ref in outcome.cited_refs:
            hit = state.hits.get(ref.strip())
            if hit is None:
                continue  # never cite something that was not retrieved for this requester
            key = (hit.doc.doc_id, hit.chunk.locator)
            if key not in seen:
                seen.add(key)
                citations.append({"doc_id": key[0], "locator": key[1]})

        route, answer, tool_calls = outcome.route, outcome.answer, list(state.tool_calls)
        if outcome.kind == "access_denied":
            route, tool_calls, citations = "refuse", [], []
        elif outcome.kind in ("proposal", "write_rejected", "executed"):
            route, citations = "tool", []
        elif route in ("clarify", "no_answer", "refuse"):
            tool_calls, citations = [], []
            if route == "no_answer" and outcome.kind != "error" and "HR" not in answer:
                answer = (answer + " Please contact the HR team for help.").strip()
        elif citations and tool_calls:
            route = "hybrid"
        elif citations:
            route = "rag"
        elif tool_calls:
            route = "tool"
        else:
            # The model claimed a grounded answer but nothing backs it: do not guess.
            route, answer = "no_answer", NO_ANSWER_TEXT

        with repo.session(ctx.db_path) as conn:
            scrubber = AnswerScrubber(conn, ctx.requester)
            answer = scrubber.scrub(answer or REFUSE_TEXT)
            for call in tool_calls:
                call["args"] = {k: scrubber.scrub(v) for k, v in call["args"].items()}
        return {
            "request_id": request_id,
            "answer": answer,
            "route": route,
            "citations": citations,
            "tool_calls": tool_calls,
            "pending_confirmation": outcome.kind == "proposal",
        }
