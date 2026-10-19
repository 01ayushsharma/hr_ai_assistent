"""In-memory conversation sessions.

A session belongs to the employee who sent its first message (Section 7.1).
It holds the text of earlier turns and at most one write proposal waiting for
confirmation (Section 8.2).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

MAX_HISTORY_MESSAGES = 12


class SessionOwnershipError(Exception):
    pass


@dataclass
class Proposal:
    name: str  # "apply_leave" | "cancel_leave"
    args: dict


@dataclass
class Session:
    owner_id: str
    history: list[dict] = field(default_factory=list)
    pending: Proposal | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def remember(self, user_message: str, answer: str) -> None:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        del self.history[:-MAX_HISTORY_MESSAGES]

    def recent_user_messages(self, n: int = 3) -> list[str]:
        return [m["content"] for m in self.history if m["role"] == "user"][-n:]


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, employee_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._sessions[session_id] = Session(owner_id=employee_id)
            elif session.owner_id != employee_id:
                raise SessionOwnershipError(session_id)
            return session
