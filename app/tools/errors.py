"""Tool outcomes that are not plain results."""
from __future__ import annotations

NOT_PERMITTED_MESSAGE = "I can't share information about that employee."
BULK_MESSAGE = ("I can't share leave information about several employees at once. "
                "I can help with one employee at a time.")


class AccessDenied(Exception):
    """The requester may not see this data. Ends the turn with route `refuse`.

    The message is fixed, so a missing employee and a forbidden one look the same.
    """

    def __init__(self, message: str = NOT_PERMITTED_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


class ToolRejected(Exception):
    """A business rule or argument check failed; the reason is shown to the user."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
