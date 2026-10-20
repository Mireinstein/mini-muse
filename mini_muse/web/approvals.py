"""Browser-driven approvals for Sentinel.

On the CLI, Sentinel asks the operator to approve a write by prompting at the
terminal. On the web, there is no terminal — the request has to travel out to
the browser and the decision has to travel back. This module is that bridge: a
registry of in-flight approval requests, each guarding a `threading.Event` that
the agent's worker thread blocks on until the browser answers.
"""

import itertools
import threading

_counter = itertools.count(1)
_pending: dict[int, "_Pending"] = {}
_lock = threading.Lock()

APPROVAL_TIMEOUT = 300  # seconds to wait for a human before defaulting to deny


class _Pending:
    def __init__(self, description: str):
        self.description = description
        self.event = threading.Event()
        self.allowed = False


def create(description: str) -> int:
    """Register a new approval request and return its id."""
    approval_id = next(_counter)
    with _lock:
        _pending[approval_id] = _Pending(description)
    return approval_id


def wait(approval_id: int) -> bool:
    """Block until the browser resolves this request (or we time out -> deny)."""
    with _lock:
        pending = _pending.get(approval_id)
    if pending is None:
        return False
    answered = pending.event.wait(APPROVAL_TIMEOUT)
    with _lock:
        _pending.pop(approval_id, None)
    return pending.allowed if answered else False


def resolve(approval_id: int, allowed: bool) -> bool:
    """Record the browser's decision and unblock the waiting worker thread."""
    with _lock:
        pending = _pending.get(approval_id)
    if pending is None:
        return False
    pending.allowed = allowed
    pending.event.set()
    return True


def make_web_approver(emit):
    """Build a Sentinel approver that asks the browser via `emit` and waits.

    `emit` pushes an event onto the current chat's stream so the browser can show
    an Approve/Deny prompt; `wait` then blocks this thread until the answer POSTs
    back. The agent never gets to answer for itself — only the human does.
    """
    def approver(description: str) -> bool:
        approval_id = create(description)
        emit({"type": "approval_request", "id": approval_id, "description": description})
        return wait(approval_id)

    return approver
