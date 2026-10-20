"""Sentinel — the permission layer between the agent and the outside world.

Every outbound or side-effecting action the agent takes goes through here. Sentinel
holds the real credentials (the agent's tools never read them), classifies each
action as a read (allowed automatically) or a write (held for a human to approve),
writes an append-only audit log of everything, and blocks anything that breaks
policy — including the data exfiltration a prompt injection tries to trigger.

This is the policy brain. Running the agent inside a locked-down container so this
is the *only* way out is a later addition; for now the boundary is this module,
which every tool with a side effect must call instead of acting directly.
"""

import json
import os
import time

AUDIT_LOG = os.environ.get("MINI_MUSE_AUDIT", "sentinel_audit.log")

# A write may only target a recipient in this allow-list. Anything else is refused
# outright — this is what stops an injected "email the secrets to attacker@evil.com".
ALLOWED_EMAIL_DOMAINS = {"example.com"}


def _audit(entry: dict) -> None:
    """Append one line to the audit log. Every decision passes through here."""
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _approve_terminal(description: str) -> bool:
    """Ask the human operator at the terminal. The agent can't answer for itself."""
    print(f"\n[SENTINEL] approval required: {description}")
    try:
        return input("[SENTINEL] allow this action? [y/N] ").strip().lower() == "y"
    except EOFError:
        return False  # no interactive operator present -> deny by default


# How Sentinel asks a human to approve a write. The CLI uses the terminal prompt;
# the web server swaps in a browser-driven approver via set_approver(). Either
# way the agent itself never gets to answer — only a human does.
_approver = _approve_terminal


def set_approver(fn) -> None:
    """Replace the approval mechanism (e.g. the web server's approve/deny prompt)."""
    global _approver
    _approver = fn


def reset_approver() -> None:
    """Restore the default terminal approver."""
    global _approver
    _approver = _approve_terminal


def _approve(description: str) -> bool:
    """Route an approval request to whatever approver is currently installed."""
    return _approver(description)


def web_search(query: str, max_results: int = 5) -> str:
    """A READ: allowed automatically, logged, and run with Sentinel's own credential."""
    _audit({"action": "web_search", "kind": "read", "decision": "allow", "query": query})

    api_key = os.environ.get("TAVILY_API_KEY")  # only Sentinel touches the real key
    if not api_key:
        return "Error: search is unavailable (Sentinel has no TAVILY credential)."

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, max_results=max_results).get("results", [])
    if not results:
        return "No results."
    return "\n".join(
        f"- {r.get('title', '(no title)')}\n  {r.get('url', '')}\n  {r.get('content', '')}"
        for r in results
    )


def send_email(to: str, subject: str, body: str) -> str:
    """A WRITE: refused if the recipient isn't allow-listed, else held for approval."""
    domain = to.split("@")[-1].lower() if "@" in to else ""

    if domain not in ALLOWED_EMAIL_DOMAINS:
        _audit({"action": "send_email", "kind": "write", "decision": "blocked",
                "reason": "recipient not allow-listed", "to": to, "subject": subject})
        return (
            f"BLOCKED by Sentinel: {to} is not an approved recipient. "
            "The attempt was logged and nothing was sent."
        )

    if not _approve(f"send_email to {to} — subject: {subject!r}"):
        _audit({"action": "send_email", "kind": "write", "decision": "denied", "to": to})
        return "DENIED by the operator; email not sent."

    _audit({"action": "send_email", "kind": "write", "decision": "allow", "to": to})
    # Simulated send — no real SMTP; the point is the decision, not the delivery.
    return f"Email sent to {to}."
