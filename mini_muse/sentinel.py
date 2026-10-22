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
# Configurable via MINI_MUSE_ALLOWED_EMAIL_DOMAINS (comma-separated); defaults to
# example.com so the injection-block demo keeps working out of the box.
ALLOWED_EMAIL_DOMAINS = {
    d.strip().lower()
    for d in os.environ.get("MINI_MUSE_ALLOWED_EMAIL_DOMAINS", "example.com").split(",")
    if d.strip()
}


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


FETCH_MAX_BYTES = 200_000  # cap extracted page text so one page can't flood the context


def _html_to_text(html: str) -> str:
    """Strip tags and script/style blocks, leaving visible text."""
    from html.parser import HTMLParser

    class _Extract(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: list[str] = []
            self.skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript"):
                self.skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript") and self.skip:
                self.skip -= 1

        def handle_data(self, data):
            if not self.skip and data.strip():
                self.parts.append(data.strip())

    parser = _Extract()
    try:
        parser.feed(html)
    except Exception:
        pass
    return "\n".join(parser.parts)


def fetch_url(url: str) -> str:
    """A READ: fetch a web page and return its visible text. Logged, allowed, outbound."""
    _audit({"action": "fetch_url", "kind": "read", "decision": "allow", "url": url})

    if not url.lower().startswith(("http://", "https://")):
        return "Error: only http(s) URLs are supported."

    import httpx

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=20,
                         headers={"User-Agent": "Mini-Muse/1.0"})
        resp.raise_for_status()
    except Exception as exc:
        return f"Error fetching {url}: {exc}"

    text = _html_to_text(resp.text)
    return text[:FETCH_MAX_BYTES] or "(no readable text found on the page)"


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
    return _deliver(to, subject, body)


def write_file(path: str, content: str) -> str:
    """A WRITE: creating or overwriting a local file is held for human approval."""
    if not _approve(f"write_file to {path} ({len(content)} bytes)"):
        _audit({"action": "write_file", "kind": "write", "decision": "denied", "path": path})
        return "DENIED by the operator; file not written."

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as exc:  # bad path / permissions — log and report, don't raise
        _audit({"action": "write_file", "kind": "write", "decision": "error",
                "path": path, "error": str(exc)})
        return f"Error writing {path}: {exc}"

    _audit({"action": "write_file", "kind": "write", "decision": "allow",
            "path": path, "bytes": len(content)})
    return f"Wrote {len(content)} bytes to {path}."


def _deliver(to: str, subject: str, body: str) -> str:
    """Send the approved email over SMTP, using Sentinel's own credential.

    Only Sentinel reads the SMTP credential, so the agent's tools never see it. If
    no credential is configured, the send is simulated — the approval and audit
    still happened, so the demo works without a mailbox.
    """
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not (sender and password):
        return f"Email to {to} approved (no SMTP credential set, so nothing was actually sent)."

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
    except Exception as exc:  # delivery failed after approval — log it and say so
        _audit({"action": "send_email", "kind": "write", "decision": "error",
                "to": to, "error": str(exc)})
        return f"Email to {to} was approved but delivery failed: {exc}"

    return f"Email sent to {to}."
