"""Assemble Admire's material into the docs/ folder for indexing.

The website itself is JavaScript-rendered and mostly links out, so the answerable
content is: a hand-written About note (committed as docs/about.md), the resume PDF
served at admiremadyira.com/resume.pdf, and public GitHub project descriptions.
`sync` fetches the latter two; both are best-effort, so indexing still works
offline from whatever is already in docs/.
"""

import json
import os
import urllib.request

RESUME_URL = "https://admiremadyira.com/resume.pdf"
GITHUB_USER = "Mireinstein"


def _get(url: str, headers: dict | None = None, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _sync_resume(folder: str) -> str | None:
    """Download the resume PDF into docs/. Returns the path, or None on failure."""
    try:
        data = _get(RESUME_URL, headers={"User-Agent": "mini-muse"})
        path = os.path.join(folder, "resume.pdf")
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception as exc:  # network down, 404, etc. — keep going
        print(f"  (skipped resume: {exc})")
        return None


def _sync_github(folder: str) -> str | None:
    """Write public repo names + descriptions to docs/projects.md. Best-effort."""
    try:
        raw = _get(
            f"https://api.github.com/users/{GITHUB_USER}/repos?per_page=100&sort=updated",
            headers={"User-Agent": "mini-muse", "Accept": "application/vnd.github+json"},
        )
        repos = json.loads(raw)
    except Exception as exc:
        print(f"  (skipped GitHub: {exc})")
        return None

    lines = ["# Projects (GitHub)\n"]
    for r in repos:
        if r.get("fork"):
            continue
        name = r.get("name", "")
        desc = r.get("description") or "(no description)"
        lang = r.get("language") or ""
        url = r.get("html_url", "")
        lines.append(f"## {name}" + (f" ({lang})" if lang else ""))
        lines.append(desc)
        lines.append(f"Repository: {url}\n")

    if len(lines) == 1:  # no non-fork repos
        return None
    path = os.path.join(folder, "projects.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def sync(folder: str = "docs") -> list[str]:
    """Pull the resume and GitHub projects into `folder`. Returns the files written."""
    os.makedirs(folder, exist_ok=True)
    written = [p for p in (_sync_resume(folder), _sync_github(folder)) if p]
    for p in written:
        print(f"  synced {p}")
    return written
