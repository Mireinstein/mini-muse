"""Rebuild api/corpus.txt from Admire's material.

Run this locally whenever the resume or projects change, then commit the updated
corpus and redeploy:

    pip install pypdf          # only needed to run this script
    python build_corpus.py

It pulls the resume PDF and public GitHub repos, combines them with docs/about.md,
and writes the single corpus file the serverless function reads. The deploy itself
needs none of this — only api/chat.py, api/corpus.txt, and the `groq` package.
"""

import json
import os
import urllib.request

RESUME_URL = "https://admiremadyira.com/resume.pdf"
GITHUB_USER = "Mireinstein"
HERE = os.path.dirname(os.path.abspath(__file__))


def _get(url: str, headers: dict) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def _resume_text() -> str:
    from pypdf import PdfReader  # local-only dependency

    tmp = os.path.join(HERE, "docs", "resume.pdf")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(_get(RESUME_URL, {"User-Agent": "mini-muse"}))
    reader = PdfReader(tmp)
    return "\n".join(p.extract_text() or "" for p in reader.pages).strip()


def _github_text() -> str:
    raw = _get(
        f"https://api.github.com/users/{GITHUB_USER}/repos?per_page=100&sort=updated",
        {"User-Agent": "mini-muse", "Accept": "application/vnd.github+json"},
    )
    lines = ["# Projects (GitHub)"]
    for r in json.loads(raw):
        if r.get("fork"):
            continue
        lang = f" ({r['language']})" if r.get("language") else ""
        lines.append(f"\n## {r.get('name', '')}{lang}")
        lines.append(r.get("description") or "(no description)")
        lines.append(f"Repository: {r.get('html_url', '')}")
    return "\n".join(lines)


def main() -> None:
    parts = []
    with open(os.path.join(HERE, "docs", "about.md"), encoding="utf-8") as f:
        parts.append(f.read().strip())
    try:
        parts.append("# Resume\n\n" + _resume_text())
    except Exception as exc:
        print(f"  (skipped resume: {exc})")
    try:
        parts.append(_github_text())
    except Exception as exc:
        print(f"  (skipped GitHub: {exc})")

    corpus = "\n\n---\n\n".join(parts)
    out = os.path.join(HERE, "api", "corpus.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"wrote {out} ({len(corpus)} chars)")


if __name__ == "__main__":
    main()
