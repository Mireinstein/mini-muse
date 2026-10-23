"""Rebuild api/corpus.txt and the retrieval index api/index.json from Admire's material.

Run this locally whenever the resume or projects change, then commit both files
and redeploy:

    pip install pypdf          # only needed to run this script
    python build_corpus.py     # needs GEMINI_API_KEY, from the environment or .env

It pulls the resume PDF and public GitHub repos, combines them with docs/about.md,
splits the result into chunks, and embeds each chunk once. The serverless function
only has to embed the incoming question and rank these stored vectors.
"""

import json
import os
import re
import urllib.request

RESUME_URL = "https://admiremadyira.com/resume.pdf"
GITHUB_USER = "Mireinstein"
HERE = os.path.dirname(os.path.abspath(__file__))

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768
CHUNK_CHARS = 800


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
    # The default mode breaks this PDF into one word per line; layout mode keeps
    # the lines intact but pads them with runs of spaces
    text = "\n".join(p.extract_text(extraction_mode="layout") or "" for p in reader.pages)
    return re.sub(r"[ \t]+", " ", text).strip()


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


def _load_env() -> None:
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            if key and not key.startswith("#"):
                os.environ.setdefault(key, value.strip("\"'"))


def chunk(corpus: str) -> list[dict]:
    """Split on markdown headings, then cap each section at CHUNK_CHARS on line
    boundaries. Every chunk carries its heading path so it reads on its own."""
    chunks = []
    h1 = h2 = ""
    lines: list[str] = []

    def flush() -> None:
        heading = f"{h1} > {h2}" if h2 else h1
        window: list[str] = []
        for line in lines + [None]:
            if line is None or (window and len("\n".join(window + [line])) > CHUNK_CHARS):
                text = "\n".join(window).strip()
                if text:
                    chunks.append({"heading": heading, "text": text})
                window = []
            if line is not None:
                window.append(line)
        lines.clear()

    for line in corpus.splitlines():
        if line.startswith("## "):
            flush()
            h2 = line[3:].strip()
        elif line.startswith("# "):
            flush()
            h1, h2 = line[2:].strip(), ""
        elif line.strip() != "---":
            lines.append(line)
    flush()
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    body = {
        "requests": [
            {
                "model": f"models/{EMBED_MODEL}",
                "content": {"parts": [{"text": t}]},
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": EMBED_DIMS,
            }
            for t in texts
        ]
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return [e["values"] for e in json.load(resp)["embeddings"]]


def main() -> None:
    _load_env()
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

    chunks = chunk(corpus)
    # Embed the heading with the text so a chunk like "Grey-Parrot (JavaScript)"
    # still matches a question about projects
    vectors = embed([f"{c['heading']}\n{c['text']}" for c in chunks])
    for c, v in zip(chunks, vectors):
        c["embedding"] = [round(x, 5) for x in v]
    out = os.path.join(HERE, "api", "index.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print(f"wrote {out} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
