"""The three tools the agent can call, in the OpenAI/Groq function-calling style.

Each tool is two things: a *schema* in TOOL_SCHEMAS (what the model sees — the
name, description, and arguments, as JSON Schema) and a plain Python *function*.
DISPATCH maps a tool name to its function; the loop in agent.py looks the handler
up there and runs it.

Each function returns a string. On failure it returns an error string (rather
than raising), so a bad call comes back to the model as a readable result it can
recover from instead of stopping the run.
"""

import os
import subprocess
import sys

# run_python kills any snippet that runs longer than this (seconds). Stage 1 has
# no real sandbox — that's Stage 4 (Docker). This is just a plain subprocess, so
# the timeout is the only guard against a runaway script.
PYTHON_TIMEOUT = 15

# read_file refuses anything larger than this so a huge file can't blow up the
# context window.
MAX_FILE_BYTES = 100_000


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily and return a compact title/url/snippet list."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set, so web search is unavailable."

    from tavily import TavilyClient  # imported here so the key is only needed when searching

    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, max_results=max_results).get("results", [])
    if not results:
        return "No results."

    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '(no title)')}\n  {r.get('url', '')}\n  {r.get('content', '')}")
    return "\n".join(lines)


def run_python(code: str) -> str:
    """Run a Python snippet in a subprocess and return stdout/stderr/exit code."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"Error: code did not finish within {PYTHON_TIMEOUT}s and was killed."

    return f"exit code: {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def read_file(path: str) -> str:
    """Read a UTF-8 text file, with clear errors for missing/oversized/binary files."""
    if not os.path.isfile(path):
        return f"Error: no file at {path!r}."

    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        return f"Error: file is {size} bytes; the limit is {MAX_FILE_BYTES}."

    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return f"Error: {path!r} is not a UTF-8 text file."


# What the model sees. This is the OpenAI/Groq "tools" format: a list of function
# specs, each with a JSON-Schema `parameters` block. `required` lists the args the
# model must supply.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information and return a short list of results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {"type": "integer", "description": "How many results to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a short Python 3 script and return its stdout, stderr, and exit code. Use this for calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python source to run."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the local filesystem and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                },
                "required": ["path"],
            },
        },
    },
]

# name -> function. The agent calls DISPATCH[name](**args).
DISPATCH = {
    "web_search": web_search,
    "run_python": run_python,
    "read_file": read_file,
}
