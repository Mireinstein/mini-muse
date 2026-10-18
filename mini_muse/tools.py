"""The three tools the agent can call.

Each tool is a plain Python function wrapped with the SDK's @beta_tool
decorator. The decorator reads the function's type hints and docstring and
generates the JSON schema the model sees — so the `query: str` hint and the
"Args:" lines below *are* the tool's API description. The tool runner (see
agent.py) matches a model tool call to the right function and runs it for you.

Each function returns a string. On failure it returns an error string (rather
than raising), so a bad call comes back to the model as a readable result it can
recover from instead of stopping the run.
"""

import os
import subprocess
import sys

from anthropic import beta_tool

# run_python kills any snippet that runs longer than this (seconds). Stage 1 has
# no real sandbox — that's Stage 4 (Docker). This is just a plain subprocess, so
# the timeout is the only guard against a runaway script.
PYTHON_TIMEOUT = 15

# read_file refuses anything larger than this so a huge file can't blow up the
# context window (and the bill).
MAX_FILE_BYTES = 100_000


@beta_tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information and return a short list of results.

    Args:
        query: The search query.
        max_results: How many results to return.
    """
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


@beta_tool
def run_python(code: str) -> str:
    """Run a short Python 3 script and return its stdout, stderr, and exit code. Use this for calculations.

    Args:
        code: The Python source to run.
    """
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


@beta_tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the local filesystem and return its contents.

    Args:
        path: Path to the file.
    """
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


# The list handed to the tool runner.
TOOLS = [web_search, run_python, read_file]
