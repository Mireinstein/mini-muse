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

# run_python kills any snippet that runs longer than this (seconds). There's no
# sandbox yet — this is a plain subprocess, so the timeout is the only guard
# against a runaway script.
PYTHON_TIMEOUT = 15

# read_file refuses anything larger than this so a huge file can't blow up the
# context window.
MAX_FILE_BYTES = 100_000


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Routed through Sentinel, which holds the real credential."""
    from . import sentinel
    return sentinel.web_search(query, max_results)


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Routed through Sentinel, which gates this write action."""
    from . import sentinel
    return sentinel.send_email(to, subject, body)


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


def search_docs(query: str, k: int = 5) -> str:
    """Search the user's indexed notes/PDFs and return matching passages with sources."""
    from .rag import search  # imported lazily so the embedding model only loads when used

    hits = search(query, k)
    if not hits:
        return "No indexed documents. Run `python -m mini_muse --ingest <folder>` first."

    blocks = []
    for h in hits:
        tag = f"{os.path.basename(h['source'])} #{h['chunk']}"
        blocks.append(f"[{tag}]\n{h['text']}")
    return "\n\n".join(blocks)


def add_task(text: str, due: str = "") -> str:
    """Save a to-do / reminder for the user. A local write — nothing leaves the machine."""
    from . import storage
    storage.init_db()  # make sure the tasks table exists on a fresh run
    task_id = storage.add_task(text, due or None)
    when = f" (due {due})" if due else ""
    return f"Saved task #{task_id}: {text}{when}"


def list_tasks() -> str:
    """List the user's open to-dos / reminders."""
    from . import storage
    storage.init_db()
    tasks = storage.list_tasks()
    if not tasks:
        return "No open tasks."
    return "\n".join(
        f"#{t['id']}: {t['text']}" + (f" (due {t['due']})" if t["due"] else "")
        for t in tasks
    )


def complete_task(id: int) -> str:
    """Mark one of the user's tasks done by its id."""
    from . import storage
    storage.init_db()
    ok = storage.complete_task(id)
    return f"Completed task #{id}." if ok else f"Error: no task #{id}."


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
            "name": "send_email",
            "description": "Send an email to a recipient. A write action; it is checked by the permission layer before sending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject."},
                    "body": {"type": "string", "description": "Email body."},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the user's own indexed notes and PDFs for relevant passages. Use this to answer questions about the user's documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "k": {"type": "integer", "description": "How many passages to return (default 5)."},
                },
                "required": ["query"],
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
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Save a to-do or reminder for the user. Use this when the user asks to be reminded of or to remember something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "What to remember/do."},
                    "due": {"type": "string", "description": "Optional when it's due, free-form e.g. 'tomorrow 5pm'."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List the user's open to-dos / reminders.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark one of the user's saved tasks as done, by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The task id to complete."},
                },
                "required": ["id"],
            },
        },
    },
]

# name -> function. The agent calls DISPATCH[name](**args).
DISPATCH = {
    "web_search": web_search,
    "run_python": run_python,
    "send_email": send_email,
    "search_docs": search_docs,
    "read_file": read_file,
    "add_task": add_task,
    "list_tasks": list_tasks,
    "complete_task": complete_task,
}
