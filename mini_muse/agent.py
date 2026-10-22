"""The agent loop, written against Groq's OpenAI-compatible chat API.

This is the standard function-calling loop: send the conversation, if the model
replies with tool calls run them and append the results, then send again, until
the model answers with no tool call. We write the loop by hand here (Groq has no
auto tool-runner), which is the common industry pattern for OpenAI-compatible
APIs — and it keeps the request → tool → result cycle in plain view.

The loop lives in `stream_agent`, a generator that yields an event per thing
that happens (a tool call, a tool result, a token of the answer). The web server
streams those events to the browser; `run_agent` drains them into a single
`AgentResult` for the CLI, worker, and evaluator.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from groq import Groq

from .tools import DISPATCH, TOOL_SCHEMAS

DEFAULT_MODEL = "openai/gpt-oss-20b"  # smallest Groq-hosted model that still supports tool calling (lightest on the free TPM cap)
MAX_TOKENS = 4096

SYSTEM = (
    "You are Mini Muse, a personal AI agent. You don't just answer questions — you "
    "carry out tasks using your tools, and you keep track of what matters to the user.\n\n"
    "Your tools: web_search and fetch_url (find and read web pages), run_python "
    "(calculate or process data), search_docs (the user's own notes and PDFs), "
    "read_file and write_file (local files), add_task / list_tasks / complete_task "
    "(reminders), remember / recall (long-term memory of the user's preferences and "
    "facts), and send_email (gated by a permission layer).\n\n"
    "How to work:\n"
    "- Break a goal into steps and use the tools to actually do it, not just describe it.\n"
    "- When the user shares a lasting preference or detail about themselves, call "
    "remember so you can use it in future conversations; recall what you know when relevant.\n"
    "- When you answer from the user's documents via search_docs, cite the source of "
    "each fact in brackets, e.g. [notes.md #3].\n"
    "- Actions that leave the machine (like send_email) pass through a permission layer "
    "that may ask the user to approve them — that is expected, not an error.\n"
    "- Be proactive: when you notice a useful next step, offer it briefly."
)


class UsageTracker:
    """Running total of tokens across the whole run.

    Groq's free tier costs nothing, so there's no dollar figure to show — but
    token counts still matter (the free tier is rate-limited by tokens/minute),
    and tracking them helps spot a run that's doing far more work than expected.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def summary(self) -> str:
        return (
            f"tokens: {self.prompt_tokens} in / {self.completion_tokens} out  "
            f"(Groq free tier, $0.00)"
        )


@dataclass
class AgentResult:
    """What one agent run produced — the answer plus a trace for evaluation."""
    answer: str
    usage: UsageTracker
    tools: list[str] = field(default_factory=list)  # names of tools called, in order


def run_tool(name: str, raw_arguments: str) -> str:
    """Run one tool call. Always returns a string — errors come back as text.

    The model sends arguments as a JSON string, so we parse it first. A bad JSON
    blob, an unknown tool, or wrong arguments all return an error string the model
    can read and retry, instead of crashing the run.
    """
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return f"Error: arguments for {name} were not valid JSON."

    func = DISPATCH.get(name)
    if func is None:
        return f"Error: unknown tool {name!r}."

    try:
        return func(**args)
    except TypeError as e:  # wrong or missing arguments for the function
        return f"Error: bad arguments for {name}: {e}"


def stream_agent(messages: list[dict], model: str = DEFAULT_MODEL, max_steps: int = 12) -> Iterator[dict]:
    """Drive the loop, yielding one event dict per thing that happens.

    `messages` must already include the system prompt and the user turn(s); it is
    mutated in place as the conversation grows (assistant tool calls and tool
    results are appended), which is what lets the loop carry context across turns.

    Event types yielded:
      {"type": "token", "text": ...}                  a piece of the answer text
      {"type": "tool_call", "name": ..., "args": ...}  the model asked for a tool
      {"type": "tool_result", "name": ..., "result": ...}  what the tool returned
      {"type": "usage", "prompt_tokens": ..., "completion_tokens": ...}
      {"type": "answer", "content": ...}               the final answer, once
    """
    client = Groq()

    for _ in range(max_steps):
        # stream=True so we can emit tokens and tool calls as they arrive. Groq
        # attaches token counts to the final chunk's `.usage` on its own.
        stream = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",  # let the model decide whether to call a tool
            stream=True,
        )

        content_parts: list[str] = []
        tool_acc: dict[int, dict] = {}  # index -> partial {id, name, args}, assembled across chunks

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                yield {"type": "usage", "prompt_tokens": usage.prompt_tokens,
                       "completion_tokens": usage.completion_tokens}
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content_parts.append(delta.content)
                yield {"type": "token", "text": delta.content}

            # Tool calls stream as fragments keyed by index; stitch them together.
            for tc in (delta.tool_calls or []):
                slot = tool_acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments

        content = "".join(content_parts)

        # No tool calls means the model is done — emit the final answer and stop.
        if not tool_acc:
            yield {"type": "answer", "content": content}
            return

        # Keep the assistant turn (with its tool_calls) in history, then run each
        # call and append a "tool" message carrying the result.
        calls = [tool_acc[i] for i in sorted(tool_acc)]
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"]}}
                for c in calls
            ],
        })
        for c in calls:
            yield {"type": "tool_call", "name": c["name"], "args": c["args"]}
            result = run_tool(c["name"], c["args"])
            yield {"type": "tool_result", "name": c["name"], "result": result}
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    yield {"type": "answer", "content": f"[reached the {max_steps}-step limit without finishing]"}


def run_agent(task: str, model: str = DEFAULT_MODEL, max_steps: int = 12,
              verbose: bool = False, history: list[dict] | None = None) -> AgentResult:
    """Drive the loop for a single task. Returns the answer, usage, and tool trace.

    It returns rather than printing, so the worker and the evaluator can capture
    each run's result; the CLI prints what it gets back. `history` lets a caller
    (e.g. the web server) carry prior conversation turns into the run.
    """
    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": task})

    usage = UsageTracker()
    tools_used: list[str] = []
    answer = ""

    for event in stream_agent(messages, model=model, max_steps=max_steps):
        kind = event["type"]
        if kind == "tool_call":
            tools_used.append(event["name"])
            if verbose:
                print(f"[tool] {event['name']}({event['args']})")
        elif kind == "tool_result":
            if verbose:
                print(f"[result] {event['result'][:300]}")
        elif kind == "usage":
            usage.add(event["prompt_tokens"], event["completion_tokens"])
        elif kind == "answer":
            answer = event["content"]

    return AgentResult(answer, usage, tools_used)
