"""The agent loop, written against Groq's OpenAI-compatible chat API.

This is the standard function-calling loop: send the conversation, if the model
replies with tool calls run them and append the results, then send again, until
the model answers with no tool call. We write the loop by hand here (Groq has no
auto tool-runner), which is the common industry pattern for OpenAI-compatible
APIs — and it keeps the request → tool → result cycle in plain view.
"""

import json

from groq import Groq

from .tools import DISPATCH, TOOL_SCHEMAS

DEFAULT_MODEL = "openai/gpt-oss-20b"  # smallest Groq-hosted model that still supports tool calling (lightest on the free TPM cap)
MAX_TOKENS = 4096

SYSTEM = (
    "You are Mini Muse, a tool-using assistant. Break a task into steps and use "
    "the tools to search, compute, and read files. When you have the answer, "
    "state it directly."
)


class UsageTracker:
    """Running total of tokens across the whole run.

    Groq's free tier costs nothing, so there's no dollar figure to show — but
    token counts still matter (the free tier is rate-limited by tokens/minute),
    and tracking them is the Stage 1 'cost tracking' skill we'll build on later.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage):
        if usage is None:
            return
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens

    def summary(self) -> str:
        return (
            f"tokens: {self.prompt_tokens} in / {self.completion_tokens} out  "
            f"(Groq free tier, $0.00)"
        )


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


def run_agent(task: str, model: str = DEFAULT_MODEL, max_steps: int = 10, verbose: bool = False) -> None:
    """Drive the loop for a single task and print the final answer."""
    client = Groq()
    usage = UsageTracker()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",  # let the model decide whether to call a tool
        )
        usage.add(response.usage)
        message = response.choices[0].message

        # No tool calls means the model is done — print its answer and stop.
        if not message.tool_calls:
            print(message.content)
            break

        # Keep the assistant turn (with its tool_calls) in history, then answer
        # each tool call with a "tool" message carrying the result.
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })
        for tc in message.tool_calls:
            if verbose:
                print(f"[tool] {tc.function.name}({tc.function.arguments})")
            result = run_tool(tc.function.name, tc.function.arguments)
            if verbose:
                print(f"[result] {result[:300]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        print(f"[stopped: reached the {max_steps}-step limit without finishing]")

    print("\n" + usage.summary())
