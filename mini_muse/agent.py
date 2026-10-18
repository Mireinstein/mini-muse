"""The agent loop, driven by the Anthropic SDK's tool runner.

`client.beta.messages.tool_runner(...)` is the standard way to run a custom-tool
agent: you hand it your tools and the first message, and it does the loop for you
— call the model, if the model asks for a tool it runs the matching function,
feeds the result back, and calls the model again, repeating until the model
answers with no tool call. Iterating the runner yields each assistant message as
it happens, which is where we read usage (for cost) and the tool calls (for
--verbose). `max_iterations` is the step limit.
"""

from anthropic import Anthropic

from .tools import TOOLS

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000

SYSTEM = (
    "You are Mini Muse, a tool-using assistant. Break a task into steps and use "
    "the tools to search, compute, and read files. When you have the answer, "
    "state it directly."
)

# claude-sonnet-5 list price, dollars per million tokens. (Intro pricing through
# 2026-08-31 is $2 / $10 — swap these two if you want the cheaper estimate.)
PRICE_IN_PER_M = 3.0
PRICE_OUT_PER_M = 15.0


class CostTracker:
    """Running total of tokens and estimated dollar cost across the whole run."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, usage):
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def summary(self) -> str:
        cost = (
            self.input_tokens / 1_000_000 * PRICE_IN_PER_M
            + self.output_tokens / 1_000_000 * PRICE_OUT_PER_M
        )
        return (
            f"tokens: {self.input_tokens} in / {self.output_tokens} out  "
            f"estimated cost: ${cost:.4f}"
        )


def run_agent(task: str, model: str = DEFAULT_MODEL, max_steps: int = 10, verbose: bool = False) -> None:
    """Drive the loop for a single task and print the final answer."""
    client = Anthropic()
    cost = CostTracker()

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        tools=TOOLS,
        messages=[{"role": "user", "content": task}],
        max_iterations=max_steps,
    )

    # Each iteration is one assistant message. The runner runs any tools it asked
    # for before the next iteration, so by looping we also watch the tool calls go by.
    final = None
    for message in runner:
        cost.add(message.usage)
        final = message
        if verbose:
            for block in message.content:
                if block.type == "tool_use":
                    print(f"[tool] {block.name}({block.input})")

    # The last message is the model's final turn. If it still wanted a tool, the
    # runner stopped because it hit the step limit rather than because we finished.
    if final is not None:
        if final.stop_reason == "tool_use":
            print(f"[stopped: reached the {max_steps}-step limit without finishing]")
        for block in final.content:
            if block.type == "text":
                print(block.text)

    print("\n" + cost.summary())
