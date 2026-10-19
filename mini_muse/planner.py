"""Turn a goal into an ordered list of concrete steps.

This is one Groq call with no tools: we ask the model to break the goal into a
small, ordered list of steps and return them as JSON, so we get structured data
to store in SQLite rather than a prose paragraph we'd have to parse by hand.
"""

import json

from groq import Groq

from .agent import DEFAULT_MODEL

PLANNER_SYSTEM = (
    "You are a planner. Break the user's goal into 2 to 6 concrete, ordered steps "
    "that a tool-using agent (web search, Python, file reading) can each carry out. "
    'Return ONLY JSON of the form {"steps": ["first step", "second step", ...]}.'
)


def make_plan(goal: str, model: str = DEFAULT_MODEL) -> list[str]:
    """Ask the model to decompose `goal` into a list of step descriptions."""
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": goal},
        ],
        response_format={"type": "json_object"},  # ask Groq to guarantee valid JSON
    )
    data = json.loads(response.choices[0].message.content)
    steps = data.get("steps", [])
    if not steps:
        raise ValueError("The planner returned no steps.")
    return [str(s) for s in steps]
