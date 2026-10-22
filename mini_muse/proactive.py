"""Proactive suggestions — the agent offering useful next steps unprompted.

Muse surfaces suggestions without being asked. This produces a few short,
concrete suggestions from what the agent knows about the user plus the recent
conversation, as one cheap model call (no tools, no streaming).
"""

import json

from groq import Groq

from .agent import DEFAULT_MODEL

_SUGGEST_SYSTEM = (
    "You are the proactive side of a personal AI agent. Given what you know about "
    "the user and the recent conversation, propose up to 3 short, specific, useful "
    "next actions you could take for them. Each is one line, phrased as an action "
    "the agent could do (e.g. 'Draft a reply to the landlord', 'Remind you to renew "
    "your passport'). Return ONLY a JSON array of strings — no prose, no keys."
)


def suggestions(history: list[dict], memories: list[str], model: str = DEFAULT_MODEL) -> list[str]:
    """Up to three proactive suggestions, or [] when there's nothing to go on."""
    context: list[str] = []
    if memories:
        context.append("What you remember about the user:\n" + "\n".join(f"- {m}" for m in memories))
    if history:
        recent = history[-6:]
        context.append("Recent conversation:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in recent))
    if not context:
        return []

    client = Groq()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[
            {"role": "system", "content": _SUGGEST_SYSTEM},
            {"role": "user", "content": "\n\n".join(context)},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()

    # Prefer the requested JSON array; fall back to parsing lines if the model
    # wrapped it in prose or bullets.
    try:
        data = json.loads(text[text.index("["): text.rindex("]") + 1])
        if isinstance(data, list):
            return [str(s).strip() for s in data if str(s).strip()][:3]
    except (ValueError, json.JSONDecodeError):
        pass
    return [line.strip("-•* ").strip() for line in text.splitlines() if line.strip()][:3]
