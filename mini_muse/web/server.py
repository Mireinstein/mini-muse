"""FastAPI server that puts a web chat in front of the Mini Muse agent.

The agent, Sentinel, RAG, and the background worker are reused unchanged; this
module only adds transport. The one subtlety is streaming: each chat runs the
agent loop in a worker thread that pushes events onto a queue, and the HTTP
response drains that queue as Server-Sent Events. That indirection is what lets
a Sentinel approval request reach the browser mid-run and the decision travel
back before the blocked tool call continues.
"""

import asyncio
import json
import os
import queue
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import sentinel, storage
from ..agent import DEFAULT_MODEL, SYSTEM, stream_agent
from ..planner import make_plan
from ..worker import run_plan
from . import approvals

load_dotenv()  # pick up GROQ_API_KEY / TAVILY_API_KEY from a local .env

app = FastAPI(title="Mini Muse")

# Per-conversation message history, in memory (persisting it is a later refinement).
_conversations: dict[str, list[dict]] = {}

# Sentinel marking the end of a chat's event stream.
_STREAM_DONE = object()


def _sse(event: dict) -> str:
    """Format one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat")
async def chat(request: Request):
    """Stream the agent's reply to one user message as Server-Sent Events."""
    body = await request.json()
    conversation_id = body.get("conversation_id", "default")
    user_message = body["message"]
    model = body.get("model", DEFAULT_MODEL)

    history = _conversations.setdefault(conversation_id, [])
    messages = [{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": user_message}]

    events: queue.Queue = queue.Queue()

    def run() -> None:
        # Route Sentinel approvals to this chat's stream while the run is live.
        sentinel.set_approver(approvals.make_web_approver(events.put))
        answer = ""
        try:
            for event in stream_agent(messages, model=model):
                events.put(event)
                if event["type"] == "answer":
                    answer = event["content"]
        except Exception as exc:  # surface the failure to the browser, don't hang
            events.put({"type": "error", "message": str(exc)})
        finally:
            sentinel.reset_approver()
            history.append({"role": "user", "content": user_message})
            if answer:
                history.append({"role": "assistant", "content": answer})
            events.put(_STREAM_DONE)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, events.get)
            if event is _STREAM_DONE:
                yield _sse({"type": "done"})
                return
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/approve")
async def approve(request: Request):
    """Resolve a pending Sentinel approval (the browser's Approve/Deny click)."""
    body = await request.json()
    ok = approvals.resolve(int(body["id"]), bool(body["allow"]))
    return JSONResponse({"ok": ok})


@app.post("/api/goals")
async def start_goal(request: Request):
    """Plan a goal into steps and run it in the background (the worker thread)."""
    body = await request.json()
    goal = body["goal"]
    model = body.get("model", DEFAULT_MODEL)

    storage.init_db()
    steps = make_plan(goal, model=model)
    plan_id = storage.create_plan(goal)
    storage.add_steps(plan_id, steps)
    threading.Thread(target=run_plan, args=(plan_id,), kwargs={"model": model}, daemon=True).start()
    return JSONResponse({"plan_id": plan_id, "steps": steps})


@app.get("/api/plans")
async def list_plans():
    """All plans, newest first — the jobs panel polls this for live progress."""
    storage.init_db()
    return JSONResponse({"plans": [dict(p) for p in storage.list_plans()]})


@app.get("/api/plans/{plan_id}")
async def plan_detail(plan_id: int):
    """One plan with its steps and their statuses."""
    storage.init_db()
    plan = storage.get_plan(plan_id)
    if plan is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"plan": dict(plan), "steps": [dict(s) for s in storage.plan_steps(plan_id)]})


# Serve the built frontend (after `npm run build`) at a single origin, if present.
# Registered last so the /api routes above take precedence.
_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
