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

from .. import memory, sentinel, storage
from ..agent import DEFAULT_MODEL, SYSTEM, stream_agent
from ..planner import make_plan
from ..worker import run_plan
from . import approvals

load_dotenv()  # pick up GROQ_API_KEY / TAVILY_API_KEY from a local .env

app = FastAPI(title="Mini Muse")

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

    storage.init_db()
    history = [{"role": m["role"], "content": m["content"]} for m in storage.get_messages(conversation_id)]

    # Surface what we already know about the user that's relevant to this message,
    # so the agent uses their preferences without being asked (like Muse's memory).
    try:
        recalled = memory.recall(user_message, k=3)
    except Exception:
        recalled = []
    preface = (
        [{"role": "system", "content": "Relevant things you remember about the user:\n"
          + "\n".join(f"- {r}" for r in recalled)}]
        if recalled else []
    )

    messages = [{"role": "system", "content": SYSTEM}, *preface, *history,
                {"role": "user", "content": user_message}]

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
            storage.add_message(conversation_id, "user", user_message)
            if answer:
                storage.add_message(conversation_id, "assistant", answer)
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


@app.get("/api/conversations")
async def conversations():
    """Every conversation with its message count — newest activity first."""
    storage.init_db()
    return JSONResponse({"conversations": [dict(c) for c in storage.list_conversations()]})


@app.get("/api/conversations/{conversation_id}")
async def conversation_history(conversation_id: str):
    """The stored messages for one conversation, so a page reload restores the chat."""
    storage.init_db()
    return JSONResponse({"messages": [dict(m) for m in storage.get_messages(conversation_id)]})


@app.get("/api/memories")
async def memories_list():
    """Everything the agent remembers about the user — a 'what do you know about me' view."""
    try:
        return JSONResponse({"memories": memory.all_memories()})
    except Exception as exc:
        return JSONResponse({"memories": [], "error": str(exc)})


@app.delete("/api/memories/{mem_id}")
async def memories_forget(mem_id: str):
    """Forget one memory by id (the user's 'forget this' control)."""
    return JSONResponse({"ok": memory.forget(mem_id)})


@app.get("/api/suggestions")
async def suggestions_endpoint(conversation_id: str = "default"):
    """Proactive next-step suggestions from memory + the recent conversation."""
    from ..proactive import suggestions as make_suggestions

    storage.init_db()
    history = [{"role": m["role"], "content": m["content"]} for m in storage.get_messages(conversation_id)]
    try:
        if history:
            mems = memory.recall(" ".join(m["content"] for m in history[-3:]), k=5)
        else:
            mems = [m["text"] for m in memory.all_memories()][:5]
    except Exception:
        mems = []
    try:
        return JSONResponse({"suggestions": make_suggestions(history, mems, model=DEFAULT_MODEL)})
    except Exception as exc:
        return JSONResponse({"suggestions": [], "error": str(exc)})


# Serve the built frontend (after `npm run build`) at a single origin, if present.
# Registered last so the /api routes above take precedence.
_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
