"""FastAPI server for the Ask-Admire Q&A chat.

One endpoint does the work: it takes a visitor's question (and the recent turns of
the conversation for follow-ups), runs the retrieve-then-answer path in
`agent.answer_stream`, and relays the reply to the browser as Server-Sent Events.
The service is stateless — nothing is stored between requests — so it scales to
zero and needs no database.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..agent import DEFAULT_MODEL, answer_stream

load_dotenv()  # pick up GROQ_API_KEY from a local .env

app = FastAPI(title="Ask Admire")


def _sse(event: dict) -> str:
    """Format one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat")
async def chat(request: Request):
    """Stream the answer to one question as Server-Sent Events."""
    body = await request.json()
    question = body["message"]
    history = body.get("history", [])  # [{role, content}, ...] recent turns, optional
    model = body.get("model", DEFAULT_MODEL)

    def gen():
        try:
            yield from answer_stream(question, history=history, model=model)
        except Exception as exc:  # surface the failure instead of hanging the stream
            yield {"type": "error", "message": str(exc)}

    async def stream():
        loop = asyncio.get_event_loop()
        it = gen()
        while True:
            event = await loop.run_in_executor(None, lambda: next(it, None))
            if event is None:
                yield _sse({"type": "done"})
                return
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream")


# Serve the built frontend (after `npm run build`) at a single origin, if present.
# Registered last so the /api route above takes precedence.
_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
