# Mini Muse

A personal AI assistant you chat with in the browser. It uses tools, works on
goals in the background, answers from your own notes with citations, and runs
every risky action past a permission layer before it happens.

Built from scratch with Python, the Groq API, and FastAPI on the backend and a
React + TypeScript front end — with no agent framework.

## Features

- **Chat web app:** A streaming chat interface that shows each tool the assistant
  uses the moment it runs, renders answers as Markdown, chips the document
  sources behind an answer, and surfaces permission prompts inline. React +
  TypeScript + Tailwind, served by a FastAPI backend over Server-Sent Events.
- **Tool-using agent loop:** The model picks a tool, runs it, feeds the result
  back, and repeats until the task is done — web search, a Python sandbox, a file
  reader, and a to-do/reminders store. The loop is written by hand, with a step
  limit and clear errors on malformed tool calls.
- **Goals and background work:** Turns a goal into a stored plan and runs the
  steps on a background worker, persisting state to SQLite so work survives a
  crash or restart. A jobs panel shows each step's progress live and keeps going
  after you leave the page; failed steps retry and surface a clear failure state.
- **RAG over your documents:** Ingests notes and PDFs, chunks and embeds them into
  a vector index, and retrieves the relevant passages before answering. Every
  answer cites the source it came from.
- **Sentinel permission layer:** Holds the real credentials so the agent's tools
  never read them, classifies each action as a read (allowed automatically) or a
  write (held for your approval), audits everything to an append-only log, and
  blocks the data exfiltration a prompt injection tries to trigger. In the web
  app, write approvals appear as inline Approve / Deny.

## Quick start

Requires Python 3.11+, Node 18+, and a Groq API key (plus a Tavily key for web
search).

### 1. Install

```bash
git clone <repo-url>
cd mini-muse

# backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
export TAVILY_API_KEY=tvly-...

# frontend
cd frontend && npm install && cd ..
```

### 2. Run the web app

```bash
# terminal 1 — backend API
python -m mini_muse --serve

# terminal 2 — frontend dev server (open the URL it prints)
cd frontend && npm run dev
```

For a single-origin build, `cd frontend && npm run build`, then
`python -m mini_muse --serve` serves the built app directly.

### Or use the command line

```bash
# Answer a multi-step question (search, compute, summarize)
python -m mini_muse "What's the average of the last three US CPI year-over-year figures?"

# Turn a goal into a stored plan and run it (resume after a stop with --resume)
python -m mini_muse --goal "Research the three largest cities in Japan and rank them by population"

# Answer from your own documents, with a citation
# (drop .md/.txt/.pdf files in ./docs first)
python -m mini_muse --ingest
python -m mini_muse "What did I decide about the chunking strategy?"
```

## Tech stack

- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend / API:** FastAPI with Server-Sent Events for streaming
- **Language:** Python 3.11+
- **Model:** Groq API (gpt-oss-20b)
- **Retrieval:** Chroma vector store with sentence-transformer embeddings
- **Persistence:** SQLite for plans, tasks, and audit logs
