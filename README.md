# Mini Muse

A personal agent that uses tools, works in the background, answers from your own
notes, and runs every risky action past a permission layer.

Built with Python, the Anthropic API, FAISS, SQLite, and Docker — from scratch,
with no agent framework.

## Features

- **Tool-using agent loop:** The model picks a tool, runs it, feeds the result
  back, and repeats until the task is done. Three tools — web search, a Python
  sandbox, and a file reader — driven from a CLI with a step limit and clear
  errors on malformed tool calls.
- **Goals and background work:** Turns a goal into a stored plan, runs the steps
  on a background worker, and persists state to SQLite so work survives a crash
  or restart. Failed steps retry and surface a clear failure state; every step
  and tool call is logged.
- **RAG over your documents:** Ingests notes and PDFs, chunks and embeds them
  into a vector index, and retrieves with reranking before answering. Every
  answer cites the passage it came from.
- **Sentinel permission layer:** The agent runs sandboxed with no direct network
  access while a separate proxy carries the real credentials. Reads pass
  automatically, writes such as sending an email or deleting a file wait for
  approval, every action is audited, and prompt-injection attempts are blocked.

## Quick Start

Requires Python 3.11+ and a model API key.

### 1. Install

```bash
git clone <repo-url>
cd mini-muse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

### 2. Run

```bash
# Answer a multi-step question (search, compute, summarize)
python -m mini_muse "What's the average of the last three US CPI year-over-year figures?"

# Give it a goal to run in the background
python -m mini_muse --goal "Draft a weekly summary of my project notes" --background

# Ask a question answered from your own documents, with a citation
python -m mini_muse "What did I decide about the chunking strategy?" --docs ./notes
```

## Tech Stack

- **Language:** Python 3.11+
- **Model:** Anthropic API (Claude)
- **Retrieval:** FAISS vector index with sentence-transformer embeddings
- **Persistence:** SQLite for plans, tasks, and audit logs
- **Isolation:** Docker container with an outbound proxy (Sentinel)
