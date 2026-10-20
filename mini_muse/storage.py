"""SQLite persistence for plans, steps, and logs.

Everything the agent is working on lives in a SQLite file on disk, so a goal
survives the process being killed and restarted. A plan has many ordered steps;
each step carries its own status, so on restart we can see exactly which ones
still need doing.
"""

import os
import sqlite3
import time

DB_PATH = os.environ.get("MINI_MUSE_DB", "mini_muse.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    goal       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'running',   -- running | done | failed
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id     INTEGER NOT NULL REFERENCES plans(id),
    idx         INTEGER NOT NULL,                 -- order within the plan
    description TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | failed
    result      TEXT,
    error       TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id    INTEGER,
    step_id    INTEGER,
    kind       TEXT NOT NULL,                     -- e.g. step_done, step_failed, retry
    content    TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    due        TEXT,                              -- free-form, e.g. "tomorrow 5pm"
    done       INTEGER NOT NULL DEFAULT 0,        -- 0 = open, 1 = completed
    created_at REAL NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows accessible by column name
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def create_plan(goal: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO plans (goal, created_at) VALUES (?, ?)", (goal, time.time())
        )
        return cur.lastrowid


def add_steps(plan_id: int, descriptions: list[str]) -> None:
    with connect() as conn:
        conn.executemany(
            "INSERT INTO steps (plan_id, idx, description, updated_at) VALUES (?, ?, ?, ?)",
            [(plan_id, i, d, time.time()) for i, d in enumerate(descriptions)],
        )


def reset_running_steps(plan_id: int) -> None:
    """A step left 'running' means the process died mid-step; put it back to pending."""
    with connect() as conn:
        conn.execute(
            "UPDATE steps SET status='pending', updated_at=? WHERE plan_id=? AND status='running'",
            (time.time(), plan_id),
        )


def next_pending_step(plan_id: int) -> sqlite3.Row | None:
    """The lowest-index step still waiting to run, or None if there are none."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM steps WHERE plan_id=? AND status='pending' ORDER BY idx LIMIT 1",
            (plan_id,),
        ).fetchone()


def mark_step(step_id: int, status: str, result: str | None = None, error: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE steps SET status=?, result=?, error=?, updated_at=? WHERE id=?",
            (status, result, error, time.time(), step_id),
        )


def bump_attempt(step_id: int) -> int:
    """Increment a step's attempt counter and return the new count."""
    with connect() as conn:
        conn.execute("UPDATE steps SET attempts = attempts + 1 WHERE id=?", (step_id,))
        return conn.execute("SELECT attempts FROM steps WHERE id=?", (step_id,)).fetchone()[0]


def set_plan_status(plan_id: int, status: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE plans SET status=? WHERE id=?", (status, plan_id))


def get_plan(plan_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()


def latest_unfinished_plan() -> sqlite3.Row | None:
    """The most recent plan that still has work to do — used by --resume."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM plans WHERE status='running' ORDER BY id DESC LIMIT 1"
        ).fetchone()


def list_plans(limit: int = 50) -> list[sqlite3.Row]:
    """Most recent plans first — powers the web jobs panel."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM plans ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def plan_steps(plan_id: int) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM steps WHERE plan_id=? ORDER BY idx", (plan_id,)
        ).fetchall()


def log(kind: str, content: str = "", plan_id: int | None = None, step_id: int | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO logs (plan_id, step_id, kind, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (plan_id, step_id, kind, content, time.time()),
        )


# --- tasks / reminders ------------------------------------------------------

def add_task(text: str, due: str | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (text, due, created_at) VALUES (?, ?, ?)",
            (text, due, time.time()),
        )
        return cur.lastrowid


def list_tasks(include_done: bool = False) -> list[sqlite3.Row]:
    query = "SELECT * FROM tasks"
    if not include_done:
        query += " WHERE done=0"
    query += " ORDER BY id"
    with connect() as conn:
        return conn.execute(query).fetchall()


def complete_task(task_id: int) -> bool:
    """Mark a task done; returns False if no such task exists."""
    with connect() as conn:
        cur = conn.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
        return cur.rowcount > 0
