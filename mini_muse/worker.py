"""The background worker: run a plan's steps one at a time, with retries.

It pulls the next pending step from SQLite, runs it through the agent,
and writes the result (or error) straight back to the database before moving on.
Because state is written as each step finishes, killing the process and running
`--resume` picks up exactly where it left off — no work is repeated or lost.
"""

import time

from groq import RateLimitError

from . import storage
from .agent import DEFAULT_MODEL, run_agent

MAX_ATTEMPTS = 3            # how many times to retry a failing step before giving up
RATE_LIMIT_WAIT = 5        # seconds to wait out a Groq free-tier 429, then retry


def run_plan(plan_id: int, model: str = DEFAULT_MODEL, verbose: bool = False) -> None:
    """Process every pending step of a plan until all are done or one fails out."""
    storage.reset_running_steps(plan_id)  # recover any step interrupted by a previous kill

    while True:
        step = storage.next_pending_step(plan_id)
        if step is None:
            break  # nothing left to do

        storage.mark_step(step["id"], "running")
        print(f"\n== step {step['idx'] + 1}: {step['description']}")

        try:
            answer = run_agent(step["description"], model=model, verbose=verbose).answer
        except RateLimitError:
            # Free-tier throughput limit — wait and retry the SAME step without
            # spending one of its attempts (it's not the step's fault).
            print(f"   rate limited; waiting {RATE_LIMIT_WAIT}s and retrying")
            storage.mark_step(step["id"], "pending")
            time.sleep(RATE_LIMIT_WAIT)
            continue
        except Exception as e:  # the step genuinely failed
            attempts = storage.bump_attempt(step["id"])
            if attempts >= MAX_ATTEMPTS:
                storage.mark_step(step["id"], "failed", error=str(e))
                storage.log("step_failed", str(e), plan_id, step["id"])
                print(f"   failed after {attempts} attempts: {e}")
                storage.set_plan_status(plan_id, "failed")
                return
            storage.mark_step(step["id"], "pending", error=str(e))
            storage.log("retry", str(e), plan_id, step["id"])
            print(f"   error (attempt {attempts}/{MAX_ATTEMPTS}), will retry: {e}")
            continue

        storage.mark_step(step["id"], "done", result=answer)
        storage.log("step_done", answer, plan_id, step["id"])
        print(f"   done: {answer[:200]}")

    storage.set_plan_status(plan_id, "done")
    print("\n== plan complete")
