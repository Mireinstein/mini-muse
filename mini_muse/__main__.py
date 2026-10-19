"""Command-line entry point.

Three ways to run:
  python -m mini_muse "a question"        one-shot: answer now
  python -m mini_muse --goal "a goal"     plan the goal, store it, run it
  python -m mini_muse --resume            continue the latest unfinished plan
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from . import storage
from .agent import DEFAULT_MODEL, run_agent
from .planner import make_plan
from .worker import run_plan


def main() -> None:
    load_dotenv()  # pull GROQ_API_KEY / TAVILY_API_KEY from a local .env

    parser = argparse.ArgumentParser(prog="mini_muse", description="A from-scratch tool-calling agent.")
    parser.add_argument("task", nargs="?", help="A one-shot question for the agent to answer now.")
    parser.add_argument("--goal", help="A goal to plan into steps, store, and run in the background.")
    parser.add_argument("--resume", action="store_true", help="Continue the latest unfinished plan.")
    parser.add_argument("--max-steps", type=int, default=10, help="Max loop iterations per agent task.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model to call.")
    parser.add_argument("--verbose", action="store_true", help="Print each tool call and its result.")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Add it to your environment or a .env file.")

    if args.resume:
        storage.init_db()
        plan = storage.latest_unfinished_plan()
        if plan is None:
            sys.exit("No unfinished plan to resume.")
        print(f"resuming plan {plan['id']}: {plan['goal']}")
        run_plan(plan["id"], model=args.model, verbose=args.verbose)
        return

    if args.goal:
        storage.init_db()
        steps = make_plan(args.goal, model=args.model)
        plan_id = storage.create_plan(args.goal)
        storage.add_steps(plan_id, steps)
        print(f"plan {plan_id} ({len(steps)} steps):")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
        run_plan(plan_id, model=args.model, verbose=args.verbose)
        return

    if args.task:
        answer, usage = run_agent(args.task, model=args.model, max_steps=args.max_steps, verbose=args.verbose)
        print(answer)
        print("\n" + usage.summary())
        return

    parser.error("give a task, --goal, or --resume")


if __name__ == "__main__":
    main()
