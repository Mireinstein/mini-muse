"""Command-line entry point: `python -m mini_muse "your task"`."""

import argparse
import os
import sys

from dotenv import load_dotenv

from .agent import DEFAULT_MODEL, run_agent


def main() -> None:
    load_dotenv()  # pull ANTHROPIC_API_KEY / TAVILY_API_KEY from a local .env

    parser = argparse.ArgumentParser(prog="mini_muse", description="A from-scratch tool-calling agent.")
    parser.add_argument("task", help="What you want the agent to do.")
    parser.add_argument("--max-steps", type=int, default=10, help="Max loop iterations before giving up.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model to call.")
    parser.add_argument("--verbose", action="store_true", help="Print each tool call and its result.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to your environment or a .env file.")

    run_agent(args.task, model=args.model, max_steps=args.max_steps, verbose=args.verbose)


if __name__ == "__main__":
    main()
