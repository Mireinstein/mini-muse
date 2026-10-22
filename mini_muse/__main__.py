"""Command-line entry point.

Ways to run:
  python -m mini_muse --ingest           sync site content into ./docs and index it
  python -m mini_muse --ingest FOLDER    index a specific folder instead
  python -m mini_muse --serve            start the web app (the Q&A chat)
"""

import argparse
import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()  # pull GROQ_API_KEY from a local .env

    parser = argparse.ArgumentParser(prog="mini_muse", description="Ask-Admire website Q&A bot.")
    parser.add_argument("--ingest", nargs="?", const="docs", metavar="FOLDER",
                        help="Sync site content into ./docs (or FOLDER) and index it.")
    parser.add_argument("--no-sync", action="store_true",
                        help="With --ingest, skip fetching the resume/GitHub; index what's already there.")
    parser.add_argument("--serve", action="store_true", help="Start the web app.")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve (default 8000).")
    args = parser.parse_args()

    # Ingestion is fully local (no model calls), so it doesn't need a Groq key.
    if args.ingest:
        from .rag import ingest
        if not args.no_sync and args.ingest == "docs":
            from .content import sync
            print("syncing content…")
            sync(args.ingest)
        count = ingest(args.ingest)
        print(f"indexed {count} chunks from {args.ingest}")
        return

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Add it to your environment or a .env file.")

    if args.serve:
        import uvicorn
        from .web.server import app
        # A host sets $PORT (and we bind 0.0.0.0 there); locally it stays on
        # 127.0.0.1:8000 unless HOST/PORT say otherwise.
        host = os.environ.get("HOST", "127.0.0.1")
        port = int(os.environ.get("PORT", args.port))
        print(f"Ask Admire running on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
        return

    parser.error("give --ingest or --serve")


if __name__ == "__main__":
    main()
