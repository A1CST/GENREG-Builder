"""Post a notice to the GENREG Agent panel (the floating Agent panel on every page).

For AI assistants and humans working in the workspace terminals:

    python agent_notify.py "title" ["body"] [--kind info|test|run|alert]
                           [--source name] [--run-id ID]

Pass "-" as body to read it from stdin (handy for piping test output):

    pytest -q 2>&1 | python agent_notify.py "test results" - --kind test

Writes straight to agent_store/notices.jsonl via agent_board — no server
needed; the panel polls the file (through Flask) and badges within ~8 s.
"""
import argparse
import sys

import genreg_paths                               # noqa: F401
import agent_board


def main():
    ap = argparse.ArgumentParser(
        description="Post a notice to the GENREG Agent panel.")
    ap.add_argument("title", help="short headline (shown in the panel list)")
    ap.add_argument("body", nargs="?", default="",
                    help="details; '-' reads from stdin")
    ap.add_argument("--kind", default="info", choices=list(agent_board.KINDS),
                    help="info (default) | test | run | alert")
    ap.add_argument("--source", default="cli",
                    help="who is posting (e.g. claude, gpt, human)")
    ap.add_argument("--run-id", default=None, help="related run id, if any")
    a = ap.parse_args()
    body = sys.stdin.read() if a.body == "-" else a.body
    n = agent_board.post(a.title, body, kind=a.kind, source=a.source,
                         run_id=a.run_id)
    print(f"posted notice #{n['id']} [{n['kind']}] {n['title']}")


if __name__ == "__main__":
    main()
