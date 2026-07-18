"""Agent board — the shared notice feed behind the floating "Agent" panel.

AI assistants (and automated jobs) post updates, test results, and run-finished
alarms here; every project page shows them in a draggable/minimizable panel
with an unread badge. Notices are plain JSON lines in agent_store/notices.jsonl
so posting works with or without the Flask server running.

Ways to post:
    python:   import agent_board; agent_board.post("title", "body", kind="test")
    CLI:      python agent_notify.py "title" "body" --kind test
    HTTP:     POST /api/agent/notices  {"title": ..., "body": ..., "kind": ...}
Training jobs (engine /train, Tree LM, encoder, sweeps, DiffEvo) post
automatically when they finish.
"""
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import datetime
import json
import os
import threading

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_store")
PATH = os.path.join(_DIR, "notices.jsonl")

KINDS = ("info", "test", "run", "alert")
MAX_KEEP = 1000          # trim the file when it grows past this many notices

_lock = threading.Lock()
_last_id = None          # lazily initialized from the file, then monotonic


def _load():
    out = []
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return out


def post(title, body="", kind="info", source="agent", run_id=None):
    """Append one notice. Returns the stored notice dict. Never raises on
    bad field types — everything is coerced and length-capped."""
    global _last_id
    kind = str(kind).strip().lower()
    if kind not in KINDS:
        kind = "info"
    with _lock:
        # Re-scan the file's max id every post: several PROCESSES write this
        # file (Flask, CLI posts from terminals), so a cached in-memory counter
        # hands out duplicate ids. The in-memory value still guards same-process
        # monotonicity if a concurrent writer lands between scan and write.
        items = _load()
        file_max = max((int(n.get("id", 0)) for n in items), default=0)
        _last_id = max(file_max, _last_id or 0) + 1
        n = {
            "id": _last_id,
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "source": str(source).strip()[:40] or "agent",
            "title": str(title).strip()[:200],
            "body": str(body).strip()[:4000],
        }
        if run_id:
            n["run_id"] = str(run_id)[:80]
        os.makedirs(_DIR, exist_ok=True)
        if len(items) >= MAX_KEEP:
            keep = items[-(MAX_KEEP - 1):] + [n]
            with open(PATH, "w", encoding="utf-8") as f:
                for it in keep:
                    f.write(json.dumps(it) + "\n")
        else:
            with open(PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(n) + "\n")
    return n


def list_notices(since=0, limit=100):
    """Notices with id > since, newest first, capped at `limit`."""
    try:
        since, limit = int(since), int(limit)
    except (TypeError, ValueError):
        since, limit = 0, 100
    items = [n for n in _load() if int(n.get("id", 0)) > since]
    items.sort(key=lambda n: int(n.get("id", 0)), reverse=True)
    return items[:max(1, min(limit, 500))]


def post_run_event(program, ev):
    """The end-of-run alarm: turn a trainer's terminal event ('done' /
    'sweep_done' / 'stopped') into a notice. Called from the job hubs and the
    /train socket — best-effort, callers guard with try/except."""
    t = ev.get("type", "done")
    if t == "error":
        return post(f"{program} run FAILED", str(ev.get("message", ""))[:1000],
                    kind="alert", source=program, run_id=ev.get("run_id"))
    reason = ev.get("reason") or ("stopped" if t == "stopped" else "finished")
    title = f"{program} run {reason}"
    keys = ("run_id", "seconds", "gen", "accuracy", "bigram_accuracy",
            "trained_nodes", "best", "l1_by_level", "improvement", "saved")
    bits = []
    for k in keys:
        v = ev.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            v = round(v, 4)
        s = json.dumps(v, default=str)
        bits.append(f"{k}: {s[:300]}")
    if t == "sweep_done":
        results = ev.get("results") or []
        title = f"{program} sweep {reason} — {len(results)} candidates"
        if results:
            best = results[0]
            bits.append(f"best: {json.dumps(best, default=str)[:300]}")
    return post(title, "\n".join(bits), kind="run", source=program,
                run_id=ev.get("run_id"))
