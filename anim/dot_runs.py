"""dot_runs.py — run recording + end-of-run alert for the attention line
(dot_track / dot_shape), per AGENTS.md rules 3 & 4.

Writes the standard five-file run record into runs/<env>/<run-id>/ so the run
shows up on the /runs page next to every other project, and posts an Agent-panel
notice when the run ends. Both are best-effort (never break training).
"""
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import datetime
import hashlib
import json
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def record(env, cfg, hist, stats, label, tags=None, log_lines=None, notify=True):
    """env: run environment / project bucket (e.g. 'animation'). cfg: config dict.
    hist: list of per-round dicts (round/fitness/added/n). stats: final metrics.
    Returns the run id (or None on failure)."""
    rid = None
    try:
        ts = datetime.datetime.now()
        h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]
        rid = f"{ts.strftime('%Y%m%d-%H%M%S')}-{env}-{h}"
        base = os.path.join(_HERE, "runs", env, rid)
        os.makedirs(base, exist_ok=True)
        created = ts.isoformat(timespec="seconds")
        with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "environment": env, "created": created,
                       "config": cfg, "status": "finished"}, f, indent=2)
        with open(os.path.join(base, "history.jsonl"), "w", encoding="utf-8") as f:
            for r in (hist or []):
                f.write(json.dumps({"gen": r.get("round"), "fitness": r.get("fitness"),
                                    "added": r.get("added"), "n": r.get("n")}) + "\n")
        with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "environment": env, "status": "finished",
                       "finished": created, "best": stats, "checkpoint": None}, f, indent=2)
        with open(os.path.join(base, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"label": label, "favorite": False, "group": "attention",
                       "tags": tags or []}, f, indent=2)
        with open(os.path.join(base, "report.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "kind": env, "created": created, "params": cfg,
                       "stats": stats, "log": [str(x)[:300] for x in (log_lines or [])][-200:]},
                      f, indent=2)
    except Exception as exc:
        print(f"[dot-runs] run record failed (non-fatal): {exc}", flush=True)
    if notify:
        try:
            import agent_board
            agent_board.post(f"Run finished: {label}",
                             json.dumps(stats, default=str),
                             kind="run", source="claude", run_id=rid)
        except Exception as exc:
            print(f"[dot-runs] notice failed (non-fatal): {exc}", flush=True)
    return rid
