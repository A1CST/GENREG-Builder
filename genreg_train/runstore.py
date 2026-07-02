"""Persist training runs: config, per-generation metrics, and the final checkpoint.

Layout on disk (one folder per run, grouped by environment so the dashboard can
render a per-environment tech tree):

    runs/<env>/<run_id>/
        config.json     # the launch config + started-event metadata + status
        history.jsonl   # one line per generation (fitness + champion stats)
        summary.json    # final result (status, best, checkpoint filename)
        checkpoint.pkl  # the champion genome (engine checkpoint format)

run_id = "<YYYYmmdd-HHMMSS>-<env>-<confighash>".
"""
import datetime
import hashlib
import json
import os

from engine_api import save as save_genome, load as load_genome

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def _now():
    return datetime.datetime.now()


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def create_run(cfg, started):
    """Create a run folder from the launch config + the 'started' event."""
    ts = _now()
    env = started.get("environment") or cfg.get("environment") or "snake"
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]
    rid = f"{stamp}-{env}-{h}"
    d = os.path.join(RUNS_DIR, env, rid)
    os.makedirs(d, exist_ok=True)
    meta = {
        "id": rid, "environment": env, "created": ts.isoformat(timespec="seconds"),
        "config": cfg,
        "started": {k: started.get(k) for k in
                    ("n_in", "n_out", "constraints", "po", "population", "generations", "notes")},
        "status": "running",
    }
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(d, "history.jsonl"), "w").close()
    return {"id": rid, "dir": d, "env": env, "config": cfg}


def append_metric(run, ev):
    rec = {"gen": ev.get("gen"), "fitness": ev.get("fitness"),
           "best": {k: (ev.get("best") or {}).get(k) for k in ("score", "base", "H", "steps")}}
    try:
        with open(os.path.join(run["dir"], "history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def finalize(run, done, genome):
    d = run["dir"]
    ckpt_name = None
    if genome is not None:
        try:
            save_genome(genome, os.path.join(d, "checkpoint.pkl"))
            ckpt_name = "checkpoint.pkl"
        except Exception:
            ckpt_name = None
    summary = {
        "id": run["id"], "environment": run["env"],
        "status": done.get("reason", "finished"),
        "finished": _now().isoformat(timespec="seconds"),
        "gen": done.get("gen"), "best": done.get("best"),
        "checkpoint": ckpt_name,
    }
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    cfg = _read_json(os.path.join(d, "config.json")) or {}
    cfg["status"] = summary["status"]
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------- reads for the API
def _run_dir(rid):
    if not os.path.isdir(RUNS_DIR):
        return None
    for env in os.listdir(RUNS_DIR):
        d = os.path.join(RUNS_DIR, env, rid)
        if os.path.isdir(d):
            return d
    return None


def list_runs():
    runs = []
    if not os.path.isdir(RUNS_DIR):
        return runs
    for env in sorted(os.listdir(RUNS_DIR)):
        envd = os.path.join(RUNS_DIR, env)
        if not os.path.isdir(envd):
            continue
        for rid in os.listdir(envd):
            cfg = _read_json(os.path.join(envd, rid, "config.json"))
            if not cfg:
                continue
            summ = _read_json(os.path.join(envd, rid, "summary.json")) or {}
            runs.append({
                "id": rid, "environment": env, "created": cfg.get("created"),
                "status": summ.get("status", cfg.get("status", "running")),
                "population": cfg.get("config", {}).get("population"),
                "generations": cfg.get("config", {}).get("generations"),
                "device": cfg.get("config", {}).get("device", "cpu"),
                "constraints": cfg.get("config", {}).get("constraints", []),
                "best": summ.get("best"),
                "has_checkpoint": bool(summ.get("checkpoint")),
            })
    runs.sort(key=lambda r: r.get("created") or "", reverse=True)
    return runs


def get_run(rid):
    d = _run_dir(rid)
    if not d:
        return None
    cfg = _read_json(os.path.join(d, "config.json"))
    summ = _read_json(os.path.join(d, "summary.json"))
    history = []
    try:
        with open(os.path.join(d, "history.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return {"id": rid, "config": cfg, "summary": summ, "history": history}


def get_traces(rid):
    """Saved routing traces for a tree run (traces.jsonl), newest first."""
    d = _run_dir(rid)
    if not d:
        return []
    out = []
    try:
        with open(os.path.join(d, "traces.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        return []
    out.reverse()
    return out


def load_checkpoint(rid):
    d = _run_dir(rid)
    if not d:
        return None
    path = os.path.join(d, "checkpoint.pkl")
    if not os.path.exists(path):
        return None
    return load_genome(path)


def infer(rid):
    """Load a run's checkpoint and play one fresh (unconstrained) episode.

    Returns {env, frames, stats, base} for the dashboard's inference viewer.
    """
    import numpy as np
    from envs import SnakeEnv, Game2048Env
    from agent import rollout

    meta = get_run(rid) or {}
    cfg_meta = meta.get("config") or {}
    cfg = cfg_meta.get("config") or {}
    env_name = cfg_meta.get("environment", "snake")

    if env_name == "tree":
        # tree-of-models run: "replay" = generate a text sample from the model
        from genreg_train import tree_service
        d = _run_dir(rid)
        model_path = os.path.join(d, "model.npz") if d else None
        if not model_path or not os.path.exists(model_path):
            return None
        return tree_service.infer_run(model_path, cfg)

    g = load_checkpoint(rid)
    if g is None:
        return None
    rng = np.random.default_rng()
    if env_name == "2048":
        env = Game2048Env(rng=rng)
    else:
        sn = cfg.get("snake") or {}
        env = SnakeEnv(int(sn.get("w", 20)), int(sn.get("h", 15)), rng)
    r = rollout(g, env, None, rng, record=True)
    return {"env": env_name, "frames": r.frames, "stats": r.stats, "base": round(r.base_score, 3)}
