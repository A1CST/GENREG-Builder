"""GENREG — Flask web interface for building the project.

Thin relay between the browser and the terminal daemon. The daemon owns the
real ConPTY PowerShell terminals, so they survive Flask restarts:

    Browser  <-- WebSocket -->  Flask (app.py)  <-- TCP -->  Daemon  <-- PTY -->  PowerShell(s)

A single WebSocket carries everything both ways (keystrokes, resize, control
ops one way; raw terminal output the other), which is what makes interactive
programs like the `claude` TUI work in the browser.

Run:  python app.py    (the daemon is started automatically if not running)
Then open http://127.0.0.1:5000
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time

from flask import (Flask, render_template, jsonify, request, send_file,
                   send_from_directory)
from flask_sock import Sock
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "lm"))   # lm-package path
import genreg_paths                               # noqa: F401

app = Flask(__name__)
app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
# Pick up template + static edits without needing a manual cache-buster.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
sock = Sock(app)

# ---------------------------------------------------------------------------
# THE PROJECT REGISTRY — one source of truth for BOTH the top navigation
# (grouped dropdowns) AND the terminal-tag picker (static/app.js). Add a
# project here once and it appears in the nav group you assign it to and in
# the "tag this terminal" menu automatically — no second list to keep in sync.
#
# Each entry: (key, label, href, color). `key` must match the `nav_active`
# value a page sets and the tag stored per terminal. Groups are ordered; the
# order inside a group is the order shown in its dropdown.
# ---------------------------------------------------------------------------
PROJECT_GROUPS = [
    ("Vision", [
        ("mnist",     "MNIST",     "/mnist",        "#e3b341"),
        ("cifar",     "CIFAR",     "/cifar",        "#ff7b72"),
        ("resnet",    "ResNet",    "/resnet",       "#d29922"),
        ("xray",      "X-Ray",     "/xray",         "#79c0ff"),
        ("radial",    "Radial",    "/radial",       "#f778ba"),
        ("rdemo",     "Demo",      "/radial/demo",  "#f0a0c8"),
        ("vision_demo", "Vision Demo", "/vision_demo", "#5fd0c0"),
        ("ocr",       "OCR",       "/ocr",          "#4fd0e0"),
    ]),
    ("Sequence", [
        ("lm",        "LM",        "/lm",           "#56d364"),
        ("lm_demo",   "LM Demo",   "/lm_demo",      "#3fdba0"),
        ("tsdb",      "TSDB",      "/tsdb",         "#39c5cf"),
        ("replicate", "Replicate", "/replicate",    "#c39bf0"),
    ]),
    ("Evolve", [
        ("diff",      "DiffEvo",   "/diff",         "#d2a8ff"),
        ("animation", "Animation", "/animation",    "#ff9e64"),
        ("humanoid",  "Humanoid",  "/humanoid",     "#ffa657"),
        ("pure",      "PURE",      "/pure",         "#7ee787"),
    ]),
    ("Media & Net", [
        ("images",    "Images",    "/images",       "#a5a5f5"),
        ("video",     "Video",     "/video",        "#f0883e"),
        ("i2",        "I2",        "/i2",           "#2ea043"),
        ("pia",       "PIA",       "/pia",          "#db61a2"),
    ]),
    ("Workspace", [
        ("build",     "Build",     "/",             "#4ea1ff"),
        ("plan",      "Plan",      "/plan",         "#8ab4f8"),
        ("history",   "History",   "/history",      "#c8a2ff"),
        ("runs",      "Runs",      "/runs",         "#58a6ff"),
        ("progress",  "Progress",  "/progress",     "#7ee787"),
        ("docs",      "Docs",      "/docs",         "#8b95a1"),
    ]),
]


@app.context_processor
def _inject_project_nav():
    """Expose the project registry to every template: `nav_groups` drives the
    grouped nav dropdowns, `projects_json` is emitted as window.GENREG_PROJECTS
    so static/app.js builds the terminal-tag picker from the SAME list."""
    import json as _json
    groups = []
    flat = []
    for gname, items in PROJECT_GROUPS:
        gi = []
        for key, label, href, color in items:
            entry = {"key": key, "label": label, "href": href, "color": color}
            gi.append(entry)
            flat.append({"key": key, "label": label, "color": color})
        groups.append({"label": gname, "memberkeys": [i["key"] for i in gi], "members": gi})
    return {"nav_groups": groups, "projects_json": _json.dumps(flat)}

# Neuroevolution training (Snake / 2048). Imported lazily-guarded so a missing
# numpy/engine never stops the terminal interface from serving.
try:
    from genreg_train import create_trainer, parse_config, runstore
    TRAIN_OK, TRAIN_ERR = True, None
except Exception as _e:                       # pragma: no cover
    TRAIN_OK, TRAIN_ERR = False, str(_e)

# MNIST-Pipe — the WordPipe recipe applied to images (stats layer built from
# data, semantic layer evolved, output mixer evolved). Proof outside language.
try:
    from genreg_train import mnist_service
    MN_OK, MN_ERR = True, None
except Exception as _e:                       # pragma: no cover
    MN_OK, MN_ERR = False, str(_e)

# CIFAR-Pipe — the MNIST-Pipe program, verbatim, on CIFAR-10 (staged).
try:
    from genreg_train import cifar_service
    CF_OK, CF_ERR = True, None
except Exception as _e:                       # pragma: no cover
    CF_OK, CF_ERR = False, str(_e)

# I2 latent-content node (separate program; shares this server).
try:
    import i2_service
    I2_OK, I2_ERR = True, None
except Exception as _e:                       # pragma: no cover
    I2_OK, I2_ERR = False, str(_e)

# LM — the language pipeline rebuild, starting from genome #1 (intent
# recognition). Fresh start after /evolang was archived (archive/evolang_v1).
try:
    from genreg_train import lm_service
    LM_OK, LM_ERR = True, None
except Exception as _e:                       # pragma: no cover
    LM_OK, LM_ERR = False, str(_e)

# DiffEvo — denoising diffusion by neuroevolution (separate program).
try:
    from genreg_train import diffuse_service
    DIFF_OK, DIFF_ERR = True, None
except Exception as _e:                       # pragma: no cover
    DIFF_OK, DIFF_ERR = False, str(_e)

# Animation Evo — mutation-only shape classifier on the animation dataset.
try:
    from genreg_train import animation_evo
    ANIM_OK, ANIM_ERR = True, None
except Exception as _e:                       # pragma: no cover
    ANIM_OK, ANIM_ERR = False, str(_e)

# Images — pretrained Stable Diffusion text-to-image (separate program).
try:
    from genreg_train import sd_service
    SD_OK, SD_ERR = True, None
except Exception as _e:                       # pragma: no cover
    SD_OK, SD_ERR = False, str(_e)

# Images — reverse direction: image/video -> prompt (BLIP + CLIP).
try:
    from genreg_train import reverse_service
    REV_OK, REV_ERR = True, None
except Exception as _e:                       # pragma: no cover
    REV_OK, REV_ERR = False, str(_e)

# Video editor — library + ffmpeg jobs (cut / stitch / convert / export).
try:
    import video_service
    VID_OK, VID_ERR = video_service.available(), None
    if not VID_OK:
        VID_ERR = "ffmpeg not found (tools/ffmpeg-*/bin, PATH, imageio-ffmpeg)"
except Exception as _e:                       # pragma: no cover
    VID_OK, VID_ERR = False, str(_e)

# Animation platform — SVG rigs + scene timelines rendered onto the library.
try:
    import anim_service
    ANIMP_OK, ANIMP_ERR = True, None
except Exception as _e:                       # pragma: no cover
    ANIMP_OK, ANIMP_ERR = False, str(_e)

# Agent board — shared notice feed for the floating Agent panel (all pages).
try:
    import agent_board
    BOARD_OK = True
except Exception:                             # pragma: no cover
    agent_board, BOARD_OK = None, False

# PIA — Personal AI: local Ollama RAG assistant over the project docs. The
# Ollama server is only ever started from the PIA page (never automatically).
try:
    import pia_service
    PIA_OK, PIA_ERR = True, None
except Exception as _e:                        # pragma: no cover
    pia_service, PIA_OK, PIA_ERR = None, False, str(_e)

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 5001
DAEMON_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cli", "terminal_daemon.py")


# --------------------------------------------------------------------------
# Daemon management
# --------------------------------------------------------------------------
def daemon_up():
    try:
        with socket.create_connection((DAEMON_HOST, DAEMON_PORT), timeout=0.4):
            return True
    except OSError:
        return False


def ensure_daemon():
    """Start the terminal daemon as a detached background process if needed."""
    if daemon_up():
        return True
    flags = 0
    if os.name == "nt":
        flags = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(
            [sys.executable, DAEMON_SCRIPT],
            creationflags=flags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    for _ in range(50):
        if daemon_up():
            return True
        time.sleep(0.1)
    return False


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    # no-store so the browser never serves a stale HTML shell against fresh JS.
    resp = app.make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/plan")
def plan_page():
    """Personal day-plan tracker — Sunday, July 12, 2026 full-day schedule.
    Self-contained page: every block logs DONE / PARTIAL / SKIPPED plus notes,
    an unscheduled-break log, and an end-of-day scorecard. All state persists in
    the browser's localStorage (no backend), so it survives reloads."""
    resp = app.make_response(render_template("plan.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/changelog")
def changelog():
    path = os.path.join(os.path.dirname(__file__), "CHANGELOG.md")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return app.response_class(fh.read(), mimetype="text/plain")
    except OSError:
        return app.response_class("Changelog not found.", mimetype="text/plain", status=404)


DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documentation")


@app.route("/docs")
def docs_page():
    resp = app.make_response(render_template("docs.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/docs")
def api_docs():
    """List every file in documentation/ (recursive) with type metadata."""
    out = []
    for root, _dirs, files in os.walk(DOCS_DIR):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, DOCS_DIR).replace(os.sep, "/")
            try:
                st = os.stat(full)
            except OSError:
                continue
            out.append({
                "path": rel,
                "name": fn,
                "ext": os.path.splitext(fn)[1].lower().lstrip("."),
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    out.sort(key=lambda f: f["name"].lower())
    return jsonify(out)


@app.route("/api/docs/file/<path:relpath>")
def api_doc_file(relpath):
    """Serve one documentation file. send_from_directory blocks traversal."""
    ext = os.path.splitext(relpath)[1].lower()
    # Text-ish formats: force an inline text mimetype so the browser/fetch
    # gets utf-8 text instead of a download; PDFs keep application/pdf so
    # the <embed> viewer works; anything else downloads.
    text_exts = {".md", ".markdown", ".txt", ".json", ".log", ".csv"}
    if ext in text_exts:
        resp = send_from_directory(DOCS_DIR, relpath, mimetype="text/plain")
        resp.headers["Content-Type"] = "text/plain; charset=utf-8"
        return resp
    if ext == ".pdf":
        return send_from_directory(DOCS_DIR, relpath, mimetype="application/pdf")
    return send_from_directory(DOCS_DIR, relpath, as_attachment=True)


# --------------------------------------------------------------------------
# PIA — Personal AI (Ollama RAG over docs + changelogs)
# --------------------------------------------------------------------------
@app.route("/pia")
def pia_page():
    resp = app.make_response(render_template("pia.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/pia/status")
def api_pia_status():
    if not PIA_OK:
        return jsonify({"available": False, "error": PIA_ERR})
    st = pia_service.status()
    st["available"] = True
    return jsonify(st)


@app.route("/api/pia/start", methods=["POST"])
def api_pia_start():
    if not PIA_OK:
        return jsonify({"ok": False, "message": PIA_ERR or "unavailable"}), 503
    ok, msg = pia_service.start_server()
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/pia/stop", methods=["POST"])
def api_pia_stop():
    if not PIA_OK:
        return jsonify({"ok": False, "message": PIA_ERR or "unavailable"}), 503
    ok, msg = pia_service.stop_server()
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/pia/pull", methods=["POST"])
def api_pia_pull():
    if not PIA_OK:
        return jsonify({"ok": False, "message": PIA_ERR or "unavailable"}), 503
    if not pia_service.is_running():
        return jsonify({"ok": False,
                        "message": "Start Ollama first."}), 400
    pia_service.pull_models_async()
    return jsonify({"ok": True, "message": "Pulling models..."})


@app.route("/api/pia/reindex", methods=["POST"])
def api_pia_reindex():
    if not PIA_OK:
        return jsonify({"ok": False, "message": PIA_ERR or "unavailable"}), 503
    if not pia_service.is_running():
        return jsonify({"ok": False,
                        "message": "Start Ollama first."}), 400
    pia_service.build_index_async()
    return jsonify({"ok": True, "message": "Rebuilding index..."})


@app.route("/api/pia/chat", methods=["POST"])
def api_pia_chat():
    if not PIA_OK:
        return jsonify({"error": PIA_ERR or "PIA unavailable"}), 503
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    history = data.get("history") or []
    result = pia_service.chat(message, history)
    code = 200 if "answer" in result else 400
    return jsonify(result), code


@app.route("/runs")
def runs_page():
    resp = app.make_response(render_template("runs.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/runs")
def api_runs():
    if not TRAIN_OK:
        return jsonify([])
    return jsonify(runstore.list_runs())


@app.route("/api/runs/<rid>")
def api_run(rid):
    if not TRAIN_OK:
        return jsonify({"error": "training unavailable"}), 503
    r = runstore.get_run(rid)
    return (jsonify(r), 200) if r else (jsonify({"error": "not found"}), 404)


# page scope -> run env dirs (None = engine envs) for the run-config panel
RUN_SCOPES = {"build": None, "tree": ("tree", "encoder"), "diff": ("diffevo",)}


@app.route("/api/active-run")
def api_active_run():
    """Newest run for a page's project scope — feeds the run-config panel."""
    if not TRAIN_OK:
        return jsonify(None)
    scope = request.args.get("scope", "")
    if scope not in RUN_SCOPES:
        return jsonify(None)
    return jsonify(runstore.latest_run(RUN_SCOPES[scope]))


@app.route("/api/run-history")
def api_run_history():
    """Newest runs across all projects — the config panel's history list."""
    if not TRAIN_OK:
        return jsonify([])
    return jsonify(runstore.recent_runs(request.args.get("limit", 10)))


@app.route("/api/agent/notices")
def api_agent_notices():
    """Notice feed for the Agent panel (newest first)."""
    if not BOARD_OK:
        return jsonify([])
    return jsonify(agent_board.list_notices(
        since=request.args.get("since", 0), limit=request.args.get("limit", 100)))


@app.route("/api/agent/notices", methods=["POST"])
def api_agent_post():
    """Post a notice (AIs/tools; also reachable via agent_notify.py offline)."""
    if not BOARD_OK:
        return jsonify({"error": "agent board unavailable"}), 503
    data = request.get_json(silent=True) or {}
    if not str(data.get("title", "")).strip():
        return jsonify({"error": "title required"}), 400
    return jsonify(agent_board.post(
        data.get("title"), data.get("body", ""), kind=data.get("kind", "info"),
        source=data.get("source", "http"), run_id=data.get("run_id")))


@app.route("/api/runs/<rid>/meta", methods=["POST"])
def api_run_meta(rid):
    """Update a run's dashboard metadata (label / favorite / group / tags)."""
    if not TRAIN_OK:
        return jsonify({"error": "training unavailable"}), 503
    patch = request.get_json(silent=True) or {}
    meta = runstore.set_meta(rid, patch)
    return (jsonify(meta), 200) if meta is not None else (jsonify({"error": "not found"}), 404)


@app.route("/api/runs/<rid>/traces")
def api_traces(rid):
    if not TRAIN_OK:
        return jsonify([])
    return jsonify(runstore.get_traces(rid))


@app.route("/api/runs/<rid>/embedding")
def api_embedding(rid):
    if not TRAIN_OK:
        return jsonify({"error": "training unavailable"}), 503
    try:
        r = runstore.embedding(rid)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return (jsonify(r), 200) if r else (jsonify({"error": "no encoder saved for this run"}), 404)


@app.route("/api/runs/<rid>/words")
def api_word_embedding(rid):
    """Word-level context-vector cloud (tree/encoder runs)."""
    if not TRAIN_OK:
        return jsonify({"error": "training unavailable"}), 503
    try:
        r = runstore.word_embedding(rid)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return (jsonify(r), 200) if r else (jsonify({"error": "no encoder saved for this run"}), 404)


@app.route("/api/runs/<rid>/replay")
def api_replay(rid):
    if not TRAIN_OK:
        return jsonify({"error": "training unavailable"}), 503
    try:
        r = runstore.infer(rid)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return (jsonify(r), 200) if r else (jsonify({"error": "no checkpoint"}), 404)


@app.route("/i2")
def i2_page():
    """I2 — separate program sharing this server: canvas + the same terminals."""
    resp = app.make_response(render_template("i2.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/i2/genome")
def api_i2_genome():
    """Current genome (decoder weights included) — the browser's auto-update."""
    if not I2_OK:
        return jsonify({"error": f"i2 unavailable: {I2_ERR}"}), 503
    return jsonify(i2_service.genome())


@app.route("/api/i2/pages")
def api_i2_pages():
    if not I2_OK:
        return jsonify({"error": f"i2 unavailable: {I2_ERR}"}), 503
    return jsonify(i2_service.list_pages())


@app.route("/api/i2/page/<name>")
def api_i2_page(name):
    """One latent document. The server never sends readable page text."""
    if not I2_OK:
        return jsonify({"error": f"i2 unavailable: {I2_ERR}"}), 503
    doc = i2_service.get_page(name)
    return (jsonify(doc), 200) if doc else (jsonify({"error": "no such page"}), 404)


@app.route("/api/i2/publish", methods=["POST"])
def api_i2_publish():
    if not I2_OK:
        return jsonify({"error": f"i2 unavailable: {I2_ERR}"}), 503
    data = request.get_json(silent=True) or {}
    try:
        meta = i2_service.publish(
            str(data.get("name", "")), str(data.get("title", "")),
            str(data.get("content", "")), origin=str(data.get("origin", "local")))
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(meta)


@app.route("/lm")
def lm_page():
    """LM — the language pipeline rebuild. Genome #1: intent recognition."""
    resp = app.make_response(render_template("lm.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/lm/status")
def api_lm_status():
    """Intent genome load status (lazy-loads the trained artifact on first hit)."""
    if not LM_OK:
        return jsonify({"ready": False, "err": LM_ERR or "lm unavailable"})
    lm_service.SERVICE.ensure()
    return jsonify(lm_service.SERVICE.status())


@app.route("/api/lm/recognize")
def api_lm_recognize():
    """Given a word run (no trailing punctuation), rank which mark's intent
    fits next."""
    if not LM_OK:
        return jsonify({"err": LM_ERR or "lm unavailable"})
    lm_service.SERVICE.ensure()
    text = request.args.get("text", "")
    return jsonify(lm_service.SERVICE.recognize(text))


@app.route("/api/lm/recognize_opener")
def api_lm_recognize_opener():
    """Given ONLY a sentence's first word, rank which end-mark intent
    (statement/question/exclaim) the sentence is headed for."""
    if not LM_OK:
        return jsonify({"err": LM_ERR or "lm unavailable"})
    lm_service.SERVICE.ensure()
    word = request.args.get("word", "")
    return jsonify(lm_service.SERVICE.recognize_opener(word))


@app.route("/api/lm/generate")
def api_lm_generate():
    """Hangman-style generation from a single seed word: intent decided
    once (opener genomes), then a confidence-driven grow/fill tick loop
    (length_continue + fill_word). Variable length, no fixed target."""
    if not LM_OK:
        return jsonify({"err": LM_ERR or "lm unavailable"})
    lm_service.SERVICE.ensure()
    a = request.args
    word = a.get("word", "")
    try:
        seed = int(a.get("seed", 0))
    except (TypeError, ValueError):
        seed = 0
    try:
        temperature = max(0.05, min(2.0, float(a.get("temperature", 0.7))))
    except (TypeError, ValueError):
        temperature = 0.7
    return jsonify(lm_service.SERVICE.generate(word, seed=seed, temperature=temperature))


@app.route("/mnist")
def mnist_page():
    """MNIST — the specialist-pipeline recipe applied to images."""
    resp = app.make_response(render_template("mnist.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/mnist/status")
def api_mnist_status():
    """Pipeline load status (lazy-loads genomes + statistics layer on first hit)."""
    if not MN_OK:
        return jsonify({"ready": False, "err": MN_ERR or "mnist unavailable"})
    mnist_service.SERVICE.ensure()
    return jsonify(mnist_service.SERVICE.status())


@app.route("/api/mnist/eval")
def api_mnist_eval():
    """Test accuracy + confusion matrix for the enabled layer subset."""
    if not MN_OK:
        return jsonify({"err": MN_ERR or "mnist unavailable"})
    a = request.args
    mnist_service.SERVICE.ensure()
    return jsonify(mnist_service.SERVICE.evaluate(
        use_mixer=a.get("mixer") == "1", use_pairs=a.get("pairs") == "1"))


@app.route("/api/mnist/sample")
def api_mnist_sample():
    """A grid of test digits with the pipeline's predictions."""
    if not MN_OK:
        return jsonify({"err": MN_ERR or "mnist unavailable"})
    a = request.args
    try:
        seed = int(a.get("seed", 0))
    except (TypeError, ValueError):
        seed = 0
    mnist_service.SERVICE.ensure()
    return jsonify(mnist_service.SERVICE.sample(
        seed=seed, use_mixer=a.get("mixer") == "1", use_pairs=a.get("pairs") == "1",
        only_errors=a.get("errors") == "1"))


@app.route("/api/mnist/reload", methods=["POST"])
def api_mnist_reload():
    """Reload champions from disk (after a training run finishes)."""
    if not MN_OK:
        return jsonify({"err": MN_ERR or "mnist unavailable"})
    mnist_service.SERVICE.reload()
    return jsonify({"ok": True})


@app.route("/api/mnist/radial")
def api_mnist_radial():
    """The radial seed-stack ladder (single / union / composed-across-seed) —
    the manufactured-rotation static-classification result. Reads the exported
    JSON written by mnist_radial.run(); returns {} until the first run lands."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "radial_data", "mnist_radial.json")
    if not os.path.exists(p):
        return jsonify({})
    with open(p) as f:
        return jsonify(json.load(f))


# ── TSDB ────────────────────────────────────────────────────────────
# A small Float64 block store (TSDB.js). The page runs an in-browser port
# of it and feeds it real MNIST pipeline metrics as Float64 row data.
@app.route("/tsdb")
def tsdb_page():
    """TSDB — the Float64 block store, driven with real MNIST metrics."""
    resp = app.make_response(render_template("tsdb.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


_TSDB_SETS = {
    "mnist": {
        "cache": "mnist_genomes.pkl",
        "classes": [str(i) for i in range(10)],
    },
    "cifar": {
        "cache": "cifar_genomes.pkl",
        "classes": ["plane", "car", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
    },
}


@app.route("/api/tsdb/mnist")            # back-compat alias
@app.route("/api/tsdb/data")
def api_tsdb_data():
    """Numeric pipeline series (MNIST or CIFAR), shaped as Float64 rows.

    Everything here is plain numbers pulled from the frozen champions
    (demo/<set>_genomes.pkl) — no model load, no test-set pass. The browser
    serializes each series into TSDB blocks and reads them straight back.
    Pick the set with ?set=mnist|cifar (default mnist)."""
    import pickle
    which = (request.args.get("set", "mnist") or "mnist").lower()
    spec = _TSDB_SETS.get(which)
    if spec is None:
        return jsonify({"err": f"unknown set '{which}'"}), 400
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "demo", spec["cache"])
    if not os.path.exists(cache):
        return jsonify({"err": f"demo/{spec['cache']} not found"}), 404
    try:
        with open(cache, "rb") as f:
            d = pickle.load(f)
    except Exception as exc:                       # pragma: no cover
        return jsonify({"err": f"{type(exc).__name__}: {exc}"}), 500

    res = d.get("results", {}) or {}
    # Layer progression: the store's `time` axis is the pipeline stage index.
    stages = [
        ("centroid", res.get("centroid_test")),
        ("argmax",   d.get("argmax_val_acc")),
        ("mixer",    d.get("mixer_val_acc")),
        ("joint",    d.get("joint_base_val_acc")),
        ("joint+bias", d.get("joint_val_acc")),
        ("+pairs",   res.get("joint_test")),
        ("full",     res.get("full_test")),
    ]
    layers, layer_labels, t = [], [], 0
    for lbl, a in stages:
        if a is None:
            continue
        layers.append({"t": t, "acc": float(a)})
        layer_labels.append(lbl)
        t += 1

    # One-vs-one specialists: 45 rows keyed by pair index.
    pv = d.get("pair_val_acc", {}) or {}
    pairs = []
    for i, (k, acc) in enumerate(sorted(pv.items())):
        try:
            a, b = k.split("v")
            pairs.append({"t": i, "a": int(a), "b": int(b), "acc": float(acc)})
        except (ValueError, TypeError):
            continue

    # Per-class detectors (one-vs-rest), when the champions carry them.
    classes = spec["classes"]
    dets = []
    dva = d.get("det_val_acc") or {}
    if isinstance(dva, dict):
        for c in sorted(dva.keys(), key=lambda x: int(x)):
            ci = int(c)
            name = classes[ci] if 0 <= ci < len(classes) else str(ci)
            dets.append({"t": ci, "cls": ci, "name": name, "acc": float(dva[c])})

    return jsonify({
        "ok": True,
        "set": which,
        "classes": classes,
        "joint_val_acc": d.get("joint_val_acc"),
        "joint_base_val_acc": d.get("joint_base_val_acc"),
        "feat_version": d.get("feat_version"),
        "layers": layers,
        "layer_labels": layer_labels,
        "pairs": pairs,
        "detectors": dets,
    })


@app.route("/api/tsdb/run", methods=["POST"])
def api_tsdb_run():
    """Record a TSDB op (round-trip verify / stress load) as a run under the
    `tsdb` environment, then post the Agent-panel notice carrying that run_id
    so the notice deep-links straight to the run on /runs. Falls back to a
    plain notice if the run store isn't available."""
    data = request.get_json(silent=True) or {}
    kind = data.get("kind", "info")
    title = str(data.get("title", "TSDB op")).strip() or "TSDB op"
    body = str(data.get("body", ""))
    ok = bool(data.get("ok", True))
    metrics = data.get("metrics") or {}
    op = data.get("op", "op")            # 'verify' | 'stress'

    run_id = None
    if TRAIN_OK:
        try:
            cfg = {"environment": "tsdb", "config": {"op": op, **metrics}}
            run = runstore.create_run(cfg, {"environment": "tsdb", "notes": body})
            runstore.finalize(
                run,
                {"reason": "finished" if ok else "failed",
                 "best": {"score": metrics.get("score")}},
                None)
            run_id = run["id"]
        except Exception:                # pragma: no cover — recording is best-effort
            run_id = None

    if BOARD_OK:
        try:
            agent_board.post(title, body, kind=kind, source="tsdb-page", run_id=run_id)
        except Exception:                # pragma: no cover
            pass
    return jsonify({"ok": True, "run_id": run_id})


@app.route("/cifar")
def cifar_page():
    """CIFAR — the MNIST specialist pipeline, verbatim, on CIFAR-10."""
    resp = app.make_response(render_template("cifar.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/cifar/status")
def api_cifar_status():
    if not CF_OK:
        return jsonify({"ready": False, "err": CF_ERR or "cifar unavailable"})
    cifar_service.SERVICE.ensure()
    return jsonify(cifar_service.SERVICE.status())


@app.route("/api/cifar/eval")
def api_cifar_eval():
    if not CF_OK:
        return jsonify({"err": CF_ERR or "cifar unavailable"})
    a = request.args
    cifar_service.SERVICE.ensure()
    return jsonify(cifar_service.SERVICE.evaluate(
        use_mixer=a.get("mixer") == "1", use_pairs=a.get("pairs") == "1"))


@app.route("/api/cifar/sample")
def api_cifar_sample():
    if not CF_OK:
        return jsonify({"err": CF_ERR or "cifar unavailable"})
    a = request.args
    try:
        seed = int(a.get("seed", 0))
    except (TypeError, ValueError):
        seed = 0
    cifar_service.SERVICE.ensure()
    return jsonify(cifar_service.SERVICE.sample(
        seed=seed, use_mixer=a.get("mixer") == "1", use_pairs=a.get("pairs") == "1",
        only_errors=a.get("errors") == "1"))


@app.route("/api/cifar/reload", methods=["POST"])
def api_cifar_reload():
    if not CF_OK:
        return jsonify({"err": CF_ERR or "cifar unavailable"})
    cifar_service.SERVICE.reload()
    return jsonify({"ok": True})


@app.route("/api/cifar/radial")
def api_cifar_radial():
    """The radial seed-stack ladder on CIFAR (single / stats-only / composed) +
    the genome ablation — where the evolved genomes finally earn residual. Reads
    the export written by cifar_radial/mnist_radial.run(); {} until it lands."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "radial_data", "cifar_radial.json")
    if not os.path.exists(p):
        return jsonify({})
    with open(p) as f:
        return jsonify(json.load(f))


@app.route("/diff")
def diff_page():
    """DiffEvo — denoising diffusion by neuroevolution (separate program)."""
    resp = app.make_response(render_template("diff.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/animation")
def animation_page():
    """Animation — new program page (blank scaffold for now)."""
    resp = app.make_response(render_template("animation.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/images")
def images_page():
    """Images — new program page (blank scaffold for now)."""
    resp = app.make_response(render_template("images.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/images/generate", methods=["POST"])
def api_images_generate():
    """Text -> image via the pretrained SD1.5 pipeline (lazy-loaded, singleton)."""
    if not SD_OK:
        return jsonify({"error": f"stable diffusion unavailable: {SD_ERR}"}), 503
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    try:
        result = sd_service.HUB.generate(
            prompt,
            negative_prompt=data.get("negative_prompt", ""),
            steps=data.get("steps", 25),
            guidance=data.get("guidance", 7.5),
            seed=data.get("seed"),
            width=data.get("width", 512),
            height=data.get("height", 512),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/images/reverse", methods=["POST"])
def api_images_reverse():
    """Image/video -> the prompt a diffusion model would've used, per frame.
    Uploads an image or video; results (frames + prompt .txt files) land in
    a structured job folder under runs/images/reverse/<job_id>/."""
    if not REV_OK:
        return jsonify({"error": f"reverse interrogation unavailable: {REV_ERR}"}), 503
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file uploaded"}), 400
    import tempfile
    ext = os.path.splitext(f.filename or "")[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".bin")
    tmp.close()
    try:
        f.save(tmp.name)
        try:
            max_side = int(request.form.get("max_side", 768))
            max_new_tokens = int(request.form.get("max_new_tokens", 40))
            top_k = int(request.form.get("top_k", 1))
            if ext in reverse_service.IMAGE_EXTS:
                manifest = reverse_service.HUB.process_image_file(
                    tmp.name, f.filename, max_side=max_side,
                    max_new_tokens=max_new_tokens, top_k=top_k)
            else:
                stride = int(request.form.get("stride", 1))
                max_frames = int(request.form.get("max_frames", 30))
                manifest = reverse_service.HUB.process_video_file(
                    tmp.name, f.filename, stride=stride, max_frames=max_frames, max_side=max_side,
                    max_new_tokens=max_new_tokens, top_k=top_k)
        except (TypeError, ValueError):
            return jsonify({"error": "bad parameters"}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify(manifest)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/api/images/file/<path:relpath>")
def api_images_file(relpath):
    """Serve a frame/prompt file from a reverse-interrogation job folder."""
    if not REV_OK:
        return jsonify({"error": f"reverse interrogation unavailable: {REV_ERR}"}), 503
    return send_from_directory(reverse_service.OUT_DIR, relpath)


@app.route("/pure")
def pure_page():
    """PURE — the baseline model: a plain GA with nothing added, the control
    that every GENREG bell and whistle gets measured against. Blank scaffold
    (terminal dock + layout) for now."""
    resp = app.make_response(render_template("pure.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/pure/frames", methods=["POST"])
def pure_frames():
    """Server-side video → frames for the PURE Data node. Uses imageio's bundled
    ffmpeg, so ANY format works (mkv, avi, mov, mp4, …) — the browser's <video>
    only handles mp4/webm/ogg. Returns flat 0-255 pixel arrays per frame."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no video file uploaded"}), 400
    try:
        size = max(2, min(256, int(request.form.get("size", 32))))
        start = max(0, int(request.form.get("start", 0)))
        skip = max(0, int(request.form.get("skip", 0)))
        maxf = max(1, min(4000, int(request.form.get("max", 64))))
        gray = request.form.get("gray", "1") == "1"
    except (TypeError, ValueError):
        return jsonify({"error": "bad frame params"}), 400
    import tempfile
    ext = os.path.splitext(f.filename or "video")[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.close()
    try:
        f.save(tmp.name)
        try:
            import imageio
            import numpy as np
            from PIL import Image
        except Exception as exc:
            return jsonify({"error": f"server missing video libs ({exc}); pip install imageio imageio-ffmpeg"}), 500
        reader = imageio.get_reader(tmp.name, "ffmpeg")
        stride = skip + 1
        frames = []
        try:
            for i, frame in enumerate(reader):
                if i < start or (i - start) % stride != 0:
                    continue
                img = Image.fromarray(frame).resize((size, size))
                img = img.convert("L") if gray else img.convert("RGB")
                frames.append(np.asarray(img).reshape(-1).astype(int).tolist())
                if len(frames) >= maxf:
                    break
        finally:
            reader.close()
        if not frames:
            return jsonify({"error": "no frames decoded — check the start frame vs the video length"}), 400
        return jsonify({"size": size, "gray": gray, "count": len(frames), "frames": frames})
    except Exception as exc:
        return jsonify({"error": f"could not decode this video: {exc}"}), 400
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/xray")
def xray_page():
    """X-Ray — stress test of the 'radial address space' theory: is a genome's
    rotation sequence a reversible, function-clustering coordinate? Real SO(3)
    experiments report which claims survive and which don't."""
    resp = app.make_response(render_template("xray.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/xray/transform", methods=["POST"])
def xray_transform():
    """Apply a solved MNIST genome to a shared sample of real digits and return
    each point's raw (tangled) and post-genome (clustered) 2D position, so the
    page can animate the separation. First call builds features (~11s), cached."""
    try:
        import genome_xray
        data = request.get_json(silent=True) or {}
        gid = str(data.get("genome", "r5"))
        use_pairs = bool(data.get("use_pairs", True))
        use_mixer = bool(data.get("use_mixer", True))
        npc = max(10, min(100, int(data.get("n_per_class", 50))))
        return jsonify(genome_xray.transform(gid, use_pairs, use_mixer, npc))
    except Exception as exc:
        return jsonify({"error": f"xray transform failed: {exc}"}), 500


@app.route("/radial")
def radial_page():
    """Radial Map v2 — deterministic index-addressed activation lenses,
    characterized by BEHAVIOR on baseline data (numeric loops first), projected
    to a 2D map; a closed-form linear model on the lens bank tests whether lens
    diversity alone does the heavy lifting. v1 archived in archive/radial_v1/."""
    resp = app.make_response(render_template("radial.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/radial/demo")
def radial_demo_page():
    """Radial rotation demo — no models: a 20x20x20 grid cube of red ground-
    truth dots centered on the origin, with a checkbox that rotates the DATA
    (uniform Y-axis rotation of every dot, not a camera effect) and a checkbox
    that overlays a stationary blue copy of the same cube for comparison."""
    resp = app.make_response(render_template("radial_demo.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/radial/demo/cousins")
def radial_cousins_page():
    """Space Cousin Finder — sub-page of the radial demo: a 6x6x6 grid of
    deterministic composed-activation lens programs is rotated about Y and
    rotated-onto neighbors are Pearson-correlated; pairs above threshold are
    'cousins' (behaviorally redundant views). Pure front-end, no models."""
    resp = app.make_response(render_template("radial_cousins.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/radial/demo/record", methods=["POST"])
def radial_demo_record():
    """Persist a cousin/sibling search from /radial/demo/cousins as a run in
    the runs store (runs/Demo_Radial/<rid>/), so it appears on the Runs page.
    Writes the standard config.json/history.jsonl/summary.json trio plus
    report.json — the full downloadable run report."""
    try:
        import datetime
        import hashlib
        d = request.get_json(silent=True) or {}
        kind = str(d.get("kind", "search"))[:24]
        params = d.get("params") or {}
        stats = d.get("stats") or {}
        log_lines = [str(x)[:300] for x in (d.get("log") or [])][:100]
        report = d.get("report") or {}
        ts = datetime.datetime.now()
        h = hashlib.sha1(json.dumps([kind, params], sort_keys=True,
                                    default=str).encode()).hexdigest()[:6]
        rid = f"{ts.strftime('%Y%m%d-%H%M%S')}-demoradial-{h}"
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "runs", "Demo_Radial", rid)
        os.makedirs(base, exist_ok=True)
        created = ts.isoformat(timespec="seconds")
        with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "environment": "Demo_Radial", "created": created,
                       "config": dict(params, kind=kind), "status": "finished"},
                      f, indent=2)
        with open(os.path.join(base, "history.jsonl"), "w", encoding="utf-8") as f:
            for i, line in enumerate(log_lines):
                f.write(json.dumps({"gen": i, "note": line}) + "\n")
        with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "environment": "Demo_Radial", "status": "finished",
                       "finished": created, "best": stats, "checkpoint": None},
                      f, indent=2)
        with open(os.path.join(base, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"label": f"{kind} search", "favorite": False,
                       "group": "", "tags": [kind, "radial-demo"]}, f, indent=2)
        with open(os.path.join(base, "report.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "kind": kind, "created": created,
                       "params": params, "stats": stats, "log": log_lines,
                       "report": report}, f, indent=2)
        return jsonify({"id": rid})
    except Exception as exc:
        return jsonify({"error": f"record failed: {exc}"}), 500


@app.route("/api/radial/demo/report/<rid>")
def radial_demo_report(rid):
    """Download a Demo_Radial run's report.json as an attachment."""
    if not rid.replace("-", "").replace("_", "").isalnum():
        return jsonify({"error": "bad run id"}), 400
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "runs", "Demo_Radial", rid)
    if not os.path.isfile(os.path.join(base, "report.json")):
        return jsonify({"error": "not found"}), 404
    resp = send_from_directory(base, "report.json", mimetype="application/json",
                               as_attachment=True)
    resp.headers["Content-Disposition"] = f"attachment; filename={rid}_report.json"
    return resp


@app.route("/api/radial/map", methods=["POST"])
def radial_map_api():
    try:
        import radial_map as rmap
        d = request.get_json(silent=True) or {}
        return jsonify(rmap.build_map(
            n_lens=max(50, min(2500, int(d.get("n", 1200)))),
            kind=str(d.get("kind", "loops"))))
    except Exception as exc:
        return jsonify({"error": f"radial map failed: {exc}"}), 500


@app.route("/api/radial/lens", methods=["POST"])
def radial_lens_api():
    try:
        import radial_map as rmap
        d = request.get_json(silent=True) or {}
        return jsonify(rmap.lens_detail(int(d.get("i", 0)),
                                        kind=str(d.get("kind", "loops"))))
    except Exception as exc:
        return jsonify({"error": f"radial lens failed: {exc}"}), 500


@app.route("/api/radial/training_state")
def radial_training_state():
    """Live numbers for the demo page: the gradient-free ladder, per-seed
    substrates, ensembles, and stacked stages — read from the run exports."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))

        def rd(path, *keys):
            try:
                with open(os.path.join(base, path)) as f:
                    d = _json.load(f)
                for k in keys:
                    d = d[k]
                return d
            except Exception:
                return None

        seeds = []
        for name, path in (
                ("seed 7 (local)", "radial_data/evo2x_cifar.json"),
                ("seed 13 (pod)", "runpod_shadow/radial_data/evo2x_cifar.json"),
                ("seed 19 (pod)", "runpod_shadow/radial_data/evo2_s19_cifar.json"),
                ("seed 29 (pod)", "runpod_shadow/radial_data/evo2_s29_cifar.json"),
                ("seed 37 (local)", "radial_data/evo2_s37_cifar.json"),
                ("seed 43 (local)", "radial_data/evo2_s43_cifar.json"),
                ("pop-128 (pod)", "runpod_shadow/radial_data/evo2p128_cifar.json")):
            t = rd(path, "test_acc")
            n = rd(path, "n_frozen")
            seeds.append({"name": name, "test": t, "n": n,
                          "status": "done" if t else "running"})
        ladder = [
            {"name": "raw pixels + ridge", "test": 0.324, "kind": "baseline"},
            {"name": "PCA + ridge", "test": 0.360, "kind": "baseline"},
            {"name": "Coates-Ng (hand-crafted)", "test": 0.5904, "kind": "baseline"},
            {"name": "v1 evolved genomes", "test": 0.6198, "kind": "evolved"},
            {"name": "v1 stacked tower", "test": 0.6378, "kind": "evolved"},
            {"name": "v2 grammar (all-gene)", "test": rd("radial_data/evo2_cifar.json", "test_acc") or 0.7035, "kind": "evolved"},
            {"name": "v3 stack (gates+meta)", "test": rd("radial_data/push80_cifar.json", "test_acc") or 0.7079, "kind": "evolved"},
            {"name": "fresh-val tower", "test": rd("radial_data/push80_s3_cifar.json", "test_acc") or 0.7144, "kind": "evolved"},
            {"name": "2-seed union", "test": rd("radial_data/ensemble2_cifar.json", "test_acc") or 0.7313, "kind": "ensemble"},
            {"name": "3-seed union", "test": rd("radial_data/ensemble3_cifar.json", "test_acc") or 0.7452, "kind": "ensemble"},
            {"name": "4-seed union", "test": rd("radial_data/ensemble4_cifar.json", "test_acc") or 0.7570, "kind": "ensemble"},
        ]
        fu = rd("runpod_shadow/radial_data/final_union_cifar.json", "test_acc")
        if fu:
            ladder.append({"name": "final union (all substrates)", "test": fu,
                           "kind": "ensemble"})
        return jsonify({"ladder": ladder, "seeds": seeds})
    except Exception as exc:
        return jsonify({"error": f"training state failed: {exc}"}), 500


@app.route("/api/animation/radial")
def animation_radial_state():
    """Live results for the rewired Animation tab: the temporal radial runs.
    ?task=path (default) or ?task=shape — same sequences, opposite labels."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        sfx = "_shape" if request.args.get("task") == "shape" else ""
        for rel in (f"radial_data/anim_radial{sfx}.json",
                    f"runpod_shadow/radial_data/anim_radial{sfx}.json"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation radial failed: {exc}"}), 500


@app.route("/api/animation/infer")
def animation_infer():
    """Run the saved temporal-radial checkpoint on random held-out test
    sequences. First call kicks off a one-time local rebuild (genomes from
    the checkpoint, head refit closed-form); returns {building: true} until
    ready — the page polls."""
    try:
        import anim_infer
        n = request.args.get("n", 12, type=int)
        task = request.args.get("task", "path")
        return jsonify(anim_infer.classify(n=max(1, min(32, n)), task=task))
    except Exception as exc:
        return jsonify({"error": f"animation infer failed: {exc}"}), 500


@app.route("/api/animation/ablation")
def animation_ablation():
    """Shape-checkpoint ablation suite results (unseen motion regimes)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ("radial_data/anim_ablation.json",
                    "runpod_shadow/radial_data/anim_ablation.json"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation ablation failed: {exc}"}), 500


@app.route("/api/animation/validation")
def animation_validation():
    """Adversarial validation suite results (anim_validate.py)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ("radial_data/anim_validation.json",
                    "runpod_shadow/radial_data/anim_validation.json"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation validation failed: {exc}"}), 500


@app.route("/api/animation/bg")
def animation_bg():
    """Scaling module — background-robustness A/B: the motion model trained on
    solid-black vs per-frame random-color backgrounds (matched settings)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        out = {}
        for tag in ("black", "color"):
            for rel in (f"radial_data/anim_radial_{tag}.json",
                        f"runpod_shadow/radial_data/anim_radial_{tag}.json"):
                path = os.path.join(base, rel)
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        out[tag] = _json.load(f)
                    break
        if not out:
            return jsonify({"pending": True})
        return jsonify(out)
    except Exception as exc:
        return jsonify({"error": f"animation bg failed: {exc}"}), 500


_BG_SAMPLE_KINDS = {"black": "black", "color": "randcolor", "inv": "inv"}


@app.route("/api/animation/bg_samples")
def animation_bg_samples():
    """Animated preview sequences for the scaling modules — a few 6-frame
    windows per background treatment as base64 RGB frames. ?kinds=black,color,inv
    (default black,color)."""
    try:
        import base64
        import radial_anim as ra
        n = max(1, min(12, request.args.get("n", 6, type=int)))
        size = request.args.get("size", "fixed")
        res = max(16, min(64, request.args.get("res", 32, type=int)))
        kinds = [k for k in request.args.get("kinds", "black,color").split(",")
                 if k in _BG_SAMPLE_KINDS]
        out = {}
        for tag in kinds:
            X8, y, ysh = ra.sample_seqs(n=n, bg=_BG_SAMPLE_KINDS[tag], seed=7,
                                        size=size, res=res)
            out[tag] = [{"data": base64.b64encode(X8[i].tobytes()).decode(),
                         "size": int(X8.shape[2]), "frames": int(X8.shape[1]),
                         "path": ra.PATH_NAMES[int(y[i])],
                         "shape": ra.SHAPE_NAMES[int(ysh[i])]} for i in range(len(X8))]
        return jsonify(out)
    except Exception as exc:
        return jsonify({"error": f"animation bg samples failed: {exc}"}), 500


@app.route("/api/animation/cursor")
def animation_cursor():
    """Attention thread, Model 1 — the cursor tracker following a moving cursor
    (dot_infer.py demo + dot_track.py metrics)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))

        def _load(name):
            for rel in (f"radial_data/{name}", f"runpod_shadow/radial_data/{name}"):
                p = os.path.join(base, rel)
                if os.path.exists(p):
                    with open(p, encoding="utf-8") as f:
                        return _json.load(f)
            return None
        demo = _load("dot_cursor_demo.json")
        if not demo:
            return jsonify({"pending": True})
        demo["metrics"] = _load("dot_track.json")
        return jsonify(demo)
    except Exception as exc:
        return jsonify({"error": f"animation cursor failed: {exc}"}), 500


@app.route("/api/animation/shape")
def animation_shape():
    """Attention thread, Model 1b — recognize the shape under the cursor via the
    tracker's own attention (dot_shape.py demo + result)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))

        def _load(name):
            for rel in (f"radial_data/{name}", f"runpod_shadow/radial_data/{name}"):
                p = os.path.join(base, rel)
                if os.path.exists(p):
                    with open(p, encoding="utf-8") as f:
                        return _json.load(f)
            return None
        demo = _load("dot_shape_demo.json")
        if not demo:
            return jsonify({"pending": True})
        demo["result"] = _load("dot_shape.json")
        return jsonify(demo)
    except Exception as exc:
        return jsonify({"error": f"animation shape failed: {exc}"}), 500


@app.route("/api/animation/footprint")
def animation_footprint():
    """Parameter counts + on-disk size for every model on the page (how tiny
    these gradient-free, CPU-inference models are)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ("radial_data/anim_footprint.json",
                    "runpod_shadow/radial_data/anim_footprint.json"):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation footprint failed: {exc}"}), 500


@app.route("/api/animation/ood")
def animation_ood():
    """Attention thread — out-of-distribution stress tests for the tracker +
    shape classifier (dot_ood.py export)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ("radial_data/dot_ood.json", "runpod_shadow/radial_data/dot_ood.json"):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation ood failed: {exc}"}), 500


@app.route("/api/animation/cursor_field")
def animation_cursor_field():
    """Interactive Model-1b — a random scene + a precomputed grid of the model's
    reads (where the cursor is + what shape is under it) so the browser can show
    live inference as the mouse moves. ?seed=<int>&stride=<int>. Runs torch
    inference locally; models must be present."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        need = ("dot_model.json", "dot_shape_model.json")
        if not all(os.path.exists(os.path.join(base, "radial_data", n)) for n in need):
            return jsonify({"pending": True,
                            "error": "run dot_track.py then dot_shape.py first"})
        seed = int(request.args.get("seed", 1))
        stride = max(1, min(4, int(request.args.get("stride", 2))))
        import dot_live
        try:
            return jsonify(dot_live.compute(seed, stride=stride))
        except Exception:
            try:                                    # transient GPU OOM: free and retry once
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return jsonify(dot_live.compute(seed, stride=stride))
    except Exception as exc:
        return jsonify({"error": f"animation cursor_field failed: {exc}"}), 500


@app.route("/api/animation/multires")
def animation_multires():
    """Scaling module 6 — one model across resolutions: the generalization
    matrix + the continue-on-resolution-mix repairs."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))

        def _load(name):
            for rel in (f"radial_data/{name}", f"runpod_shadow/radial_data/{name}"):
                p = os.path.join(base, rel)
                if os.path.exists(p):
                    with open(p, encoding="utf-8") as f:
                        return _json.load(f)
            return None
        out = {"matrix": _load("anim_multires.json"),
               "mix": _load("anim_continue_res.json"),
               "lowhigh": _load("anim_res_lowhigh.json")}
        if not any(out.values()):
            return jsonify({"pending": True})
        return jsonify(out)
    except Exception as exc:
        return jsonify({"error": f"animation multires failed: {exc}"}), 500


@app.route("/api/animation/res")
def animation_res():
    """Scaling module 5 — resolution scaling (crank): motion model trained
    natively at 32/48/64 (anim_res.py)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ("radial_data/anim_res.json",
                    "runpod_shadow/radial_data/anim_res.json"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation res failed: {exc}"}), 500


@app.route("/api/animation/size")
def animation_size():
    """Scaling module 4 — the random-color motion model, frozen, on shape sizes
    it never trained on (anim_size.py)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ("radial_data/anim_size.json",
                    "runpod_shadow/radial_data/anim_size.json"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation size failed: {exc}"}), 500


@app.route("/api/animation/continue")
def animation_continue():
    """Scaling module 3 — continue-training (warm-start) repairing the
    inverted-B&W weakness (anim_continue.py)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ("radial_data/anim_continue.json",
                    "runpod_shadow/radial_data/anim_continue.json"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation continue failed: {exc}"}), 500


@app.route("/api/animation/bg_ood")
def animation_bg_ood():
    """Scaling module 2 — the random-color motion model, frozen, on B&W /
    inverted-B&W regimes it never trained on (anim_bg_ood.py)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in ("radial_data/anim_bg_ood.json",
                    "runpod_shadow/radial_data/anim_bg_ood.json"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation bg ood failed: {exc}"}), 500


@app.route("/api/animation/genomes")
def animation_genomes():
    """Sample real genomes from a checkpoint so users can inspect the
    actual evolved parameters. ?task=path|shape&n=6"""
    try:
        import json as _json
        import random as _random
        task = request.args.get("task", "shape")
        n = max(1, min(12, request.args.get("n", 6, type=int)))
        sfx = "_shape" if task == "shape" else ""
        base = os.path.dirname(os.path.abspath(__file__))
        for rel in (f"radial_data/anim_model{sfx}.json",
                    f"runpod_shadow/radial_data/anim_model{sfx}.json"):
            path = os.path.join(base, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    ck = _json.load(f)
                import anim_infer
                rnd = _random.Random(request.args.get("seed", 0, type=int))
                out = []
                for si, sp in enumerate(ck["spaces"]):
                    for g in rnd.sample(sp, min(len(sp), max(1, n // len(ck["spaces"])))):
                        out.append({"space": si,
                                    "params": anim_infer.count_params(g),
                                    "genome": g})
                return jsonify({"task": task, "genomes": out[:n],
                                "n_spaces": len(ck["spaces"])})
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"animation genomes failed: {exc}"}), 500


@app.route("/api/animation/file/<name>")
def animation_file(name):
    """Download the raw artifacts (checkpoints + run exports) so anyone can
    audit them. Whitelisted names only."""
    allowed = {"anim_model.json", "anim_model_shape.json",
               "anim_radial.json", "anim_radial_shape.json",
               "anim_ablation.json", "anim_validation.json"}
    if name not in allowed:
        return jsonify({"error": "unknown artifact"}), 404
    base = os.path.dirname(os.path.abspath(__file__))
    for rel in ("radial_data", os.path.join("runpod_shadow", "radial_data")):
        path = os.path.join(base, rel, name)
        if os.path.exists(path):
            return send_file(path, as_attachment=True, download_name=name)
    return jsonify({"error": "artifact not generated yet"}), 404


@app.route("/api/lm/radial")
def lm_radial_state():
    """Live results for the rewired LM tab: the temporal radial stack on
    glyph-frame next-char prediction (+ n-gram ceilings in the export)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", "lm_radial.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"lm radial failed: {exc}"}), 500


@app.route("/api/lm/radial/word")
def lm_radial_word_state():
    """Word-level radial LM results (radial_lm_word.py export)."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", "lm_radial_word.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return jsonify(_json.load(f))
        return jsonify({"pending": True})
    except Exception as exc:
        return jsonify({"error": f"lm word failed: {exc}"}), 500


@app.route("/humanoid")
def humanoid_page():
    """Humanoid — the temporal radial stack on Humanoid-v5, gradient-free
    locomotion under a continuously raising distance bar."""
    return render_template("humanoid.html")


@app.route("/api/humanoid/modules")
def humanoid_modules():
    """The append-only module registry for the Humanoid page."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", "humanoid_modules.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return jsonify(_json.load(f))
        return jsonify({"modules": []})
    except Exception as exc:
        return jsonify({"error": f"humanoid modules failed: {exc}"}), 500


@app.route("/api/humanoid/export/<name>")
def humanoid_export(name):
    """Serve a Humanoid module's export json (whitelisted patterns only)."""
    try:
        import json as _json
        import re as _re
        if not _re.fullmatch(r"humanoid_[A-Za-z0-9_]*\.json", name):
            return jsonify({"error": "not a humanoid export"}), 404
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", name)
        if not os.path.exists(path):
            return jsonify({"pending": True})
        with open(path, encoding="utf-8") as f:
            return jsonify(_json.load(f))
    except Exception as exc:
        return jsonify({"error": f"humanoid export failed: {exc}"}), 500


@app.route("/history")
def history_page():
    """History — an append-only stack of timelines and cause-and-effect maps
    (mermaid diagrams), one module per iteration, newest at the bottom."""
    return render_template("history.html")


@app.route("/api/history/modules")
def history_modules():
    """The append-only module registry for the History page."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", "history_modules.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return jsonify(_json.load(f))
        return jsonify({"modules": []})
    except Exception as exc:
        return jsonify({"error": f"history modules failed: {exc}"}), 500


@app.route("/api/history/export/<name>")
def history_export(name):
    """Serve a History module's export json (whitelisted patterns only)."""
    try:
        import json as _json
        import re as _re
        if not _re.fullmatch(r"history_[A-Za-z0-9_]*\.json", name):
            return jsonify({"error": "not a history export"}), 404
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", name)
        if not os.path.exists(path):
            return jsonify({"pending": True})
        with open(path, encoding="utf-8") as f:
            return jsonify(_json.load(f))
    except Exception as exc:
        return jsonify({"error": f"history export failed: {exc}"}), 500


@app.route("/lm_demo")
def lm_demo_page():
    """LM Demo — the model's computation, traced live and animated."""
    resp = app.make_response(render_template("lm_demo.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/lm/demo_trace")
def lm_demo_trace():
    """The recorded real-generation trace that drives /lm_demo."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", "lm_demo_trace.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return jsonify(_json.load(f))
        return jsonify({"error": "trace not generated yet - run "
                                 "python lm/lm_demo_trace.py"})
    except Exception as exc:
        return jsonify({"error": f"demo trace failed: {exc}"}), 500


@app.route("/api/lm/modules")
def lm_modules():
    """The append-only module registry for the LM page."""
    try:
        import json as _json
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", "lm_modules.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return jsonify(_json.load(f))
        return jsonify({"modules": []})
    except Exception as exc:
        return jsonify({"error": f"lm modules failed: {exc}"}), 500


@app.route("/api/lm/export/<name>")
def lm_export(name):
    """Serve a module's export json (whitelisted patterns only)."""
    try:
        import json as _json
        import re as _re
        if not _re.fullmatch(
                r"(lm_radial|lm_probe|embed_report|kid_)[A-Za-z0-9_]*\.json",
                name):
            return jsonify({"error": "not an lm export"}), 404
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", name)
        if not os.path.exists(path):
            return jsonify({"pending": True})
        with open(path, encoding="utf-8") as f:
            return jsonify(_json.load(f))
    except Exception as exc:
        return jsonify({"error": f"lm export failed: {exc}"}), 500


@app.route("/api/lm/autocomplete")
def lm_autocomplete():
    """Autocomplete a prompt with the latest word-level radial checkpoint.
    First call kicks off a one-time background build; the page polls."""
    try:
        import lm_word_infer
        prompt = request.args.get("prompt", "", type=str)
        n = request.args.get("n", 24, type=int)
        temp = request.args.get("temp", 0.7, type=float)
        steer = request.args.get("steer", "auto", type=str)
        lam = request.args.get("lam", 1.5, type=float)
        topk = request.args.get("topk", 3, type=int)
        best = request.args.get("best", 1, type=int)
        ilam = request.args.get("intent", 0.5, type=float)
        return jsonify(lm_word_infer.complete(prompt, n_words=n, temp=temp,
                                              steer=steer, lam=lam,
                                              topk=topk, best_of=best,
                                              intent_lam=ilam))
    except Exception as exc:
        return jsonify({"error": f"autocomplete failed: {exc}"}), 500


@app.route("/api/lm/radial/examples")
def lm_radial_examples():
    """A few real test windows (context chars + the next char) for display."""
    try:
        import numpy as np
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "radial_data", "lm_ids.npz")
        if not os.path.exists(path):
            return jsonify({"pending": True})
        import radial_lm
        z = np.load(path)
        n = max(1, min(8, request.args.get("n", 3, type=int)))
        rng = np.random.default_rng(request.args.get("seed", 0, type=int))
        idx = rng.choice(len(z["yte"]), size=n, replace=False)
        out = []
        for i in idx:
            i = int(i)
            out.append({
                "context": "".join(radial_lm.CHARS[c] for c in z["ctx_te"][i]),
                "next": radial_lm.CHARS[int(z["yte"][i])],
            })
        return jsonify({"examples": out})
    except Exception as exc:
        return jsonify({"error": f"lm examples failed: {exc}"}), 500


@app.route("/api/radial/baselines")
def radial_baselines_api():
    """Serve the roadmap baseline exports (radial_data/baseline_*.json +
    prebaseline_fixes.json) for the page's Baselines view."""
    try:
        import glob
        import json as _json
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radial_data")
        out = {"domains": {}, "fixes": None}
        for path in sorted(glob.glob(os.path.join(base, "baseline_*.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    d = _json.load(f)
                out["domains"][d.get("domain", os.path.basename(path))] = d
            except Exception:
                continue
        fx = os.path.join(base, "prebaseline_fixes.json")
        if os.path.exists(fx):
            with open(fx) as f:
                out["fixes"] = _json.load(f)
        return jsonify(out)
    except Exception as exc:
        return jsonify({"error": f"radial baselines failed: {exc}"}), 500


@app.route("/api/radial/ladder", methods=["POST"])
def radial_ladder_api():
    """Auto-incrementing task ladder: pass at R2 >= threshold advances to the
    next harder task; first miss is the lens bank's frontier."""
    try:
        import radial_map as rmap
        d = request.get_json(silent=True) or {}
        return jsonify(rmap.ladder_probe(
            n_lens=max(20, min(1500, int(d.get("n", 400)))),
            kind=str(d.get("kind", "loops")),
            threshold=max(0.5, min(1.0, float(d.get("threshold", 0.998))))))
    except Exception as exc:
        return jsonify({"error": f"radial ladder failed: {exc}"}), 500


@app.route("/api/radial/rotate", methods=["POST"])
def radial_rotate_api():
    """Rotate the 3D-embedded map about the Y axis 1 deg/step; per angle the
    linear probe sees only the slice of lenses in the viewing plane."""
    try:
        import radial_map as rmap
        d = request.get_json(silent=True) or {}
        return jsonify(rmap.rotation_probe(
            n_lens=max(100, min(1500, int(d.get("n", 800)))),
            kind=str(d.get("kind", "loops")),
            step_deg=max(0.5, min(15.0, float(d.get("step_deg", 1.0)))),
            frac=max(0.005, min(0.5, float(d.get("frac", 0.03))))))
    except Exception as exc:
        return jsonify({"error": f"radial rotate failed: {exc}"}), 500


@app.route("/api/radial/probe", methods=["POST"])
def radial_probe_api():
    try:
        import radial_map as rmap
        d = request.get_json(silent=True) or {}
        return jsonify(rmap.probe(n_lens=max(20, min(1500, int(d.get("n", 400)))),
                                  kind=str(d.get("kind", "loops"))))
    except Exception as exc:
        return jsonify({"error": f"radial probe failed: {exc}"}), 500


# --- ResNet: gradient-free evolved residual networks on CIFAR --------------

@app.route("/resnet")
def resnet_page():
    """ResNet (gradient-free) — evolved residual-block genomes on CIFAR-10. The
    whole lab is gradient-free (rule #1); this asks the ResNet question inside
    that law: can evolution discover the residual skip and stack it usefully,
    scored only by a closed-form linear read-out? Pipeline in resnet_evo.py;
    artifacts on F:\\Resnet."""
    resp = app.make_response(render_template("resnet.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/progress")
def progress_page():
    """PROGRESS — a dashboard over the master CHANGELOG.md: per-project activity
    over time, measurable completion toward each project's goal, and an
    impact-weighted timeline that separates scientific advancement from raw
    velocity (activity != progress)."""
    resp = app.make_response(render_template("progress.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/progress/data")
def api_progress_data():
    try:
        import progress_service
        resp = jsonify(progress_service.parse())
        resp.headers["Cache-Control"] = "no-store"   # always re-parse on visit
        return resp
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/vision_demo")
def vision_demo_page():
    """VISION DEMO — showcases two gradient-free staples on vision-grounded models:
    (1) UNION of a frozen shape recognizer + a frozen letter recognizer into one
    36-class head, and (2) CONTINUED TRAINING of the shape model until it also
    reads letters (one model, no separate letter model). Data-driven from
    radial_data/vision_demo.json (built by mm/vision_demo.py)."""
    resp = app.make_response(render_template("vision_demo.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/vision_demo/data")
def api_vision_demo_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "radial_data", "vision_demo.json")
    if not os.path.isfile(path):
        return jsonify({"error": "vision_demo.json not built yet — run mm/vision_demo.py"}), 404
    try:
        with open(path, encoding="utf-8") as f:
            resp = jsonify(json.load(f))
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/vision_demo/samples")
def api_vision_demo_samples():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "radial_data", "vision_demo_samples.json")
    if not os.path.isfile(path):
        return jsonify({"error": "samples not built — run mm/vision_samples.py"}), 404
    with open(path, encoding="utf-8") as f:
        resp = jsonify(json.load(f))
    resp.headers["Cache-Control"] = "no-store"
    return resp


_VISION_DL = {
    "shape":     ("multimodal/dot_shape_model.json",       "dot_shape_model.json"),
    "letter":    ("multimodal/kid_modelA.json",            "kid_modelA.json"),
    "union":     ("multimodal/mm_model.json",              "mm_model.json"),
    "continued": ("multimodal/vision_continue_model.json", "vision_continue_model.json"),
    "infer":     ("mm/vision_infer.py",                    "vision_infer.py"),
}
_VISION_README = (
    "GENREG vision demo — try the gradient-free vision checkpoints yourself.\n\n"
    "Files:\n"
    "  vision_infer.py             the inference CLI\n"
    "  dot_shape_model.json        shape recognizer (10 classes)\n"
    "  kid_modelA.json             letter recognizer (26 classes)\n"
    "  mm_model.json               UNION: shapes+letters fused, one 36-class head\n"
    "  vision_continue_model.json  CONTINUED: the shape model grown to read letters\n\n"
    "Run (from inside the GENREG repo, with numpy/torch/pillow installed):\n"
    "  python vision_infer.py --model continued\n"
    "  python vision_infer.py --model shape --n 12 --save out/\n\n"
    "The script reuses the repo's radial grammar so predictions match the page.\n"
    "Gradient-free: no training here, only a closed-form ridge readout.\n"
)


@app.route("/api/vision_demo/download/<name>")
def api_vision_demo_download(name):
    root = os.path.dirname(os.path.abspath(__file__))
    if name == "bundle":
        import io as _io
        import zipfile
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for rel, arc in _VISION_DL.values():
                p = os.path.join(root, rel)
                if os.path.isfile(p):
                    z.write(p, arc)
            z.writestr("README.txt", _VISION_README)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True,
                         download_name="genreg_vision_demo.zip")
    if name not in _VISION_DL:
        return jsonify({"error": "unknown download"}), 404
    rel, arc = _VISION_DL[name]
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        return jsonify({"error": f"{arc} not built yet"}), 404
    return send_file(p, as_attachment=True, download_name=arc)


# ── OCR (gradient-free screen reading, built up in stages) ──────────────────
@app.route("/ocr")
def ocr_page():
    """OCR — gradient-free recognizer trained in a rich environment (moving, colored,
    varying contrast, many fonts/sizes), built up toward screen reading. Data-driven
    from radial_data/ocr.json (assembled by ocr/ocr_demo.py)."""
    resp = app.make_response(render_template("ocr.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/ocr/modules")
def api_ocr_modules():
    """Append-only module registry for the /ocr iteration log (the /lm pattern)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radial_data",
                        "ocr_modules.json")
    reg = {"modules": []}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            reg = json.load(f)
    resp = jsonify(reg)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/ocr/export/<name>")
def api_ocr_export(name):
    """Serve a module's export json — whitelisted filename pattern only."""
    if "/" in name or "\\" in name or ".." in name or \
            not (name.startswith("ocr_") and name.endswith(".json")):
        return jsonify({"error": "bad export name"}), 400
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radial_data", name)
    if not os.path.isfile(path):
        return jsonify({"pending": True, "error": f"{name} not built yet"}), 404
    with open(path, encoding="utf-8") as f:
        resp = jsonify(json.load(f))
    resp.headers["Cache-Control"] = "no-store"
    return resp


_OCR_DL = {
    "digits": ("ocr/models/digits_model.json", "digits_model.json"),
    "letters": ("ocr/models/letters_model.json", "letters_model.json"),
    "alnum": ("ocr/models/alnum_model.json", "alnum_model.json"),
    "infer": ("ocr/ocr_infer.py", "ocr_infer.py"),
    "glyphs": ("ocr/ocr_glyphs.py", "ocr_glyphs.py"),
    "model": ("ocr/ocr_model.py", "ocr_model.py"),
}
_OCR_README = (
    "GENREG OCR — test the gradient-free glyph recognizers yourself.\n\n"
    "Files:\n"
    "  ocr_infer.py         inference CLI\n"
    "  ocr_glyphs.py        the rich-environment renderer (moving/colored/multi-font)\n"
    "  ocr_model.py         checkpoint loader + replay\n"
    "  <charset>_model.json trained recognizer(s)\n\n"
    "Run (from inside the GENREG repo, numpy/torch/pillow):\n"
    "  python ocr_infer.py --charset digits --n 8 --save out/\n\n"
    "Gradient-free: no training here, only a closed-form ridge readout.\n"
)


@app.route("/api/ocr/download/<name>")
def api_ocr_download(name):
    root = os.path.dirname(os.path.abspath(__file__))
    if name == "bundle":
        import io as _io
        import zipfile
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for rel, arc in _OCR_DL.values():
                p = os.path.join(root, rel)
                if os.path.isfile(p):
                    z.write(p, arc)
            z.writestr("README.txt", _OCR_README)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True,
                         download_name="genreg_ocr.zip")
    if name not in _OCR_DL:
        return jsonify({"error": "unknown download"}), 404
    rel, arc = _OCR_DL[name]
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        return jsonify({"error": f"{arc} not built yet — train the model first"}), 404
    return send_file(p, as_attachment=True, download_name=arc)


# ── Replicate (recognize audio by replicating it — temporal radial, realtime) ─
@app.route("/replicate")
def replicate_page():
    """Replicate — multimodal convergence. A concept exists in every perceptual
    system at once: the visual union (the eyes), the private language (label-free
    contrastive encoders), the letter bank (symbols), the shape bank (geometry).
    One ridge head notices they agree. Campaign 1 is CIFAR with every face; audio
    (realtime temporal replication) comes after. Data-driven from
    radial_data/replicate_*.json (the /lm append-only pattern)."""
    resp = app.make_response(render_template("replicate.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/replicate/modules")
def api_replicate_modules():
    """Append-only module registry for the /replicate iteration log."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radial_data",
                        "replicate_modules.json")
    reg = {"modules": []}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            reg = json.load(f)
    resp = jsonify(reg)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/replicate/export/<name>")
def api_replicate_export(name):
    """Serve a module's export json — whitelisted filename pattern only."""
    if "/" in name or "\\" in name or ".." in name or \
            not (name.startswith("replicate_") and name.endswith(".json")):
        return jsonify({"error": "bad export name"}), 400
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radial_data", name)
    if not os.path.isfile(path):
        return jsonify({"pending": True, "error": f"{name} not built yet"}), 404
    with open(path, encoding="utf-8") as f:
        resp = jsonify(json.load(f))
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ── OCR lineage: node-graph model-lineage editor + server-side execution ────
@app.route("/ocr/lineage")
def ocr_lineage_page():
    """Node-graph editor: wire Dataset -> Train -> Union/Eval and execute the graph;
    each node fills in with real performance. Backend: ocr/lineage.py."""
    resp = app.make_response(render_template("lineage.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/ocr/lineage/models")
def api_ocr_lineage_models():
    from ocr import lineage
    resp = jsonify({"models": lineage.list_models(),
                    "charsets": ["digits", "letters", "alnum"]})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/ocr/lineage/run", methods=["POST"])
def api_ocr_lineage_run():
    from ocr import lineage
    graph = request.get_json(silent=True) or {}
    if not graph.get("nodes"):
        return jsonify({"error": "empty graph"}), 400
    try:
        job_id = lineage.run_graph(graph)
        return jsonify({"job": job_id})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ocr/lineage/status/<job_id>")
def api_ocr_lineage_status(job_id):
    from ocr import lineage
    job = lineage.get_job(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    resp = jsonify(job)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _resnet_out_dir():
    """Where resnet_evo.py writes its result JSON (F:\\Resnet, or the local
    radial_data fallback when F: is absent — mirrors resnet_evo.OUT_DIR)."""
    d = os.environ.get("GENREG_RESNET_DIR", r"F:\Resnet")
    if not os.path.isdir(d):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "radial_data")
    return d


@app.route("/api/resnet/result")
def resnet_result_api():
    """Serve the latest resnet-evo result JSON for the page. Prefers the full
    run (resnet_evo_cifar.json) and falls back to the smoke output."""
    try:
        d = _resnet_out_dir()
        for name in ("resnet_evo_cifar.json", "resnet_evo_smoke.json"):
            path = os.path.join(d, name)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    out = json.load(f)
                out["_source"] = name
                out["_dir"] = d
                return jsonify(out)
        return jsonify({"error": "no resnet-evo result yet — run resnet_evo.py "
                                 "(or `python resnet_evo.py --smoke`)",
                        "_dir": d}), 404
    except Exception as exc:
        return jsonify({"error": f"resnet result failed: {exc}"}), 500


@app.route("/video")
def video_page():
    """Video — ffmpeg-backed editor: cut, stitch, convert, export."""
    resp = app.make_response(render_template("video.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/video/status")
def api_video_status():
    return jsonify({"ok": VID_OK, "err": VID_ERR,
                    "formats": video_service.FORMATS if VID_OK else []})


@app.route("/api/poses")
def api_poses_list():
    folder = "C:/Users/paytonm/Pictures/poses"
    if not os.path.isdir(folder):
        return jsonify([])
    try:
        files = []
        for root, _, filenames in os.walk(folder):
            for f in filenames:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    rel = os.path.relpath(os.path.join(root, f), folder)
                    files.append(rel.replace("\\", "/"))
        files.sort()
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/poses/<path:filename>")
def api_poses_serve(filename):
    folder = "C:/Users/paytonm/Pictures/poses"
    return send_from_directory(folder, filename)


@app.route("/api/charts")
def api_charts_list():
    if not VID_OK:
        return jsonify([])
    try:
        files = sorted([f for f in os.listdir(video_service.LIB_DIR)
                        if os.path.isfile(os.path.join(video_service.LIB_DIR, f)) and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))])
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/render_slides", methods=["POST"])
def api_video_render_slides():
    data = request.get_json(silent=True) or {}
    slides = data.get("slides") or []
    out_name = data.get("out_name", "")
    fps = int(data.get("fps", 24))
    w = int(data.get("w", 1280))
    h = int(data.get("h", 720))
    try:
        job = anim_service.render_slides(slides, out_name=out_name, fps=fps, w=w, h=h)
        return jsonify(video_service.job_view(job))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/video/meta")
def api_video_meta():
    """Duration (s) of one library file - 0 for still images."""
    try:
        name = video_service.safe_name(request.args.get("name", ""))
        src = os.path.normpath(os.path.join(video_service.LIB_DIR, name))
        if not name or not src.startswith(
                os.path.normpath(video_service.LIB_DIR))                 or not os.path.isfile(src):
            return jsonify({"error": "no such file"}), 404
        meta = video_service._meta(name, src)
        return jsonify({"duration": round(float(meta.get("duration", 0) or 0), 3)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/video/videos")
def api_video_videos():
    """Library VIDEO files (the Gemini-generated animations live here)."""
    if not VID_OK:
        return jsonify([])
    try:
        files = sorted([f for f in os.listdir(video_service.LIB_DIR)
                        if os.path.isfile(os.path.join(video_service.LIB_DIR, f))
                        and f.lower().endswith((".mp4", ".webm", ".mov",
                                                ".mkv", ".gif"))])
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/mute", methods=["POST"])
def api_video_mute():
    """Write a muted copy of a library video (ffmpeg -an, video stream
    copied untouched) as <name>_muted.<ext>."""
    try:
        data = request.get_json(silent=True) or {}
        name = video_service.safe_name(data.get("name", ""))
        src = os.path.normpath(os.path.join(video_service.LIB_DIR, name))
        if not name or not src.startswith(
                os.path.normpath(video_service.LIB_DIR))                 or not os.path.isfile(src):
            return jsonify({"error": "no such video"}), 404
        stem, ext = os.path.splitext(name)
        out = video_service.unique_path(stem + "_muted" + ext)
        import subprocess as _sp
        r = _sp.run([video_service.FFMPEG, "-y", "-i", src,
                     "-c:v", "copy", "-an", out],
                    capture_output=True, text=True, timeout=300,
                    creationflags=video_service.CREATE_NO_WINDOW)
        if r.returncode != 0:
            err_tail = " | ".join(
                r.stderr.strip().splitlines()[-4:])
            return jsonify({"error": err_tail}), 500
        video_service.invalidate_meta(os.path.basename(out))
        return jsonify({"ok": True, "output": os.path.basename(out)})
    except Exception as exc:
        return jsonify({"error": f"mute failed: {exc}"}), 500


@app.route("/api/video/tts", methods=["POST"])
def api_video_tts():
    """Generate narration via ElevenLabs into a slide clip. Key from
    ELEVENLABS_API_KEY env or .keys/elevenlabs.key (gitignored)."""
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text or len(text) > 4000:
            return jsonify({"error": "text empty or too long"}), 400
        key = ""
        for name in ("ELEVENLABS_API_KEY", "ElevenLabs", "ELEVENLABS"):
            key = os.environ.get(name, "")
            if key:
                break
        if not key:
            kp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".keys", "elevenlabs.key")
            if os.path.isfile(kp):
                with open(kp) as f:
                    key = f.read().strip()
        if not key:
            return jsonify({"error": "no ElevenLabs key - set the "
                            "ElevenLabs env var or put it in "
                            ".keys/elevenlabs.key"}), 503
        voice = (data.get("voice") or "").strip() or "nxNsTXLZ8x7PeZNBs9Js"
        import re as _re
        if not _re.fullmatch(r"[A-Za-z0-9]{8,40}", voice):
            return jsonify({"error": "bad voice id"}), 400

        # credit guard: identical (voice, text) reuses the existing clip
        import hashlib as _hl
        norm = " ".join(text.split())
        ck = _hl.sha256(f"{voice}|{norm}".encode()).hexdigest()
        cache_path = os.path.join(anim_service.SLIDE_AUDIO_DIR,
                                  "tts_cache.json")
        cache = {}
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cache = json.load(f)
            except (ValueError, OSError):
                cache = {}
        hit = cache.get(ck)
        if hit and os.path.isfile(os.path.join(anim_service.SLIDE_AUDIO_DIR,
                                               hit.get("id", ""))):
            return jsonify({"id": hit["id"], "dur": hit["dur"],
                            "cached": True})
        import urllib.request as _ur
        req = _ur.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
            data=json.dumps({
                "text": text,
                "model_id": "eleven_multilingual_v2",
            }).encode(),
            headers={"xi-api-key": key,
                     "Content-Type": "application/json",
                     "Accept": "audio/mpeg"})
        with _ur.urlopen(req, timeout=120) as resp:
            audio = resp.read()
        if not audio:
            return jsonify({"error": "empty TTS response"}), 502
        import time as _time
        import secrets as _secrets
        cid = f"{int(_time.time())}_{_secrets.token_hex(4)}.mp3"
        path = os.path.join(anim_service.SLIDE_AUDIO_DIR, cid)
        with open(path, "wb") as f:
            f.write(audio)
        dur = float(video_service.probe(path).get("duration", 0) or 0)
        cache[ck] = {"id": cid, "dur": round(dur, 2), "voice": voice,
                     "text": norm, "created": int(_time.time())}
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1)
        except OSError:
            pass
        return jsonify({"id": cid, "dur": round(dur, 2), "cached": False})
    except Exception as exc:
        return jsonify({"error": f"tts failed: {exc}"}), 500


@app.route("/api/video/slide_audio", methods=["POST"])
def api_slide_audio_upload():
    """Save a browser mic recording (webm/opus blob) for a slide clip."""
    try:
        import time as _time
        import secrets as _secrets
        data = request.get_data()
        if not data or len(data) > 30_000_000:
            return jsonify({"error": "empty or oversized recording"}), 400
        cid = f"{int(_time.time())}_{_secrets.token_hex(4)}.webm"
        path = os.path.join(anim_service.SLIDE_AUDIO_DIR, cid)
        with open(path, "wb") as f:
            f.write(data)
        return jsonify({"id": cid})
    except Exception as exc:
        return jsonify({"error": f"save failed: {exc}"}), 500


@app.route("/api/video/slide_audio/<cid>", methods=["GET", "DELETE"])
def api_slide_audio(cid):
    """Serve or delete one slide clip (whitelisted id pattern)."""
    import re as _re
    if not _re.fullmatch(r"[0-9]+_[0-9a-f]+\.(webm|mp3)", cid):
        return jsonify({"error": "bad clip id"}), 400
    path = os.path.join(anim_service.SLIDE_AUDIO_DIR, cid)
    if not os.path.isfile(path):
        return jsonify({"error": "no such clip"}), 404
    if request.method == "DELETE":
        purge = request.args.get("purge") in ("1", "true")
        if cid.endswith(".mp3"):
            cache_path = os.path.join(anim_service.SLIDE_AUDIO_DIR,
                                      "tts_cache.json")
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cch = json.load(f)
                owned = [k for k, v in cch.items() if v.get("id") == cid]
                if owned and not purge:
                    # generated narration: keep the file for credit-free
                    # reuse; the slide reference is gone client-side
                    return jsonify({"ok": True, "kept": True})
                if owned and purge:
                    # permanent delete: drop the cache entry too, so the
                    # same line re-bills if narrated again
                    for k in owned:
                        cch.pop(k, None)
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(cch, f, indent=1)
                    except OSError:
                        pass
            except (ValueError, OSError):
                pass
        try:
            os.unlink(path)
            return jsonify({"ok": True})
        except OSError:
            # Windows: the clip may still be streaming (browser playback
            # holds the handle). Retry shortly in the background; the
            # slide reference is already gone client-side either way.
            import threading as _th

            def _retry():
                import time as _t
                for _ in range(10):
                    _t.sleep(3)
                    try:
                        os.unlink(path)
                        return
                    except OSError:
                        pass
            _th.Thread(target=_retry, daemon=True).start()
            return jsonify({"ok": True, "deferred": True})
    from flask import send_file
    mt = "audio/mpeg" if cid.endswith(".mp3") else "audio/webm"
    return send_file(path, mimetype=mt)


@app.route("/api/video/tts_map")
def api_video_tts_map():
    """Reverse lookup for the Audio Studio: clip id -> the script line
    it narrates (from the TTS credit-guard cache)."""
    cache_path = os.path.join(anim_service.SLIDE_AUDIO_DIR,
                              "tts_cache.json")
    out = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            for v in cache.values():
                if v.get("id"):
                    out[v["id"]] = {"text": v.get("text", ""),
                                    "voice": v.get("voice", ""),
                                    "dur": v.get("dur", 0)}
        except (ValueError, OSError):
            pass
    return jsonify(out)


@app.route("/api/video/library")
def api_video_library():
    if not VID_OK:
        return jsonify({"error": f"video unavailable: {VID_ERR}"}), 503
    return jsonify(video_service.list_library())


@app.route("/api/video/upload", methods=["POST"])
def api_video_upload():
    if not VID_OK:
        return jsonify({"error": f"video unavailable: {VID_ERR}"}), 503
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file uploaded"}), 400
    name = video_service.safe_name(f.filename)
    allowed = (video_service.VIDEO_EXTS | video_service.AUDIO_EXTS
               | {".png", ".jpg", ".jpeg", ".gif", ".webp"})
    if os.path.splitext(name)[1].lower() not in allowed:
        return jsonify({"error": "not a recognized media format"}), 400
    path = video_service.unique_path(name)
    f.save(path)
    return jsonify({"ok": True, "name": os.path.basename(path)})


@app.route("/api/video/library/<name>", methods=["DELETE"])
def api_video_delete(name):
    if not VID_OK:
        return jsonify({"error": f"video unavailable: {VID_ERR}"}), 503
    ok = video_service.delete(name)
    return (jsonify({"ok": True}), 200) if ok else (jsonify({"error": "not found"}), 404)


@app.route("/api/video/file/<name>")
def api_video_file(name):
    """Serve a library file with Range support (browser <video> seeking)."""
    if not VID_OK:
        return jsonify({"error": f"video unavailable: {VID_ERR}"}), 503
    as_download = request.args.get("download") == "1"
    return send_from_directory(video_service.LIB_DIR, video_service.safe_name(name),
                               as_attachment=as_download, conditional=True)


@app.route("/api/video/thumb/<name>")
def api_video_thumb(name):
    if not VID_OK:
        return jsonify({"error": f"video unavailable: {VID_ERR}"}), 503
    path = video_service.thumbnail(name)
    if not path:
        return jsonify({"error": "no thumbnail"}), 404
    return send_from_directory(video_service.THUMB_DIR, os.path.basename(path),
                               mimetype="image/jpeg")


@app.route("/api/video/job", methods=["POST"])
def api_video_job():
    """Start a background ffmpeg job. op: cut | stitch | convert."""
    if not VID_OK:
        return jsonify({"error": f"video unavailable: {VID_ERR}"}), 503
    data = request.get_json(silent=True) or {}
    op = data.get("op")
    try:
        if op == "convert":
            job = video_service.convert(
                data.get("name", ""), data.get("format", "mp4"),
                crf=int(data.get("crf", 23)),
                scale_h=data.get("scale_h") or None, fps=data.get("fps") or None)
        elif op == "cut":
            job = video_service.trim(
                data.get("name", ""), float(data.get("start", 0)),
                float(data.get("end", 0)), precise=bool(data.get("precise", True)),
                crf=int(data.get("crf", 23)))
        elif op == "stitch":
            job = video_service.stitch(
                data.get("clips", []), fmt=data.get("format", "mp4"),
                crf=int(data.get("crf", 23)), scale_h=data.get("scale_h") or None,
                fps=data.get("fps") or None, out_name=data.get("out_name", ""))
        else:
            return jsonify({"error": f"unknown op: {op}"}), 400
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(video_service.job_view(job))


@app.route("/api/video/jobs")
def api_video_jobs():
    if not VID_OK:
        return jsonify([])
    return jsonify(video_service.list_jobs())


@app.route("/api/video/job/<job_id>/cancel", methods=["POST"])
def api_video_cancel(job_id):
    if not VID_OK:
        return jsonify({"error": f"video unavailable: {VID_ERR}"}), 503
    ok = video_service.cancel(job_id)
    return (jsonify({"ok": True}), 200) if ok else (jsonify({"error": "not cancellable"}), 404)


# ── animation platform (rigs / scenes / render) ────────────────────────────
def _anim_guard():
    if not ANIMP_OK:
        return jsonify({"error": f"animation unavailable: {ANIMP_ERR}"}), 503
    return None


@app.route("/api/anim/status")
def api_anim_status():
    if not ANIMP_OK:
        return jsonify({"ok": False, "err": ANIMP_ERR})
    return jsonify({"ok": True, "raster": anim_service.RASTER_OK,
                    "raster_err": anim_service.RASTER_ERR,
                    "archetypes": anim_service.ARCHETYPES,
                    "scene_templates": anim_service.SCENE_TEMPLATES,
                    "tags": anim_service.TAGS})


@app.route("/api/anim/rigs")
def api_anim_rigs():
    return _anim_guard() or jsonify(anim_service.list_rigs())


@app.route("/api/anim/rigs", methods=["POST"])
def api_anim_rig_save():
    guard = _anim_guard()
    if guard:
        return guard
    rig = request.get_json(silent=True) or {}
    try:
        return jsonify(anim_service.save_rig(rig))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/anim/rig/<name>", methods=["DELETE"])
def api_anim_rig_delete(name):
    guard = _anim_guard()
    if guard:
        return guard
    ok = anim_service.delete_rig(name)
    return (jsonify({"ok": True}), 200) if ok else (jsonify({"error": "not found"}), 404)


@app.route("/api/anim/generate", methods=["POST"])
def api_anim_generate():
    guard = _anim_guard()
    if guard:
        return guard
    data = request.get_json(silent=True) or {}
    try:
        rig = anim_service.generate_rig(data.get("archetype", "researcher"),
                                        seed=data.get("seed"),
                                        name=data.get("name", ""))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(rig)


@app.route("/api/anim/generate_scene", methods=["POST"])
def api_anim_generate_scene():
    """Template scene: bg palette + generated prop rigs placed as objects."""
    guard = _anim_guard()
    if guard:
        return guard
    data = request.get_json(silent=True) or {}
    try:
        scene = anim_service.generate_scene(data.get("template", "basic"),
                                            seed=data.get("seed"),
                                            name=data.get("name", ""))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(scene)


@app.route("/api/anim/scenes")
def api_anim_scenes():
    return _anim_guard() or jsonify(anim_service.list_scenes())


@app.route("/api/anim/scenes", methods=["POST"])
def api_anim_scene_save():
    guard = _anim_guard()
    if guard:
        return guard
    scene = request.get_json(silent=True) or {}
    try:
        return jsonify(anim_service.save_scene(scene))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/anim/scene/<name>", methods=["DELETE"])
def api_anim_scene_delete(name):
    guard = _anim_guard()
    if guard:
        return guard
    ok = anim_service.delete_scene(name)
    return (jsonify({"ok": True}), 200) if ok else (jsonify({"error": "not found"}), 404)


@app.route("/api/anim/stories")
def api_anim_stories():
    return _anim_guard() or jsonify(anim_service.list_stories())


@app.route("/api/anim/stories", methods=["POST"])
def api_anim_story_save():
    guard = _anim_guard()
    if guard:
        return guard
    story = request.get_json(silent=True) or {}
    try:
        return jsonify(anim_service.save_story(story))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/anim/story/<name>", methods=["DELETE"])
def api_anim_story_delete(name):
    guard = _anim_guard()
    if guard:
        return guard
    ok = anim_service.delete_story(name)
    return (jsonify({"ok": True}), 200) if ok else (jsonify({"error": "not found"}), 404)


@app.route("/api/anim/render_story", methods=["POST"])
def api_anim_render_story():
    """Render an ordered shot list (saved scene names) into one library mp4."""
    guard = _anim_guard()
    if guard:
        return guard
    data = request.get_json(silent=True) or {}
    try:
        job = anim_service.render_story(data.get("shots") or [],
                                        out_name=data.get("out_name", ""))
    except (TypeError, ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(video_service.job_view(job))


@app.route("/api/anim/render", methods=["POST"])
def api_anim_render():
    """Render a scene (sent inline) to an mp4 in the video library."""
    guard = _anim_guard()
    if guard:
        return guard
    scene = request.get_json(silent=True) or {}
    try:
        job = anim_service.render_scene(scene)
    except (TypeError, ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(video_service.job_view(job))


@app.route("/api/animations")
def api_animations():
    """The ten procedural shape-motion clips (24 x 64 x 64 each), rendered on
    demand. `data` is base64 of the raw uint8 frames (frames*size*size bytes,
    row-major) — the page decodes it straight into canvas ImageData."""
    import base64

    from genreg_train import animation_data

    shape_of_clip = {name: shape.__name__
                     for name, _, shape in animation_data.ANIMATIONS}
    out = []
    for name, frames in animation_data.generate_all().items():
        u8 = (frames * 255.0 + 0.5).astype("uint8")
        out.append({
            "name": name,
            "shape": shape_of_clip[name],
            "frames": int(u8.shape[0]),
            "size": int(u8.shape[1]),
            "data": base64.b64encode(u8.tobytes()).decode("ascii"),
        })
    return jsonify(out)


@sock.route("/animevo")
def animevo(ws):
    """Animation Evo socket — same viewer-onto-hub shape as /diffuse: training
    runs in animation_evo.HUB and survives navigation; (re)connecting replays
    the journal snapshot.

    Browser -> server: {"op":"start", ...config} | {"op":"stop"}.
    Server -> browser: JSON events (started / gen / sample / done / error).
    """
    send_lock = threading.Lock()

    def emit(ev):
        payload = json.dumps(ev)
        with send_lock:
            ws.send(payload)

    if not ANIM_OK:
        try:
            emit({"type": "error", "message": f"animevo unavailable: {ANIM_ERR}"})
        except Exception:
            pass
        return

    hub = animation_evo.HUB
    try:
        for ev in hub.snapshot():          # rebuild mid-run state
            emit(ev)
        emit({"type": "job", "running": hub.running()})
    except Exception:
        return
    hub.subscribe(emit)

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                continue
            op = data.get("op")
            if op == "start":
                hub.start(data, animation_evo.AnimEvoTrainer)
            elif op == "stop":
                hub.stop()
    except Exception:
        pass
    finally:
        hub.unsubscribe(emit)              # detach only — training keeps going


@sock.route("/diffuse")
def diffuse(ws):
    """DiffEvo socket — a viewer onto the server-side job hub, same shape as
    /evolang: training runs in the hub and survives navigation; (re)connecting
    replays the journal snapshot so the page rebuilds mid-run.

    Browser -> server: {"op":"start", ...config} | {"op":"stop"}.
    Server -> browser: JSON events (started / level_start / gen / level_done /
    sample / done / error), every message tagged with `type`.
    """
    send_lock = threading.Lock()

    def emit(ev):
        payload = json.dumps(ev)
        with send_lock:                    # hub thread + handler thread both send
            ws.send(payload)

    if not DIFF_OK:
        try:
            emit({"type": "error", "message": f"diffevo unavailable: {DIFF_ERR}"})
        except Exception:
            pass
        return

    hub = diffuse_service.HUB
    try:
        for ev in hub.snapshot():          # rebuild mid-run state
            emit(ev)
        emit({"type": "job", "running": hub.running()})
    except Exception:
        return
    hub.subscribe(emit)

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                continue
            op = data.get("op")
            if op == "start":
                hub.start(data, diffuse_service.DiffuseTrainer)
            elif op == "stop":
                hub.stop()
    except Exception:
        pass
    finally:
        hub.unsubscribe(emit)              # detach only — training keeps going


@sock.route("/train")
def train(ws):
    """Run a neuroevolution training job and stream per-generation events.

    Protocol (browser -> server): {"op":"start", ...config} | {"op":"stop"}.
    Server -> browser: JSON events (started / generation / done / error). The
    Trainer runs in a background thread and sends via `emit`; this handler thread
    keeps reading so a `stop` (or disconnect) can cancel it — same pattern as the
    terminal bridge below.
    """
    lock = threading.Lock()
    state = {"trainer": None, "thread": None}

    def emit(ev):
        try:
            ws.send(json.dumps(ev))
        except Exception:
            # browser gone: stop the run so the thread doesn't linger
            with lock:
                if state["trainer"]:
                    state["trainer"].stop()

    def start(cfg):
        if not TRAIN_OK:
            emit({"type": "error", "message": f"training unavailable: {TRAIN_ERR}"})
            return
        with lock:
            if state["trainer"]:
                state["trainer"].stop()
            old = state["thread"]
        if old is not None:
            old.join(timeout=5.0)

        # persist this run (config -> per-gen metrics -> checkpoint) as events flow
        holder, run_ref = {}, {"run": None}

        def store_emit(ev):
            try:
                t = ev.get("type")
                if t == "started":
                    run_ref["run"] = runstore.create_run(cfg, ev)
                elif t == "generation" and run_ref["run"]:
                    runstore.append_metric(run_ref["run"], ev)
                elif t == "done" and run_ref["run"]:
                    champ = holder["trainer"].champion() if holder.get("trainer") else None
                    runstore.finalize(run_ref["run"], ev, champ)
                    if BOARD_OK:   # end-of-run alarm for the Agent panel
                        agent_board.post_run_event(
                            cfg.get("environment", "engine"),
                            {**ev, "run_id": run_ref["run"]["id"]})
                elif t == "error" and BOARD_OK:   # crash alarm (run may not exist yet)
                    agent_board.post_run_event(
                        cfg.get("environment", "engine"),
                        {**ev, "run_id": (run_ref["run"] or {}).get("id")})
            except Exception:
                pass
            emit(ev)

        try:
            trainer = create_trainer(cfg, store_emit)
        except Exception as exc:
            emit({"type": "error", "message": f"bad config: {exc}"})
            return
        holder["trainer"] = trainer
        th = threading.Thread(target=trainer.run, name="genreg-train", daemon=True)
        with lock:
            state["trainer"], state["thread"] = trainer, th
        th.start()

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                continue
            op = data.get("op")
            if op == "start":
                start(data)
            elif op == "stop":
                with lock:
                    if state["trainer"]:
                        state["trainer"].stop()
    finally:
        with lock:
            if state["trainer"]:
                state["trainer"].stop()


@sock.route("/ws")
def ws(ws):
    """Bridge one browser WebSocket to a dedicated daemon connection."""
    if not ensure_daemon():
        ws.send(json.dumps({"type": "system", "id": 0,
                            "data": "\r\n[cannot reach terminal daemon]\r\n"}))
        return
    try:
        ds = socket.create_connection((DAEMON_HOST, DAEMON_PORT), timeout=5)
    except OSError:
        return
    ds.sendall(b'{"op":"hello"}\n')

    stop = threading.Event()

    def daemon_to_browser():
        buf = b""
        ds.settimeout(1.0)
        while not stop.is_set():
            try:
                chunk = ds.recv(16384)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        ws.send(line.decode("utf-8"))
            except socket.timeout:
                continue
            except OSError:
                break
        stop.set()
        try:
            ws.close()
        except Exception:
            pass

    pump = threading.Thread(target=daemon_to_browser, daemon=True)
    pump.start()

    try:
        while not stop.is_set():
            msg = ws.receive()          # blocks; None when the browser disconnects
            if msg is None:
                break
            line = msg.strip()
            if line:
                ds.sendall((line + "\n").encode("utf-8"))
    except Exception:
        pass
    finally:
        stop.set()
        try:
            ds.close()
        except OSError:
            pass


if __name__ == "__main__":
    ensure_daemon()
    print("GENREG interface running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
