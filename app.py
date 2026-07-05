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

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_sock import Sock

app = Flask(__name__)
app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
# Pick up template + static edits without needing a manual cache-buster.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
sock = Sock(app)

# Neuroevolution training (Snake / 2048). Imported lazily-guarded so a missing
# numpy/engine never stops the terminal interface from serving.
try:
    from genreg_train import create_trainer, parse_config, runstore
    TRAIN_OK, TRAIN_ERR = True, None
except Exception as _e:                       # pragma: no cover
    TRAIN_OK, TRAIN_ERR = False, str(_e)

# Tree-of-Models text prediction (separate program from the build interface).
try:
    from genreg_train import tree_service
    TREE_OK, TREE_ERR = True, None
except Exception as _e:                       # pragma: no cover
    TREE_OK, TREE_ERR = False, str(_e)

# I2 latent-content node (separate program; shares this server).
try:
    import i2_service
    I2_OK, I2_ERR = True, None
except Exception as _e:                       # pragma: no cover
    I2_OK, I2_ERR = False, str(_e)

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

# Agent board — shared notice feed for the floating Agent panel (all pages).
try:
    import agent_board
    BOARD_OK = True
except Exception:                             # pragma: no cover
    agent_board, BOARD_OK = None, False

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 5001
DAEMON_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terminal_daemon.py")


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


@app.route("/tree")
def tree_page():
    resp = app.make_response(render_template("tree.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
    /treelm: training runs in the hub and survives navigation; (re)connecting
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


@sock.route("/treelm")
def treelm(ws):
    """Tree-of-Models socket — a *viewer* onto the server-side job hub.

    Browser -> server: {"op":"start"|"sweep", ...config} | {"op":"stop"} |
                       {"op":"generate"|"trace", prompt, length, temperature}.
    Server -> browser: JSON events (corpus / tree / node_* / eval / sweep_* /
    done / generated / trace / error).

    Training runs in the hub (tree_service.HUB) and is NOT tied to this
    socket: navigating away only detaches the viewer; re-connecting replays
    the journal snapshot so the page rebuilds mid-run. Stopping is explicit
    (stop op or starting a new job).
    """
    send_lock = threading.Lock()

    def emit(ev):
        payload = json.dumps(ev)
        with send_lock:                    # hub thread + handler thread both send
            ws.send(payload)

    if not TREE_OK:
        try:
            emit({"type": "error", "message": f"tree-of-models unavailable: {TREE_ERR}"})
        except Exception:
            pass
        return

    hub = tree_service.HUB
    try:
        emit({"type": "model", "available": tree_service.has_model(),
              "info": tree_service.model_info()})
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
                hub.start(data, tree_service.TreeLMTrainer)
            elif op == "sweep":
                hub.start(data, tree_service.TreeSweeper)
            elif op == "train_encoder":
                hub.start(data, tree_service.EncoderTrainer)
            elif op == "encoders":
                try:
                    emit({"type": "encoders", "list": tree_service.list_encoders()})
                except Exception:
                    pass
            elif op == "stop":
                hub.stop()
            elif op == "generate":
                try:
                    text = tree_service.generate_text(
                        str(data.get("prompt", "")),
                        length=int(data.get("length", 300)),
                        temperature=float(data.get("temperature", 0.0)))
                    emit({"type": "generated", "prompt": data.get("prompt", ""),
                          "text": text})
                except Exception as exc:
                    emit({"type": "error", "message": str(exc)})
            elif op == "trace":
                try:
                    tr = tree_service.trace_generate(
                        str(data.get("prompt", "")),
                        length=int(data.get("length", 48)),
                        temperature=float(data.get("temperature", 0.8)))
                    emit({"type": "trace", **tr})
                except Exception as exc:
                    emit({"type": "error", "message": str(exc)})
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
