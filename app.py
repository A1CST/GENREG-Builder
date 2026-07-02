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

from flask import Flask, render_template, jsonify, send_from_directory
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


@app.route("/api/runs/<rid>/traces")
def api_traces(rid):
    if not TRAIN_OK:
        return jsonify([])
    return jsonify(runstore.get_traces(rid))


@app.route("/api/runs/<rid>/replay")
def api_replay(rid):
    if not TRAIN_OK:
        return jsonify({"error": "training unavailable"}), 503
    try:
        r = runstore.infer(rid)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return (jsonify(r), 200) if r else (jsonify({"error": "no checkpoint"}), 404)


@app.route("/tree")
def tree_page():
    resp = app.make_response(render_template("tree.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
