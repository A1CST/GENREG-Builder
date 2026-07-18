"""GENREG terminal daemon (PTY edition).

A standalone, long-lived process that owns real ConPTY-backed PowerShell
terminals (via pywinpty). Because it is a separate process from the Flask
web server, you can stop/restart Flask while your terminals — and any
interactive program running in them, including the `claude` TUI — keep
running. On reconnect, each terminal's recent screen output is replayed.

Protocol: newline-delimited JSON over a localhost TCP socket.

  Client -> daemon ops (one JSON object per line):
    {"op":"hello"}                                subscribe + receive a snapshot
    {"op":"create", "cols":C, "rows":R, "title":"..."}
    {"op":"input",  "id":N, "data":"..."}         raw keystrokes -> PTY stdin
    {"op":"resize", "id":N, "cols":C, "rows":R}
    {"op":"restart","id":N}                        kill + respawn the shell
    {"op":"stop",   "id":N}                        kill the shell process
    {"op":"clear",  "id":N}                        clear scrollback buffer
    {"op":"close",  "id":N}                        kill + remove the terminal

  daemon -> subscriber events (one JSON object per line):
    {"type":"output", "id":N, "data":"...raw ANSI..."}
    {"type":"system", "id":N, "data":"..."}
    {"type":"clear",  "id":N}
    {"type":"terminal_created", "id":N, "title":..., "alive":..., "cols":..., "rows":...}
    {"type":"terminal_closed",  "id":N}
  The reply to {"op":"hello"} is a single {"type":"snapshot","terminals":[...]}
  where each terminal carries its concatenated recent output as "data".

Run standalone:  python terminal_daemon.py
(app.py spawns this automatically if it isn't already running.)
"""

import datetime
import json
import os
import queue
import re
import socket
import threading
import time

from winpty import PtyProcess

HOST = "127.0.0.1"
PORT = 5001

SHELL = ["powershell.exe", "-NoLogo", "-NoProfile"]
BUFFER_LIMIT = 256 * 1024        # per-terminal replay buffer (characters)

# When a tab is closed we DON'T kill the shell right away — the session is held
# for this long so an accidental close can be re-opened with its process and
# scrollback intact. After the grace window with no reopen, it is reaped.
CLOSE_GRACE = 300                # seconds (5 minutes)

# Where the Ghostwriter records terminal transcripts.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "terminal_logs")

# Strip ANSI so the plain-text transcript reads like the on-screen conversation.
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSI_OTHER = re.compile(r"\x1b[=>()#][0-9A-Za-z]?|\x1b[@-Z\\-_]")


def _strip_ansi(s):
    s = _ANSI_OSC.sub("", s)
    s = _ANSI_CSI.sub("", s)
    s = _ANSI_OTHER.sub("", s)
    return s.replace("\x07", "").replace("\r", "")


class Ghostwriter:
    """Background transcript logger. Runs on its OWN thread so terminal I/O and
    the socket broadcast never block on disk. Every terminal event is queued
    and written two ways: a machine-readable JSONL stream (raw bytes, one line
    per event) and a human-readable per-terminal .txt of the screen output.
    """

    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        self.q = queue.Queue()
        self.titles = {}                 # id -> last-known title (for headers)
        self._txt = {}                   # id -> open text handle
        self._jsonl = None
        self._stop = object()            # sentinel
        os.makedirs(self.log_dir, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="ghostwriter",
                                       daemon=True)
        self.thread.start()

    def record(self, kind, tid, data="", title=None):
        """Queue one event. kind: out | in | sys | life. Non-blocking."""
        try:
            self.q.put_nowait({
                "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "id": tid, "dir": kind, "title": title, "data": data,
            })
        except queue.Full:              # pragma: no cover — unbounded by default
            pass

    def _jsonl_handle(self):
        if self._jsonl is None:
            stamp = datetime.date.today().isoformat()
            path = os.path.join(self.log_dir, f"session-{stamp}.jsonl")
            self._jsonl = open(path, "a", encoding="utf-8")
        return self._jsonl

    def _txt_handle(self, tid):
        h = self._txt.get(tid)
        if h is None:
            path = os.path.join(self.log_dir, f"terminal-{tid}.txt")
            h = open(path, "a", encoding="utf-8")
            title = self.titles.get(tid, f"Terminal {tid}")
            now = datetime.datetime.now().isoformat(timespec="seconds")
            h.write(f"\n===== session attached {now} — {title} =====\n")
            self._txt[tid] = h
        return h

    def _run(self):
        while True:
            ev = self.q.get()
            if ev is self._stop:
                break
            try:
                if ev.get("title"):
                    self.titles[ev["id"]] = ev["title"]
                # machine log: every event, raw
                self._jsonl_handle().write(json.dumps(ev, ensure_ascii=False) + "\n")
                # readable log: on-screen output only (keystrokes/control noise
                # stay in the JSONL), ANSI stripped.
                if ev["dir"] == "out" and ev.get("data"):
                    self._txt_handle(ev["id"]).write(_strip_ansi(ev["data"]))
            except OSError:
                pass
            # drain-then-flush so bursts don't fsync per line
            if self.q.empty():
                try:
                    if self._jsonl:
                        self._jsonl.flush()
                    for h in self._txt.values():
                        h.flush()
                except OSError:
                    pass


class Terminal:
    def __init__(self, tid, title, on_event, cols=100, rows=30):
        self.id = tid
        self.title = title
        self.on_event = on_event
        self.cols = cols
        self.rows = rows
        self.buffer = []          # list[str] of raw output chunks
        self.buflen = 0
        self.alive = False
        self.pty = None
        self._lock = threading.Lock()
        self.start()

    # -- process lifecycle -------------------------------------------------
    def start(self):
        with self._lock:
            self.pty = PtyProcess.spawn(SHELL, dimensions=(self.rows, self.cols))
            self.alive = True
        threading.Thread(target=self._reader, args=(self.pty,), daemon=True).start()

    def restart(self):
        self._kill()
        self.buffer = []
        self.buflen = 0
        self._emit({"type": "clear"}, buffer=False)
        self._emit({"type": "system", "data": "\r\n\x1b[33m[shell restarted]\x1b[0m\r\n"})
        self.start()

    def stop(self):
        self._kill()
        self._emit({"type": "system", "data": "\r\n\x1b[33m[shell stopped — press Restart]\x1b[0m\r\n"})

    def clear(self):
        self.buffer = []
        self.buflen = 0
        self._emit({"type": "clear"}, buffer=False)

    def _kill(self):
        with self._lock:
            self.alive = False
            try:
                if self.pty and self.pty.isalive():
                    self.pty.terminate(force=True)
            except Exception:
                pass

    # -- io ----------------------------------------------------------------
    def _reader(self, pty):
        try:
            while True:
                try:
                    data = pty.read(4096)
                except EOFError:
                    break
                if not data:
                    time.sleep(0.01)
                    continue
                self._emit({"type": "output", "data": data})
        except (OSError, ValueError):
            pass
        if self.alive:
            self.alive = False
            self._emit({"type": "system", "data": "\r\n\x1b[33m[process exited]\x1b[0m\r\n"})

    def write(self, data):
        try:
            if self.alive and self.pty:
                self.pty.write(data)
        except (OSError, ValueError):
            pass

    def resize(self, cols, rows):
        self.cols, self.rows = cols, rows
        try:
            if self.alive and self.pty:
                self.pty.setwinsize(rows, cols)
        except (OSError, ValueError):
            pass

    # -- events ------------------------------------------------------------
    def _emit(self, event, buffer=True):
        event = {**event, "id": self.id}
        if buffer and "data" in event:
            self.buffer.append(event["data"])
            self.buflen += len(event["data"])
            while self.buflen > BUFFER_LIMIT and len(self.buffer) > 1:
                self.buflen -= len(self.buffer.pop(0))
        self.on_event(event)

    def meta(self):
        return {"id": self.id, "title": self.title, "alive": self.alive,
                "cols": self.cols, "rows": self.rows}

    def snapshot(self):
        m = self.meta()
        m["data"] = "".join(self.buffer)
        return m


class Daemon:
    def __init__(self):
        self.terminals = {}
        self.subscribers = set()
        self.pending = {}             # tid -> {"timer": Timer, "deadline": ts}
        self.lock = threading.Lock()
        self._next_id = 1
        self.ghost = Ghostwriter()

    # map a broadcast event onto a Ghostwriter direction tag
    _GHOST_DIR = {"output": "out", "system": "sys",
                  "terminal_created": "life", "terminal_closed": "life",
                  "terminal_detached": "life", "terminal_restored": "life"}

    def broadcast(self, event):
        d = self._GHOST_DIR.get(event.get("type"))
        if d is not None:
            self.ghost.record(d, event.get("id"), event.get("data", ""),
                              title=event.get("title"))
        data = (json.dumps(event) + "\n").encode("utf-8")
        with self.lock:
            subs = list(self.subscribers)
        for sock in subs:
            try:
                sock.sendall(data)
            except OSError:
                with self.lock:
                    self.subscribers.discard(sock)

    def create(self, title=None, cols=100, rows=30):
        with self.lock:
            tid = self._next_id
            self._next_id += 1
        title = title or f"Terminal {tid}"
        term = Terminal(tid, title, self.broadcast, cols=cols, rows=rows)
        with self.lock:
            self.terminals[tid] = term
        self.broadcast({"type": "terminal_created", **term.meta()})
        return term

    def detach(self, tid):
        """Soft close: keep the shell alive and hold the session for
        CLOSE_GRACE seconds so an accidental close can be re-opened. After the
        grace window with no reopen, the session is reaped for good."""
        with self.lock:
            term = self.terminals.get(tid)
            if not term or tid in self.pending:
                return
            timer = threading.Timer(CLOSE_GRACE, self._reap, args=(tid,))
            timer.daemon = True
            self.pending[tid] = {"timer": timer,
                                 "deadline": time.time() + CLOSE_GRACE}
            timer.start()
        mins = CLOSE_GRACE // 60
        term._emit({"type": "system", "data":
                    f"\r\n\x1b[33m[tab closed — session held for {mins} min; "
                    f"reopen to resume]\x1b[0m\r\n"})
        self.broadcast({"type": "terminal_detached", "id": tid,
                        "grace": CLOSE_GRACE})

    def reopen(self, tid):
        """Cancel a pending reap and hand the client a fresh snapshot so it can
        rebuild the tab with full scrollback and the still-running process."""
        with self.lock:
            p = self.pending.pop(tid, None)
            term = self.terminals.get(tid)
        if p:
            p["timer"].cancel()
        if term:
            snap = term.snapshot()
            self.broadcast({"type": "terminal_restored", **snap})

    def _reap(self, tid):
        """Grace expired — actually kill and remove the terminal."""
        with self.lock:
            self.pending.pop(tid, None)
            term = self.terminals.pop(tid, None)
        if term:
            term._kill()
            self.broadcast({"type": "terminal_closed", "id": tid})

    def close(self, tid):
        """Hard close now (used by the reap path / explicit kill)."""
        with self.lock:
            p = self.pending.pop(tid, None)
            term = self.terminals.pop(tid, None)
        if p:
            p["timer"].cancel()
        if term:
            term._kill()
            self.broadcast({"type": "terminal_closed", "id": tid})

    def handle(self, conn):
        f = conn.makefile("r", encoding="utf-8")
        is_sub = False
        try:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                op = msg.get("op")
                if op == "hello":
                    with self.lock:
                        self.subscribers.add(conn)
                        is_sub = True
                        now = time.time()
                        terms = []
                        for t in self.terminals.values():
                            snap = t.snapshot()
                            p = self.pending.get(t.id)
                            if p:            # held after a close — client shows it reopenable
                                snap["detached"] = True
                                snap["grace_remaining"] = max(0, round(p["deadline"] - now))
                            terms.append(snap)
                    conn.sendall((json.dumps({"type": "snapshot", "terminals": terms}) + "\n").encode("utf-8"))
                elif op == "create":
                    self.create(msg.get("title"), int(msg.get("cols", 100)), int(msg.get("rows", 30)))
                else:
                    tid = msg.get("id")
                    with self.lock:
                        term = self.terminals.get(tid)
                    if op == "reopen":
                        self.reopen(tid)
                    elif op == "close":
                        self.detach(tid)         # soft close with grace window
                    elif op == "kill":
                        self.close(tid)          # explicit hard close now
                    elif term:
                        if op == "input":
                            self.ghost.record("in", tid, msg.get("data", ""))
                            term.write(msg.get("data", ""))
                        elif op == "resize":
                            term.resize(int(msg.get("cols", 100)), int(msg.get("rows", 30)))
                        elif op == "restart":
                            term.restart()
                        elif op == "stop":
                            term.stop()
                        elif op == "clear":
                            term.clear()
        except OSError:
            pass
        finally:
            if is_sub:
                with self.lock:
                    self.subscribers.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((HOST, PORT))
        except OSError:
            return                # another daemon already owns the port
        srv.listen(64)
        self.create()             # start with one terminal ready
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    Daemon().serve()
