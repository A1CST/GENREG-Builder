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

import json
import socket
import threading
import time

from winpty import PtyProcess

HOST = "127.0.0.1"
PORT = 5001

SHELL = ["powershell.exe", "-NoLogo", "-NoProfile"]
BUFFER_LIMIT = 256 * 1024        # per-terminal replay buffer (characters)


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
        self.lock = threading.Lock()
        self._next_id = 1

    def broadcast(self, event):
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

    def close(self, tid):
        with self.lock:
            term = self.terminals.pop(tid, None)
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
                        terms = [t.snapshot() for t in self.terminals.values()]
                    conn.sendall((json.dumps({"type": "snapshot", "terminals": terms}) + "\n").encode("utf-8"))
                elif op == "create":
                    self.create(msg.get("title"), int(msg.get("cols", 100)), int(msg.get("rows", 30)))
                else:
                    tid = msg.get("id")
                    with self.lock:
                        term = self.terminals.get(tid)
                    if term:
                        if op == "input":
                            term.write(msg.get("data", ""))
                        elif op == "resize":
                            term.resize(int(msg.get("cols", 100)), int(msg.get("rows", 30)))
                        elif op == "restart":
                            term.restart()
                        elif op == "stop":
                            term.stop()
                        elif op == "clear":
                            term.clear()
                        elif op == "close":
                            self.close(tid)
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
