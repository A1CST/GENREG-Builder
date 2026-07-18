"""PIA - Personal AI: a local Ollama RAG assistant over the project's docs.

PIA answers questions about what you are working on by retrieving from the
project documentation (documentation/, CHANGELOG.md, README.md, RESUME.md and
the per-page changelogs) and grounding a local Ollama chat model on the top
matches. It is a plain retrieval-augmented-generation loop:

    docs -> chunk -> embed (nomic-embed-text) -> vector index on F:\\
    question -> embed -> cosine top-k -> stuff into the chat prompt -> answer

Design rules honored here:
  * Models live on F:\\ (OLLAMA_MODELS = F:\\Ollama\\models), never C:.
  * The Ollama server is NEVER started automatically. start_server() is only
    ever called from the PIA page (POST /api/pia/start). Import and status
    checks are side-effect free.

The vector index is a single pickle on F:\\ so nothing heavy touches C:.
Everything talks to Ollama over its local HTTP API (127.0.0.1:11434).
"""

import json
import os
import pickle
import subprocess
import threading
import time

import requests

# --- locations ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documentation")

# Everything Ollama-related is kept off C:, on F:, per the user's directive.
F_ROOT = os.environ.get("PIA_F_ROOT", r"F:\Ollama")
MODELS_DIR = os.path.join(F_ROOT, "models")
INDEX_PATH = os.path.join(F_ROOT, "pia_index.pkl")

# --- ollama config --------------------------------------------------------
OLLAMA_HOST = os.environ.get("PIA_OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}"
CHAT_MODEL = os.environ.get("PIA_CHAT_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("PIA_EMBED_MODEL", "nomic-embed-text")

# Top-level project files that are not under documentation/ but are core context.
EXTRA_FILES = ["CHANGELOG.md", "README.md", "RESUME.md"]
TEXT_EXTS = {".md", ".markdown", ".txt", ".json", ".log", ".csv"}

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
RETRIEVE_K = 6

# --- shared state ---------------------------------------------------------
_lock = threading.Lock()
_proc = None                      # the `ollama serve` process we started (if any)
_index = None                     # loaded index dict (lazy)
_job = {"kind": None, "state": "idle", "msg": "", "done": 0, "total": 0}


def _set_job(**kw):
    with _lock:
        _job.update(kw)


def job_status():
    with _lock:
        return dict(_job)


# =========================================================================
# Ollama discovery / process control
# =========================================================================
def ollama_exe():
    """Best-effort locate ollama.exe (winget installs under LOCALAPPDATA)."""
    cands = []
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        cands.append(os.path.join(local, "Programs", "Ollama", "ollama.exe"))
    cands.append(r"C:\Program Files\Ollama\ollama.exe")
    for c in cands:
        if os.path.isfile(c):
            return c
    # Fall back to PATH resolution.
    from shutil import which
    return which("ollama")


def is_installed():
    return ollama_exe() is not None


def _child_env():
    """Environment for a server we spawn: force models onto F:."""
    env = dict(os.environ)
    env["OLLAMA_MODELS"] = MODELS_DIR
    return env


def _persist_models_env():
    """Persist OLLAMA_MODELS for the user so any future ollama uses F: too.

    Best-effort; failure is non-fatal (the spawned server still gets F: via
    its own environment)."""
    try:
        if os.environ.get("OLLAMA_MODELS") == MODELS_DIR:
            return
        subprocess.run(["setx", "OLLAMA_MODELS", MODELS_DIR],
                       capture_output=True, timeout=10)
        os.environ["OLLAMA_MODELS"] = MODELS_DIR
    except Exception:
        pass


def is_running():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def start_server(wait=20):
    """Start `ollama serve` with models on F:. Manual invocation only.

    No-op (reports already-running) if a server is already answering on the
    port. Returns (ok, message)."""
    global _proc
    if is_running():
        return True, "Ollama already running."
    exe = ollama_exe()
    if not exe:
        return False, "ollama.exe not found - is Ollama installed?"
    os.makedirs(MODELS_DIR, exist_ok=True)
    _persist_models_env()
    try:
        # Detached so it survives request lifetime; no console window.
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        with _lock:
            _proc = subprocess.Popen(
                [exe, "serve"],
                env=_child_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
    except Exception as exc:
        return False, f"failed to launch ollama serve: {exc}"
    # Wait for the API to come up.
    for _ in range(wait * 2):
        if is_running():
            return True, "Ollama started."
        time.sleep(0.5)
    return False, "started ollama but the API did not come up in time."


def stop_server():
    """Stop the server we started (and any stray ollama serve). Manual only."""
    global _proc
    stopped = False
    with _lock:
        p = _proc
        _proc = None
    if p is not None:
        try:
            p.terminate()
            stopped = True
        except Exception:
            pass
    # Also stop any other ollama process we can see, so a fresh F:-configured
    # server governs on next start. This includes the desktop tray app
    # ("ollama app.exe"), which otherwise auto-respawns a C:-path server and
    # breaks both the manual-start and F:-storage guarantees.
    if os.name == "nt":
        for image in ("ollama app.exe", "ollama.exe"):
            try:
                subprocess.run(["taskkill", "/F", "/IM", image],
                               capture_output=True, timeout=10)
                stopped = True
            except Exception:
                pass
    return stopped, ("Ollama stopped." if stopped else "Nothing to stop.")


def list_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def _has_model(name, present):
    # ollama tags carry ":latest"; match on the base too.
    base = name.split(":")[0]
    return any(p == name or p.split(":")[0] == base for p in present)


def pull_models_async():
    """Pull chat + embed models onto F: in a background thread."""
    def _run():
        _set_job(kind="pull", state="running", msg="starting pull...",
                 done=0, total=2)
        present = list_models()
        wanted = [CHAT_MODEL, EMBED_MODEL]
        done = 0
        for name in wanted:
            if _has_model(name, present):
                done += 1
                _set_job(done=done, msg=f"{name} already present")
                continue
            _set_job(msg=f"pulling {name} (this can take a while)...")
            try:
                with requests.post(f"{OLLAMA_URL}/api/pull",
                                   json={"name": name}, stream=True,
                                   timeout=None) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        status = ev.get("status", "")
                        if ev.get("total"):
                            gb = ev.get("completed", 0) / 1e9
                            tot = ev.get("total", 1) / 1e9
                            _set_job(msg=f"{name}: {status} {gb:.1f}/{tot:.1f} GB")
                        else:
                            _set_job(msg=f"{name}: {status}")
                done += 1
                _set_job(done=done)
            except requests.RequestException as exc:
                _set_job(state="error", msg=f"pull failed for {name}: {exc}")
                return
        _set_job(state="done", msg="models ready")

    threading.Thread(target=_run, name="pia-pull", daemon=True).start()


# =========================================================================
# Document ingestion + embedding index
# =========================================================================
def iter_doc_files():
    """Yield absolute paths of every text document PIA should ingest."""
    seen = set()
    for name in EXTRA_FILES:
        p = os.path.join(BASE_DIR, name)
        if os.path.isfile(p):
            seen.add(p)
            yield p
    if os.path.isdir(DOCS_DIR):
        for root, _dirs, files in os.walk(DOCS_DIR):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in TEXT_EXTS:
                    p = os.path.join(root, fn)
                    if p not in seen:
                        seen.add(p)
                        yield p


def _rel(path):
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")


def chunk_text(text):
    """Greedy char chunks with overlap; splits on blank lines when it can."""
    text = text.replace("\r\n", "\n")
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        # Prefer to end on a paragraph/line break inside the tail window.
        if end < n:
            window = text[i:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"))
            if cut > CHUNK_CHARS // 2:
                end = i + cut
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end <= i:
            end = i + CHUNK_CHARS
        i = max(end - CHUNK_OVERLAP, end) if end < n else n
    return chunks


def embed_one(text):
    r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json().get("embedding", [])


def _cosine_prep(vecs):
    """Return (matrix, norms) for fast cosine without numpy dependency risk."""
    import numpy as np
    m = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(m, axis=1)
    norms[norms == 0] = 1.0
    return m, norms


def build_index_async():
    """Chunk + embed every doc, save the vector index to F:. Background."""
    def _run():
        import numpy as np
        if not is_running():
            _set_job(kind="index", state="error",
                     msg="start Ollama first (models must be loadable)")
            return
        files = list(iter_doc_files())
        _set_job(kind="index", state="running", msg="reading documents...",
                 done=0, total=len(files))
        records, vectors = [], []
        for fi, path in enumerate(files):
            rel = _rel(path)
            _set_job(done=fi, msg=f"embedding {rel}")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            for ci, chunk in enumerate(chunk_text(text)):
                try:
                    vec = embed_one(chunk)
                except requests.RequestException as exc:
                    _set_job(state="error", msg=f"embed failed on {rel}: {exc}")
                    return
                if not vec:
                    continue
                records.append({"source": rel, "chunk": ci, "text": chunk})
                vectors.append(vec)
        if not vectors:
            _set_job(state="error", msg="no document chunks were embedded")
            return
        _set_job(msg="saving index to F:...", done=len(files))
        mat = np.asarray(vectors, dtype="float32")
        os.makedirs(F_ROOT, exist_ok=True)
        data = {
            "model": EMBED_MODEL,
            "built_at": time.time(),
            "records": records,
            "vectors": mat,
            "n_files": len(files),
        }
        with open(INDEX_PATH, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        global _index
        with _lock:
            _index = data
        _set_job(state="done",
                 msg=f"indexed {len(records)} chunks from {len(files)} files")

    threading.Thread(target=_run, name="pia-index", daemon=True).start()


def _load_index():
    global _index
    with _lock:
        if _index is not None:
            return _index
    if os.path.isfile(INDEX_PATH):
        try:
            with open(INDEX_PATH, "rb") as fh:
                data = pickle.load(fh)
            with _lock:
                _index = data
            return data
        except Exception:
            return None
    return None


def index_status():
    data = _load_index()
    if not data:
        return {"built": False}
    return {
        "built": True,
        "chunks": len(data.get("records", [])),
        "files": data.get("n_files", 0),
        "model": data.get("model", ""),
        "built_at": data.get("built_at", 0),
    }


def retrieve(query, k=RETRIEVE_K):
    import numpy as np
    data = _load_index()
    if not data:
        return []
    qv = embed_one(query)
    if not qv:
        return []
    q = np.asarray(qv, dtype="float32")
    qn = np.linalg.norm(q) or 1.0
    mat = data["vectors"]
    mn = np.linalg.norm(mat, axis=1)
    mn[mn == 0] = 1.0
    sims = (mat @ q) / (mn * qn)
    order = np.argsort(-sims)[:k]
    out = []
    for idx in order:
        rec = data["records"][int(idx)]
        out.append({"source": rec["source"], "text": rec["text"],
                    "score": float(sims[int(idx)])})
    return out


# =========================================================================
# Chat
# =========================================================================
SYSTEM_PROMPT = (
    "You are PIA, Payton's personal AI assistant for the GENREG project. "
    "Answer questions about what the user is working on using ONLY the "
    "project documentation and changelog context provided below. The context "
    "is retrieved from the user's own docs and changelogs. If the answer is "
    "not in the context, say so plainly and do not invent details. Be "
    "concise and technical; cite the source file names you drew from."
)


def chat(message, history=None):
    """RAG chat turn. Returns {answer, sources} or {error}."""
    if not is_running():
        return {"error": "Ollama is not running. Start it from the PIA page."}
    try:
        hits = retrieve(message)
    except requests.RequestException as exc:
        return {"error": f"retrieval failed (is the index built?): {exc}"}
    if not hits and not _load_index():
        return {"error": "No index yet. Build the index on the PIA page first."}

    context_blocks = []
    for h in hits:
        context_blocks.append(f"### {h['source']}\n{h['text']}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no matches)"

    messages = [{"role": "system",
                 "content": SYSTEM_PROMPT + "\n\n=== CONTEXT ===\n" + context}]
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat",
                          json={"model": CHAT_MODEL, "messages": messages,
                                "stream": False},
                          timeout=300)
        r.raise_for_status()
        answer = r.json().get("message", {}).get("content", "").strip()
    except requests.RequestException as exc:
        return {"error": f"chat request failed: {exc}"}

    sources = sorted({h["source"] for h in hits})
    return {"answer": answer or "(empty response)", "sources": sources}


def status():
    """Full status snapshot for the PIA page (side-effect free)."""
    running = is_running()
    return {
        "installed": is_installed(),
        "running": running,
        "models_dir": MODELS_DIR,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "models": list_models() if running else [],
        "index": index_status(),
        "job": job_status(),
    }
