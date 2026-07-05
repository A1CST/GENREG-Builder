"""Web service around the Tree-of-Models skeleton (tree_lm.py).

tree_lm.py is the blueprint, kept byte-for-byte as specified. This module only
re-implements the *orchestration* — the bottom-up training walk and the GA loop
— so it can stream per-node / per-generation events over a WebSocket and honor
a stop flag. All model math (ContextEncoder, TreeNode, build_tree, inference,
serialization) is imported from tree_lm unchanged; the GA here mirrors
tree_lm._default_evolve exactly and plugs in through the documented
`evolve_fn` GENREG hook.

Corpus: samples context windows from the 50-book Project Gutenberg corpus at
project/EEC-main/engine/corpus.txt (~49 MB) so training sees real English
instead of the toy repeated sentence. Falls back to the demo string if the
corpus file is missing.
"""

import dataclasses
import datetime
import io
import hashlib
import json
import os
import re
import threading
import time

import numpy as np

from .tree_lm import (
    TreeConfig, ContextEncoder, TreeNode,
    build_tree, tree_predict_token, save_tree, load_tree, _entropy,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS_PATH = os.path.join(ROOT, "project", "EEC-main", "engine", "corpus.txt")
MODEL_DIR = os.path.join(ROOT, "runs", "tree")
ENC_DIR = os.path.join(ROOT, "runs", "encoder")   # standalone trained encoders

_corpus_cache = {"data": None}

# Last fully trained model, shared so `generate` works after training ends
# (and across WebSocket connections).
_model_lock = threading.Lock()
_model = {"root": None, "encoder": None, "config": None, "info": None,
          "run_id": None, "run_dir": None}


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
def load_corpus() -> tuple:
    """Return (byte_array, source_label). Cached after first read."""
    if _corpus_cache["data"] is not None:
        return _corpus_cache["data"]
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "rb") as fh:
            raw = fh.read()
        data = (np.frombuffer(raw, dtype=np.uint8).astype(int),
                f"gutenberg corpus ({len(raw):,} bytes, 50 books)")
    else:
        text = "the cat sat on the mat. the dog sat on the log. " * 400
        data = (np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(int),
                "fallback demo text (corpus.txt not found)")
    _corpus_cache["data"] = data
    return data


# --------------------------------------------------------------------------
# Word-level tokenization (token_mode="word")
# --------------------------------------------------------------------------
# words (apostrophes kept: "don't"), numbers, single punctuation marks
_WORD_RE = re.compile(r"[a-z']+|[0-9]+|[^\sa-z0-9']")


class WordVocab:
    """Frequency-built word/punctuation vocabulary; id 0 = <unk>."""

    _NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "”", "’", "'", '"', "…"}
    _NO_SPACE_AFTER = {"(", "[", "{", "“", "‘"}

    def __init__(self, words):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}

    @property
    def size(self):
        return len(self.words)

    def tokenize(self, text: str):
        return _WORD_RE.findall(text.lower())

    def encode_text(self, text: str):
        return [self.index.get(t, 0) for t in self.tokenize(text)]

    def decode(self, ids):
        out, prev = [], None
        for i in ids:
            w = self.words[i] if 0 <= int(i) < len(self.words) else "<unk>"
            if out and (w in self._NO_SPACE_BEFORE or prev in self._NO_SPACE_AFTER):
                out[-1] += w
            else:
                out.append(w)
            prev = w
        return " ".join(out)


_word_cache = {}   # vocab_size -> (ids array, source label, WordVocab)


def load_word_tokens(vocab_size: int):
    """Word-tokenize the corpus with a top-`vocab_size` vocabulary
    (id 0 = <unk>). Cached per vocab size — tokenization is a few seconds."""
    vocab_size = int(vocab_size)
    if vocab_size in _word_cache:
        return _word_cache[vocab_size]
    import collections
    raw, _src = load_corpus()
    # decode as UTF-8 and fold typographic punctuation to ASCII so curly
    # quotes/dashes don't shatter into junk multi-byte vocab entries
    text = bytes(raw.astype(np.uint8)).decode("utf-8", errors="replace").lower()
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("—", "-").replace("–", "-")
                .replace("…", "..."))
    text = re.sub(r"[^\x00-\x7f]", " ", text)   # any leftover non-ASCII → space
    toks = _WORD_RE.findall(text)
    counts = collections.Counter(toks)
    vocab = WordVocab(["<unk>"] + [w for w, _ in counts.most_common(vocab_size - 1)])
    idx = vocab.index
    ids = np.fromiter((idx.get(t, 0) for t in toks), dtype=np.int64, count=len(toks))
    coverage = float(np.mean(ids != 0))
    src = (f"word-level corpus: {len(ids):,} tokens · vocab {vocab.size:,} "
           f"· coverage {coverage * 100:.1f}% (<unk> covers the rest)")
    _word_cache[vocab_size] = (ids, src, vocab)
    return _word_cache[vocab_size]


_bigram_cache = {}


def bigram_prior(vocab_size: int):
    """Full-corpus row-normalized bigram table P(y|w) for word mode (cached).
    Column 0 (<unk>) is zeroed — it is never a prediction target."""
    vocab_size = int(vocab_size)
    if vocab_size in _bigram_cache:
        return _bigram_cache[vocab_size]
    ids, _, _ = load_word_tokens(vocab_size)
    prev, nxt = ids[:-1], ids[1:]
    m = nxt != 0
    C = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    np.add.at(C, (prev[m], nxt[m]), 1)
    P = (C / np.maximum(C.sum(1), 1)[:, None]).astype(np.float32)
    P[:, 0] = 0.0
    _bigram_cache[vocab_size] = P
    return P


def sample_windows(tokens: np.ndarray, n: int, context_window: int,
                   rng: np.random.Generator) -> tuple:
    """Draw n (context, next-byte) pairs from random offsets across the corpus.

    Equivalent to tree_lm.collect_training_data but samples positions instead
    of enumerating all of them — a 49 MB corpus has ~49M windows.
    """
    max_start = len(tokens) - context_window - 1
    n = min(n, max_start)
    starts = rng.choice(max_start, size=n, replace=False)
    X = tokens[starts[:, None] + np.arange(context_window)[None, :]]
    y = tokens[starts + context_window]
    return X, y


# --------------------------------------------------------------------------
# Model registry / generation
# --------------------------------------------------------------------------
def has_model() -> bool:
    with _model_lock:
        return _model["root"] is not None


def model_info():
    with _model_lock:
        return _model["info"]


def _register_model(root, encoder, config, info, run_id=None, run_dir=None,
                    vocab=None):
    with _model_lock:
        _model.update(root=root, encoder=encoder, config=config, info=info,
                      run_id=run_id, run_dir=run_dir, vocab=vocab)


def generate_text(prompt: str, length: int = 300, temperature: float = 0.0,
                  seed=None) -> str:
    """Autoregressive generation (bytes or words, per the trained model)."""
    with _model_lock:
        root, encoder, cfg = _model["root"], _model["encoder"], _model["config"]
        vocab = _model.get("vocab")
    if root is None:
        raise RuntimeError("no trained model — run training first")
    return _generate(root, encoder, cfg, prompt, length, temperature, seed,
                     vocab=vocab)


class DeepContextEncoder(ContextEncoder):
    """Two-stage evolved mixer: flat → tanh(W1·flat+b1) → tanh(W2·h+b2).
    Pure evolution (no gradients) — the second stage lets the context encode
    interactions between positions that a single linear mix cannot. Seeded
    near-identity in stage 2 so it starts exactly equal to the 1-stage form.

    Genome: [embeddings | W1 (rows×hidden) | b1 | W2 (hidden×context_dim) | b2]
    """

    def __init__(self, config, hidden=None):
        super().__init__(config)
        rows = config.context_window * config.embed_dim
        self.hidden = int(hidden or config.context_dim)
        self.mix_weights = np.random.randn(rows, self.hidden) * 0.01   # W1
        self.mix_bias = np.zeros(self.hidden)                          # b1
        self.w2 = np.random.randn(self.hidden, config.context_dim) * 0.01
        self.b2 = np.zeros(config.context_dim)

    def encode(self, token_ids):
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
        batch, cfg = token_ids.shape[0], self.config
        ctx_len = min(token_ids.shape[1], cfg.context_window)
        if ctx_len < cfg.context_window:
            padded = np.zeros((batch, cfg.context_window), dtype=int)
            padded[:, -ctx_len:] = token_ids[:, -ctx_len:]
            token_ids = padded
        embeds = self.embeddings[token_ids] + self.positions[np.newaxis]
        flat = embeds.reshape(batch, -1)
        h = np.tanh(flat @ self.mix_weights + self.mix_bias)
        return np.tanh(h @ self.w2 + self.b2)

    @property
    def genome_size(self):
        cfg = self.config
        rows = cfg.context_window * cfg.embed_dim
        return (cfg.vocab_size * cfg.embed_dim +
                rows * self.hidden + self.hidden +
                self.hidden * cfg.context_dim + cfg.context_dim)

    def pack_genome(self):
        return np.concatenate([self.embeddings.ravel(), self.mix_weights.ravel(),
                               self.mix_bias, self.w2.ravel(), self.b2])

    def unpack_genome(self, genome):
        cfg = self.config
        rows = cfg.context_window * cfg.embed_dim
        i = cfg.vocab_size * cfg.embed_dim
        self.embeddings = genome[:i].reshape(cfg.vocab_size, cfg.embed_dim)
        self.mix_weights = genome[i:i + rows * self.hidden].reshape(rows, self.hidden)
        i += rows * self.hidden
        self.mix_bias = genome[i:i + self.hidden]
        i += self.hidden
        self.w2 = genome[i:i + self.hidden * cfg.context_dim].reshape(self.hidden, cfg.context_dim)
        i += self.hidden * cfg.context_dim
        self.b2 = genome[i:]


def _save_model(root, encoder, path, vocab=None):
    """Blueprint save_tree + encoder architecture metadata, so deep encoders
    load back correctly. Mirrors tree_lm.save_tree field-for-field.
    Word-mode models also persist their vocabulary so replay can detokenize."""
    ec = encoder.config
    data = {"encoder_genome": encoder.pack_genome(),
            "encoder_kind": np.array([2 if isinstance(encoder, DeepContextEncoder) else 1]),
            "encoder_hidden": np.array([getattr(encoder, "hidden", 0)]),
            # the encoder's ACTUAL dims: with a reused saved encoder they differ
            # from the run config's sidebar dims, and loading with the wrong
            # dims makes unpack_genome's reshape blow up.
            "encoder_dims": np.array([ec.vocab_size, ec.embed_dim,
                                      ec.context_window, ec.context_dim])}
    if vocab is not None:
        data["vocab_blob"] = np.array("\x00".join(vocab.words))

    def _save_node(node, prefix):
        data[f"{prefix}_tokens"] = np.array(node.token_set)
        data[f"{prefix}_is_leaf"] = np.array([node.is_leaf])
        data[f"{prefix}_weights"] = node.weights
        data[f"{prefix}_bias"] = node.bias
        data[f"{prefix}_depth"] = np.array([node.depth])
        if not node.is_leaf:
            data[f"{prefix}_num_children"] = np.array([len(node.children)])
            for i, child in enumerate(node.children):
                _save_node(child, f"{prefix}_c{i}")

    _save_node(root, "root")
    np.savez_compressed(path, **data)


def _load_model(buf, cfg):
    """Counterpart of _save_model; reads blueprint-format files too (missing
    encoder_kind → plain ContextEncoder). If the file records the encoder's
    real dims (encoder_dims, saved since the reused-encoder fix), they replace
    cfg's — the run config keeps the SIDEBAR dims, which are wrong whenever
    the run reused a saved encoder."""
    data = np.load(buf, allow_pickle=True)
    if "encoder_dims" in data:
        dims = data["encoder_dims"]
        cfg.vocab_size, cfg.embed_dim = int(dims[0]), int(dims[1])
        cfg.context_window, cfg.context_dim = int(dims[2]), int(dims[3])
    kind = int(data["encoder_kind"][0]) if "encoder_kind" in data else 1
    if kind == 2:
        encoder = DeepContextEncoder(cfg, hidden=int(data["encoder_hidden"][0]) or None)
    else:
        encoder = ContextEncoder(cfg)
    encoder.unpack_genome(data["encoder_genome"])
    vocab = (WordVocab(str(data["vocab_blob"]).split("\x00"))
             if "vocab_blob" in data else None)

    def _load_node(prefix):
        node = TreeNode(prefix, data[f"{prefix}_tokens"].tolist(), cfg,
                        int(data[f"{prefix}_depth"][0]))
        node.weights = data[f"{prefix}_weights"]
        node.bias = data[f"{prefix}_bias"]
        node.is_leaf = bool(data[f"{prefix}_is_leaf"][0])
        node._frozen = True
        if not node.is_leaf:
            for i in range(int(data[f"{prefix}_num_children"][0])):
                node.children.append(_load_node(f"{prefix}_c{i}"))
        return node

    return _load_node("root"), encoder, vocab


def list_encoders():
    """Saved standalone encoders (runs/encoder/*), newest first."""
    out = []
    if not os.path.isdir(ENC_DIR):
        return out
    for rid in sorted(os.listdir(ENC_DIR), reverse=True):
        d = os.path.join(ENC_DIR, rid)
        try:
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError):
            continue
        if not os.path.exists(os.path.join(d, "encoder.npz")):
            continue
        enc = s.get("encoder") or {}
        out.append({
            "id": rid, "created": s.get("finished"), "status": s.get("status"),
            "nc_accuracy": enc.get("nc_accuracy"),
            "context_dim": enc.get("context_dim"),
            "embed_dim": enc.get("embed_dim"),
            "context_window": enc.get("context_window"),
            "generations": enc.get("generations"),
            "time_constrained": enc.get("time_constrained"),
            "diversity": enc.get("diversity"),
            "evolve_dims": enc.get("evolve_dims"),
        })
    return out


def load_encoder(enc_id: str):
    """Load a saved standalone encoder → frozen ContextEncoder."""
    d = os.path.join(ENC_DIR, enc_id)
    with open(os.path.join(d, "encoder.npz"), "rb") as fh:
        data = np.load(io.BytesIO(fh.read()))   # in-memory: no Windows lock
    dims = data["dims"]
    cfg = TreeConfig(vocab_size=int(dims[0]), embed_dim=int(dims[1]),
                     context_window=int(dims[2]), context_dim=int(dims[3]))
    kind = int(data["kind"][0]) if "kind" in data else 1
    if kind == 2:
        enc = DeepContextEncoder(cfg, hidden=int(data["hidden"][0]) or None)
    else:
        enc = ContextEncoder(cfg)
    enc.unpack_genome(data["genome"])
    enc.freeze()
    return enc


def _adopt_encoder_dims(cfg, cfg_dict):
    """Legacy-file fallback: model.npz files saved before encoder_dims existed
    need the same dim adoption the trainer did — copy the reused saved
    encoder's dims over the sidebar dims recorded in the run config."""
    enc_id = str((cfg_dict or {}).get("encoder_id") or "").strip()
    if not enc_id:
        return
    try:
        ec = load_encoder(enc_id).config
        cfg.embed_dim, cfg.context_window = ec.embed_dim, ec.context_window
        cfg.context_dim = ec.context_dim
    except Exception:
        pass


def embedding_cloud(run_dir: str, cfg_dict: dict):
    """PCA-3D projection of a run's byte-embedding table (vocab × embed_dim)
    for the runs-dashboard scatter — the collapse check. Works for tree runs
    (model.npz) and standalone encoder runs (encoder.npz). Returns 256 unit-
    scaled points + the variance spectrum diagnostics."""
    enc = None
    ep = os.path.join(run_dir, "encoder.npz")
    mp = os.path.join(run_dir, "model.npz")
    if os.path.exists(ep):
        enc = load_encoder(os.path.basename(run_dir))
    vocab = None
    if enc is None and os.path.exists(mp):
        cfg, _, _ = parse_config(cfg_dict or {})
        _adopt_encoder_dims(cfg, cfg_dict)
        with open(mp, "rb") as fh:
            buf = io.BytesIO(fh.read())
        _, enc, vocab = _load_model(buf, cfg)
    if enc is None:
        return None

    E = np.asarray(enc.embeddings, dtype=float)      # (V, embed_dim)
    Xc = E - E.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    tot = max(float(var.sum()), 1e-12)
    p = var / tot
    # effective rank = exp(spectral entropy): ~embed_dim when isotropic,
    # → 1 when the cloud collapses onto a line
    eff_rank = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-12)))))
    P = Xc @ Vt[:3].T
    scale = float(np.abs(P).max()) or 1.0
    P = P / scale
    norms = np.linalg.norm(Xc, axis=1)
    out = {
        "points": [{"b": int(i), "x": round(float(q[0]), 4),
                    "y": round(float(q[1]), 4), "z": round(float(q[2]), 4)}
                   for i, q in enumerate(P)],
        "explained": [round(float(e), 4) for e in p[:3]],
        "effective_rank": round(eff_rank, 2),
        "embed_dim": int(E.shape[1]),
        "mean_norm": round(float(norms.mean()), 4),
    }
    if vocab is not None:                 # word-mode run: points are WORDS
        out["labels"] = vocab.words
    return out


_top_words_cache = {"words": None}


def _top_corpus_words(n=180):
    """Most frequent words in the training corpus (cached). Single letters are
    dropped except the real words 'a' and 'i'."""
    if _top_words_cache["words"] is not None:
        return _top_words_cache["words"][:n]
    import collections
    import re
    tokens, _ = load_corpus()
    text = bytes(tokens.astype(np.uint8)).decode("latin-1").lower()
    counts = collections.Counter(re.findall(r"[a-z']+", text))
    words = [(w, c) for w, c in counts.most_common(n * 3)
             if len(w) >= 2 or w in ("a", "i")][:max(n, 1)]
    _top_words_cache["words"] = words
    return words[:n]


def word_cloud(run_dir: str, cfg_dict: dict, n_words: int = 180):
    """PCA-3D projection of WORD context vectors: the corpus's most frequent
    words, each encoded (as ' word', the encoder's state right after reading
    it) through the run's trained context encoder. Complements the byte
    embedding cloud — bytes show the letter table, this shows whether the
    encoder places related words near each other."""
    enc = None
    cfg = None
    vocab = None
    ep = os.path.join(run_dir, "encoder.npz")
    mp = os.path.join(run_dir, "model.npz")
    if os.path.exists(ep):
        enc = load_encoder(os.path.basename(run_dir))
        cfg = enc.config if enc is not None else None
    elif os.path.exists(mp):
        cfg, _, _ = parse_config(cfg_dict or {})
        _adopt_encoder_dims(cfg, cfg_dict)
        with open(mp, "rb") as fh:
            buf = io.BytesIO(fh.read())
        _, enc, vocab = _load_model(buf, cfg)
    if enc is None:
        return None
    window = int(getattr(cfg, "context_window", None)
                 or getattr(enc.config, "context_window", 16))

    words = _top_corpus_words(n_words)
    if not words:
        return None
    if vocab is not None:
        # word-mode model: a word IS one token — encode its id directly
        kept = [(w, c, vocab.index[w]) for w, c in words if w in vocab.index]
        if not kept:
            return None
        words = [(w, c) for w, c, _t in kept]
        X = np.zeros((len(kept), window), dtype=int)
        X[:, -1] = [t for _w, _c, t in kept]
    elif enc.config.vocab_size != 256:
        return None            # word-level encoder without a vocabulary — can't map words
    else:
        X = np.zeros((len(words), window), dtype=int)   # left-padded like encode()
        for i, (w, _c) in enumerate(words):
            bs = (" " + w).encode("utf-8")[-window:]
            X[i, window - len(bs):] = np.frombuffer(bs, dtype=np.uint8)
    V = np.asarray(enc.encode(X), dtype=float)      # (n_words, context_dim)

    Xc = V - V.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    tot = max(float(var.sum()), 1e-12)
    p = var / tot
    eff_rank = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-12)))))
    P = Xc @ Vt[:3].T
    scale = float(np.abs(P).max()) or 1.0
    P = P / scale
    return {
        "points": [{"w": w, "n": int(c), "x": round(float(q[0]), 4),
                    "y": round(float(q[1]), 4), "z": round(float(q[2]), 4)}
                   for (w, c), q in zip(words, P)],
        "explained": [round(float(e), 4) for e in p[:3]],
        "effective_rank": round(eff_rank, 2),
        "context_dim": int(V.shape[1]),
    }


def infer_run(model_path: str, cfg_dict: dict, prompt: str = "the ",
              length: int = 400, temperature: float = 0.8):
    """Replay hook for the /runs dashboard: load a saved tree run and
    generate a text sample from its frozen model."""
    import io
    cfg, _, _ = parse_config(cfg_dict)
    _adopt_encoder_dims(cfg, cfg_dict)
    # read into memory: np.load keeps the file handle open inside load_tree,
    # which locks model.npz on Windows for as long as the app lives
    with open(model_path, "rb") as fh:
        buf = io.BytesIO(fh.read())
    root, encoder, vocab = _load_model(buf, cfg)
    text = _generate(root, encoder, cfg, prompt, length, temperature, None,
                     vocab=vocab)
    return {"env": "tree", "prompt": prompt, "text": text,
            "temperature": temperature}


def trace_generate(prompt: str, length: int = 48, temperature: float = 0.8,
                   seed=None) -> dict:
    """Generate like `generate_text`, but record every decision on the way:
    per step, the score vector and chosen child at each router, plus the
    leaf's top candidates with probabilities. Feeds the routing inspector.
    """
    with _model_lock:
        root, encoder, cfg = _model["root"], _model["encoder"], _model["config"]
        run_id, run_dir = _model["run_id"], _model["run_dir"]
        vocab = _model.get("vocab")
    if root is None:
        raise RuntimeError("no trained model — run training first")

    length = max(1, min(int(length), 200))
    rng = np.random.default_rng(seed)
    tokens = ((vocab.encode_text(prompt) or [0]) if vocab is not None
              else (list(prompt.encode("utf-8")) or [ord(" ")]))
    n_prompt = len(tokens)
    steps = []

    for i in range(length):
        ctx_bytes = tokens[-cfg.context_window:]
        vec = encoder.encode(np.array(ctx_bytes, dtype=int))
        node, path = root, []
        while not node.is_leaf:
            scores = (vec @ node.weights + node.bias)[0]
            c = int(np.argmax(scores))
            path.append({
                "id": node.node_id, "depth": node.depth, "chosen": c,
                "scores": [round(float(s), 4) for s in scores],
                "children": [{"t0": min(ch.token_set), "t1": max(ch.token_set) + 1,
                              "tokens": len(ch.token_set),
                              "sample": _token_sample(ch.token_set, 8, vocab)}
                             for ch in node.children],
            })
            node = node.children[c]

        local_idx, raw = node.predict(vec)
        s = raw[0].astype(float)
        z = s - s.max()
        if temperature > 0 and len(node.token_set) > 1:
            p = np.exp(z / max(temperature, 1e-6))
            p /= p.sum()
            choice = int(rng.choice(len(p), p=p))
        else:
            p = np.exp(z)
            p /= p.sum()
            choice = int(local_idx[0])
        byte = node.token_set[min(choice, len(node.token_set) - 1)]

        order = np.argsort(s)[::-1][:8]
        steps.append({
            "i": i,
            "context": (vocab.decode(ctx_bytes) if vocab is not None
                        else bytes(ctx_bytes).decode("utf-8", errors="replace")),
            "byte": int(byte),
            "token": (vocab.words[byte] if vocab is not None
                      and 0 <= byte < len(vocab.words) else None),
            "path": path,
            "leaf": {
                "id": node.node_id, "t0": min(node.token_set),
                "t1": max(node.token_set) + 1, "tokens": len(node.token_set),
                "sample": _token_sample(node.token_set, 8, vocab),
                "chosen_byte": int(byte),
                "top": [{"byte": int(node.token_set[j]),
                         "score": round(float(s[j]), 4),
                         "prob": round(float(p[j]), 4),
                         **({"token": vocab.words[int(node.token_set[j])]}
                            if vocab is not None else {})} for j in order],
            },
        })
        tokens.append(byte)

    text = (vocab.decode(tokens[n_prompt:]) if vocab is not None
            else bytes(tokens[n_prompt:]).decode("utf-8", errors="replace"))
    result = {"prompt": prompt, "text": text, "temperature": temperature,
              "steps": steps, "run_id": run_id,
              "created": datetime.datetime.now().isoformat(timespec="seconds")}

    # persist alongside the run so /runs can replay the inspector later
    if run_dir and os.path.isdir(run_dir):
        try:
            with open(os.path.join(run_dir, "traces.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            result["saved"] = True
        except OSError:
            result["saved"] = False
    else:
        result["saved"] = False
    return result


# -- UTF-8-aware sampling ----------------------------------------------------
# Generation is per-byte but the corpus is UTF-8, so an unconstrained sampler
# emits stray 0x80–0xFF bytes that never form a valid sequence and decode to
# U+FFFD (the "� boxes"). These helpers constrain each step to bytes that keep
# the output stream valid UTF-8. Model/routing behavior is unchanged — only
# invalid byte choices are masked out at the leaf.

def _utf8_state(tokens):
    """Inspect the tail of `tokens` for an incomplete UTF-8 sequence.
    Returns (n_partial, lo, hi): how many bytes of an unfinished multi-byte
    sequence were already emitted (0 = none pending) and the inclusive value
    range valid for the sequence's next byte (meaningless when n_partial=0)."""
    n = len(tokens)
    for back in (1, 2, 3):                 # a lead byte can sit at most 3 back
        if back > n:
            break
        b = tokens[n - back]
        if b < 0x80:                       # ASCII — nothing can be pending
            return 0, 0, 0
        if b >= 0xC0:                      # lead byte, `back` bytes from the end
            need = 2 if b < 0xE0 else 3 if b < 0xF0 else 4
            if back >= need or b < 0xC2 or b > 0xF4:
                return 0, 0, 0             # complete, or an invalid lead: ignore
            if back == 1:                  # next is byte 2: lead-specific ranges
                if b == 0xE0:
                    return back, 0xA0, 0xBF
                if b == 0xED:
                    return back, 0x80, 0x9F
                if b == 0xF0:
                    return back, 0x90, 0xBF
                if b == 0xF4:
                    return back, 0x80, 0x8F
            return back, 0x80, 0xBF
        # else 0x80–0xBF continuation — keep walking back toward the lead
    return 0, 0, 0                         # orphan continuations: nothing pending


def _utf8_allowed(tokens):
    """Boolean (256,) mask of next bytes that keep the stream valid UTF-8."""
    pend, lo, hi = _utf8_state(tokens)
    mask = np.zeros(256, dtype=bool)
    if pend:
        mask[lo:hi + 1] = True             # must continue the open sequence
    else:
        mask[0x00:0x80] = True             # ASCII
        mask[0xC2:0xF5] = True             # valid 2/3/4-byte lead bytes
    return mask


def _pick_byte(node, scores, local_idx, tokens, n_prompt, temperature, rng):
    """Choose the next byte from the leaf, constrained to valid UTF-8.
    If the leaf can't continue an in-progress multi-byte sequence, the
    unfinishable partial sequence is abandoned (popped off `tokens`, never
    into the prompt) and the pick retried; a leaf with no valid byte at all
    falls back to the old unmasked pick (rare — default leaves hold 64
    tokens, virtually always including ASCII)."""
    tok = node.token_set
    s = scores[0].astype(float)

    for _attempt in (0, 1):
        allowed = _utf8_allowed(tokens)[np.asarray(tok)]
        if allowed.any():
            if temperature > 0 and int(allowed.sum()) > 1:
                z = s / max(temperature, 1e-6)
                z = z - z[allowed].max()
                p = np.where(allowed, np.exp(z), 0.0)
                p /= p.sum()
                choice = int(rng.choice(len(p), p=p))
            else:
                choice = int(np.argmax(np.where(allowed, s, -np.inf)))
            return int(tok[choice])
        pend = _utf8_state(tokens)[0]
        cut = max(n_prompt, len(tokens) - pend)
        if not pend or cut >= len(tokens):
            break
        del tokens[cut:]                   # abandon the unfinishable sequence
    # no UTF-8-valid byte in this leaf — legacy unmasked behavior
    if temperature > 0 and len(tok) > 1:
        z = s / max(temperature, 1e-6)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(tok[int(rng.choice(len(p), p=p))])
    return int(tok[min(int(local_idx[0]), len(tok) - 1)])


def _generate_words(root, encoder, cfg, prompt, length, temperature, seed, vocab):
    """Word-mode generation: one routed prediction per WORD token; output is
    detokenized via the model's vocabulary (no UTF-8 masking needed — every
    token is a whole word/punctuation mark)."""
    length = max(1, min(int(length), 2000))
    rng = np.random.default_rng(seed)
    tokens = vocab.encode_text(prompt) or [0]
    n_prompt = len(tokens)
    for _ in range(length):
        ctx = np.array(tokens[-cfg.context_window:], dtype=int)
        vec = encoder.encode(ctx)
        node = root
        while not node.is_leaf:
            node = node.children[int(node.route(vec)[0])]
        local_idx, scores = node.predict(vec)
        tok = node.token_set
        s = scores[0].astype(float)
        if len(tok) > 1 and 0 in tok:
            s[list(tok).index(0)] = -np.inf   # never EMIT <unk>
        if temperature > 0 and len(tok) > 1:
            z = s / max(temperature, 1e-6)
            z -= z[np.isfinite(z)].max()
            p = np.where(np.isfinite(z), np.exp(np.where(np.isfinite(z), z, 0.0)), 0.0)
            p /= p.sum()
            choice = int(rng.choice(len(p), p=p))
        else:
            choice = int(np.argmax(s))
        tokens.append(int(tok[min(choice, len(tok) - 1)]))
    return vocab.decode(tokens[n_prompt:])


def _generate(root, encoder, cfg, prompt: str, length: int,
              temperature: float, seed, vocab=None) -> str:
    """Greedy routing through the tree (as in tree_lm inference); at the leaf,
    temperature > 0 samples from softmax(scores / T) over the leaf's token
    subset instead of argmax — pure argmax loops quickly on repeated n-grams.
    Byte mode masks sampling to valid UTF-8; word mode routes per word.
    """
    if vocab is not None:
        return _generate_words(root, encoder, cfg, prompt, length,
                               temperature, seed, vocab)
    length = max(1, min(int(length), 2000))
    rng = np.random.default_rng(seed)
    tokens = list(prompt.encode("utf-8")) or [ord(" ")]
    n_prompt = len(tokens)

    for _ in range(length):
        ctx = np.array(tokens[-cfg.context_window:], dtype=int)
        vec = encoder.encode(ctx)          # (1, context_dim)
        node = root
        while not node.is_leaf:
            node = node.children[int(node.route(vec)[0])]
        local_idx, scores = node.predict(vec)
        tokens.append(_pick_byte(node, scores, local_idx, tokens,
                                 n_prompt, temperature, rng))

    out = bytes(tokens[n_prompt:])
    return out.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Config parsing (clamped so a browser can't wedge the server)
# --------------------------------------------------------------------------
def _token_sample(tokens, k=12, vocab=None):
    """Short printable preview of a token set, e.g. '␣ e t a …' (byte mode)
    or 'the and of …' (word mode)."""
    out = []
    for t in tokens[:k]:
        if vocab is not None:
            out.append(vocab.words[t] if 0 <= t < len(vocab.words) else f"#{t}")
        elif t == 32:
            out.append("␣")
        elif 32 < t < 127:
            out.append(chr(t))
        else:
            out.append(f"\\x{t:02x}")
    return " ".join(out) + (" …" if len(tokens) > k else "")


def _num(msg, key, default, lo, hi, cast=int):
    try:
        v = cast(msg.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def parse_config(msg: dict) -> tuple:
    # token level: "byte" (vocab 256, the original) or "word" (top-K word
    # vocabulary — real semantic units, readable generation)
    token_mode = "word" if str(msg.get("token_mode", "byte")).lower() == "word" else "byte"
    vocab = 256 if token_mode == "byte" else _num(msg, "vocab_size", 2048, 256, 8192)
    bf = _num(msg, "branching_factor", 4, 2, 64)
    # routing_layers is the direct depth lever: leaf size = vocab / bf^layers.
    # 0 layers → min_leaf = vocab → the root itself is one flat specialist over
    # the whole vocabulary (no routers). Falls back to explicit min_leaf_tokens.
    if msg.get("routing_layers") is not None:
        layers = _num(msg, "routing_layers", 2, 0, 8)
        min_leaf = vocab if layers == 0 else max(1, -(-vocab // (bf ** layers)))
    else:
        min_leaf = _num(msg, "min_leaf_tokens", 16, 1, vocab)
    cfg = TreeConfig(
        vocab_size=vocab,
        branching_factor=bf,
        context_window=_num(msg, "context_window", 16, 2, 64),
        embed_dim=_num(msg, "embed_dim", 32, 4, 256),
        context_dim=_num(msg, "context_dim", 64, 8, 256),
        min_leaf_tokens=min_leaf,
        max_depth=_num(msg, "max_depth", 8, 1, 10),
        pop_size=_num(msg, "pop_size", 60, 8, 300),
        generations=_num(msg, "generations", 60, 5, 1000),
        mutation_rate=_num(msg, "mutation_rate", 0.15, 0.0, 1.0, float),
        elite_frac=_num(msg, "elite_frac", 0.2, 0.05, 0.9, float),
    )
    max_samples = _num(msg, "max_samples", 16000, 500, 100000)
    seed = msg.get("seed")
    seed = int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None
    return cfg, max_samples, seed


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------
class TreeLMTrainer:
    """Runs one full build → train → evaluate cycle, streaming events."""

    def __init__(self, msg: dict, emit):
        self.cfg, self.max_samples, self.seed = parse_config(msg)
        # token level (must match parse_config's reading of the same key)
        self.token_mode = ("word" if str(msg.get("token_mode", "byte")).lower() == "word"
                           else "byte")
        self._vocab = None                  # WordVocab in word mode
        # encoder pre-evolution (blueprint step 1); 0 generations = skip
        self.encoder_generations = _num(msg, "encoder_generations", 40, 0, 2000)
        self.encoder_samples = _num(msg, "encoder_samples", 2000, 200, 20000)
        # encoder GA population; 0 = inherit the shared pop_size (the encoder
        # genome is much larger than a node's, so it may want its own size)
        self.encoder_pop_size = _num(msg, "encoder_pop_size", 0, 0, 1000)
        # phase 2: continue evolving the champion under a time/Occam constraint
        # (fitness × 1/(1 + active_weights/budget)); 0 = off
        self.encoder_speed_generations = _num(msg, "encoder_speed_generations", 40, 0, 500)
        self.encoder_time_budget = _num(msg, "encoder_time_budget", 0.5, 0.05, 10.0, float)
        # alternative to the two-phase form: evolve the encoder from scratch
        # with the time/Occam constraint folded into its original fitness for
        # ALL encoder_generations (accepts true/false or 0/1 from sweeps)
        self.encoder_time_constrained = bool(msg.get("encoder_time_constrained"))
        # diversity constraint: push the encoder's output dims ("heads") to be
        # different — fitness ÷ (1 + mean|corr between dims|/budget)
        self.encoder_diversity = bool(msg.get("encoder_diversity"))
        self.encoder_diversity_budget = _num(msg, "encoder_diversity_budget", 0.5, 0.05, 10.0, float)
        # population novelty bonus: genomes that differ from the rest of the
        # population get a fitness BOOST at selection time (fitness sharing) —
        # distinct from head diversity, which is within-genome
        self.encoder_novelty = bool(msg.get("encoder_novelty"))
        self.encoder_novelty_strength = _num(msg, "encoder_novelty_strength", 0.5, 0.05, 5.0, float)
        # seed the embedding table from byte co-occurrence geometry (PCA of
        # context profiles) instead of random init — the embedding half of
        # the genome is otherwise a neutral plateau evolution never shapes
        self.encoder_seed_embeddings = bool(msg.get("encoder_seed_embeddings"))
        # node GA options: seed with the closed-form ridge (least-squares)
        # solution alongside nearest-centroid, and/or use a smooth fitness
        # (mean softmax prob of the target) instead of stepwise 0/1 accuracy
        self.ridge_seed = bool(msg.get("ridge_seed"))
        self.node_fitness = "prob" if str(msg.get("node_fitness", "acc")).lower() == "prob" else "acc"
        # encoder fitness: "nc" (nearest-centroid proxy) or "ridge" — held-out
        # accuracy of a closed-form ridge readout, aligned with the
        # ridge-seeded linear leaves that consume the encoding downstream
        # "nc" nearest-centroid | "ridge" held-out ridge readout | "residual"
        # (word mode): held-out accuracy of COUNTS + ridge head fitted to the
        # residual onehot(y) − P_bigram(y|w) — the encoder is only rewarded
        # for encoding what the count table does NOT already know.
        _ef = str(msg.get("encoder_fitness", "nc")).lower()
        self.encoder_fitness = _ef if _ef in ("ridge", "residual") else "nc"
        # non-stationarity constraint (EEC): rotate the node fitness sample
        # each generation so evolution can't win by memorizing one sample
        self.node_resample = bool(msg.get("node_resample"))
        # 2-stage evolved encoder (DeepContextEncoder); hidden 0 = context_dim
        self.encoder_depth = 2 if _num(msg, "encoder_depth", 1, 1, 2) == 2 else 1
        self.encoder_hidden = _num(msg, "encoder_hidden", 0, 0, 2048)
        # self-adaptive mutation scale (ES style, like the main engine): each
        # individual carries its own step size, inherited + log-perturbed
        self.sa_mutation = bool(msg.get("sa_mutation"))
        # rotate the ridge fit/val split each generation so encoder selection
        # can't slowly overfit a fixed validation half
        self.encoder_split_rotate = bool(msg.get("encoder_split_rotate"))
        # evolve the encoder's output dimension count (context_dim) instead of
        # keeping it fixed: mutation can grow/shrink the mixer, start = the
        # sidebar Context dim, bounds [4, 4×start]. Off = fixed (default).
        self.encoder_evolve_dims = bool(msg.get("encoder_evolve_dims"))
        # evolve the EMBEDDING dimension (grow/shrink embedding columns and
        # their mixer rows) — the word-level campaign showed embed dim is the
        # binding constraint. CPU path (ragged genomes), shallow encoder only.
        self.encoder_evolve_embed = bool(msg.get("encoder_evolve_embed"))
        # cluster the vocabulary by context co-occurrence instead of splitting
        # into sequential byte-ID ranges — 'e', 't' and space land in different
        # branches because their preceding-context profiles differ
        self.cluster_tokens = bool(msg.get("cluster_tokens"))
        self._byte_F = None                 # context-profile features per byte
        # compute device: "gpu" batches whole-population fitness evaluation
        # on CUDA (torch); "cpu" keeps the numpy per-individual loop
        self.device = "gpu" if str(msg.get("device", "cpu")).lower() == "gpu" else "cpu"
        self._torch = None                  # resolved at run start
        # reuse a saved standalone encoder (skips encoder evolution; its dims
        # override the sidebar embed/window/context dims)
        self.encoder_id = str(msg.get("encoder_id") or "").strip() or None
        self.routing_layers = (_num(msg, "routing_layers", 2, 0, 8)
                               if msg.get("routing_layers") is not None else None)
        self.notes = str(msg.get("notes") or "tree-of-models text prediction")
        self.emit = emit
        self._stop = threading.Event()
        self.run_id = None
        self.run_dir = None
        self._acc_sum, self._acc_n = 0.0, 0
        # encoder-training record for the runs dashboard: per-gen fitness
        # curves keyed by stage id ("encoder"/"encoder-speed") + final metrics
        self._enc_curves = {}
        self._enc_info = {}

    # -- run persistence (runstore layout: runs/tree/<id>/…) -----------------
    def _cfg_dict(self):
        c = self.cfg
        return {
            "environment": "tree",
            "token_mode": self.token_mode, "vocab_size": c.vocab_size,
            "branching_factor": c.branching_factor, "context_window": c.context_window,
            "embed_dim": c.embed_dim, "context_dim": c.context_dim,
            "min_leaf_tokens": c.min_leaf_tokens, "max_depth": c.max_depth,
            "routing_layers": self.routing_layers,
            "pop_size": c.pop_size, "generations": c.generations,
            "mutation_rate": c.mutation_rate, "elite_frac": c.elite_frac,
            "max_samples": self.max_samples, "seed": self.seed,
            "encoder_generations": self.encoder_generations,
            "encoder_samples": self.encoder_samples,
            "encoder_pop_size": self.encoder_pop_size,
            "encoder_speed_generations": self.encoder_speed_generations,
            "encoder_time_budget": self.encoder_time_budget,
            "encoder_time_constrained": self.encoder_time_constrained,
            "encoder_diversity": self.encoder_diversity,
            "encoder_diversity_budget": self.encoder_diversity_budget,
            "encoder_novelty": self.encoder_novelty,
            "encoder_novelty_strength": self.encoder_novelty_strength,
            "encoder_seed_embeddings": self.encoder_seed_embeddings,
            "ridge_seed": self.ridge_seed,
            "node_fitness": self.node_fitness,
            "encoder_fitness": self.encoder_fitness,
            "node_resample": self.node_resample,
            "encoder_depth": self.encoder_depth,
            "encoder_hidden": self.encoder_hidden,
            "sa_mutation": self.sa_mutation,
            "encoder_split_rotate": self.encoder_split_rotate,
            "encoder_evolve_dims": self.encoder_evolve_dims,
            "encoder_evolve_embed": self.encoder_evolve_embed,
            "cluster_tokens": self.cluster_tokens,
            "device": self.device,
            "encoder_id": self.encoder_id,
            # aliases the runs dashboard reads directly
            "population": c.pop_size,
        }

    def _persist_create(self):
        ts = datetime.datetime.now()
        cfg_dict = self._cfg_dict()
        h = hashlib.sha1(json.dumps(cfg_dict, sort_keys=True, default=str)
                         .encode()).hexdigest()[:6]
        self.run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}-tree-{h}"
        self.run_dir = os.path.join(MODEL_DIR, self.run_id)
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"id": self.run_id, "environment": "tree",
                           "created": ts.isoformat(timespec="seconds"),
                           "config": cfg_dict,
                           "started": {"population": cfg_dict["pop_size"],
                                       "generations": cfg_dict["generations"],
                                       "notes": self.notes},
                           "status": "running"}, f, indent=2)
            open(os.path.join(self.run_dir, "history.jsonl"), "w").close()
        except OSError:
            self.run_dir = None

    def _persist_history(self, node_id, kind, acc):
        if self.run_dir is None or acc is None:
            return
        self._acc_sum += acc
        self._acc_n += 1
        rec = {"gen": self._done, "node": node_id, "kind": kind,
               "fitness": {"best": round(acc, 4),
                           "mean": round(self._acc_sum / self._acc_n, 4)}}
        try:
            with open(os.path.join(self.run_dir, "history.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _persist_finalize(self, status, info=None, root=None, encoder=None):
        if self.run_dir is None:
            return None
        ckpt = None
        if root is not None:
            try:
                _save_model(root, encoder, os.path.join(self.run_dir, "model.npz"),
                            vocab=self._vocab)
                ckpt = "model.npz"
            except Exception:
                ckpt = None
        best = None
        if info:
            best = {"score": round(info["accuracy"], 4),
                    "base": round(info["bigram_accuracy"], 4)}
        summary = {
            "id": self.run_id, "environment": "tree", "status": status,
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
            "gen": self._done, "best": best, "checkpoint": ckpt,
            "eval": info,
        }
        if self._enc_info:                # encoder-training record (runs panel)
            summary["encoder"] = {**self._enc_info, "curves": self._enc_curves}
        try:
            with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            cpath = os.path.join(self.run_dir, "config.json")
            with open(cpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = status
            # dim evolution can change context_dim mid-run — persist the final
            # value so model.npz can be loaded back (replay / embedding cloud)
            meta.setdefault("config", {})["context_dim"] = self.cfg.context_dim
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except (OSError, ValueError):
            pass
        return os.path.join(self.run_dir, ckpt) if ckpt else None

    def stop(self):
        self._stop.set()

    @property
    def stopped(self):
        return self._stop.is_set()

    def _make_encoder(self):
        if self.encoder_depth == 2:
            return DeepContextEncoder(self.cfg,
                                      hidden=self.encoder_hidden or self.cfg.context_dim)
        return ContextEncoder(self.cfg)

    def _resolve_device(self):
        self._torch = None
        if self.device != "gpu":
            return
        try:
            import torch
            if torch.cuda.is_available():
                self._torch = torch
                self.emit({"type": "status",
                           "message": (f"compute: GPU "
                                       f"({torch.cuda.get_device_name(0)}) — "
                                       "batched population evaluation")})
            else:
                self.emit({"type": "status",
                           "message": "GPU requested but CUDA is unavailable — using CPU"})
        except Exception as exc:
            self.emit({"type": "status",
                       "message": f"GPU unavailable ({exc}) — using CPU"})

    # -- clustered token split (alternative to sequential byte ranges) ------
    def _byte_context_features(self, X, y):
        """Context-profile features per byte: the distribution of the byte(s)
        that precede it in the training data, Hellinger-scaled (sqrt of
        probabilities, so common and rare predecessors both contribute).
        Bytes that appear in similar contexts get similar rows."""
        V = self.cfg.vocab_size
        cols = []
        for k in (1, 2):
            if X.shape[1] < k:
                continue
            M = np.zeros((V, V))
            np.add.at(M, (y, X[:, -k]), 1)
            M = np.sqrt(M / np.maximum(M.sum(axis=1, keepdims=True), 1.0))
            cols.append(M)
        return np.hstack(cols) if cols else np.zeros((V, 1))

    def _balanced_kmeans(self, F, k, rng):
        """k-means then capacity-balanced assignment (most-confident tokens
        pick first), so group sizes match the sequential build's chunks and
        the routing_layers → leaf-size math still holds."""
        n = len(F)
        cent = F[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(10):
            D = ((F[:, None, :] - cent[None]) ** 2).sum(-1)     # (n, k)
            lab = D.argmin(1)
            for c in range(k):
                m = lab == c
                if m.any():
                    cent[c] = F[m].mean(0)
        D = ((F[:, None, :] - cent[None]) ** 2).sum(-1)
        cap = np.full(k, n // k)
        cap[: n % k] += 1
        groups = [[] for _ in range(k)]
        margin = (np.partition(D, 1, axis=1)[:, 1] - D.min(1)) if k > 1 else np.zeros(n)
        for i in np.argsort(-margin):
            for c in np.argsort(D[i]):
                if cap[c] > 0:
                    groups[c].append(int(i))
                    cap[c] -= 1
                    break
        return [g for g in groups if g]

    def _build_root(self):
        """Build the (untrained) routing tree: sequential byte ranges by
        default, or recursive balanced k-means over context-profile features
        when cluster_tokens is on. Deterministic per run seed, so a rebuild
        (dim evolution) reproduces the same partition."""
        cfg = self.cfg
        if not self.cluster_tokens or self._byte_F is None:
            return build_tree(cfg)
        F, rng = self._byte_F, np.random.default_rng(self.seed or 0)

        def build(tokens, depth, path):
            node = TreeNode(path, tokens, cfg, depth)
            if len(tokens) <= cfg.min_leaf_tokens or depth >= cfg.max_depth:
                node.init_as_leaf(max(len(tokens), 1))
                return node
            bf = min(cfg.branching_factor, len(tokens))
            groups = self._balanced_kmeans(F[tokens], bf, rng)
            if len(groups) <= 1:
                node.init_as_leaf(len(tokens))
                return node
            node.init_as_router(len(groups))
            for i, g in enumerate(groups):
                node.children.append(
                    build([tokens[j] for j in g], depth + 1, f"{path}.{i}"))
            return node

        return build(list(range(cfg.vocab_size)), 0, "R")

    # -- structure ---------------------------------------------------------
    def _tree_manifest(self, root, encoder):
        """Node list for the icicle. t0/t1 are DFS *positions* — identical to
        byte values for the sequential build, and a contiguous layout axis
        for clustered trees (whose token sets aren't ID ranges). `sample`
        previews the actual tokens for tooltips."""
        nodes, total_params = [], encoder.genome_size
        pos = [0]

        def walk(node, parent):
            nonlocal total_params
            total_params += node.genome_size
            entry = {
                "id": node.node_id, "parent": parent, "depth": node.depth,
                "leaf": node.is_leaf, "tokens": len(node.token_set),
                "sample": _token_sample(node.token_set),
            }
            nodes.append(entry)
            if node.is_leaf:
                entry["t0"] = pos[0]
                pos[0] += len(node.token_set)
                entry["t1"] = pos[0]
            else:
                entry["t0"] = pos[0]
                for c in node.children:
                    walk(c, node.node_id)
                entry["t1"] = pos[0]

        walk(root, None)
        return nodes, total_params

    # -- GA (mirrors tree_lm._default_evolve, + events, stop, seeding) ------
    def _evolve(self, genome_size, fitness_fn, node_id,
                generations=None, seed_pop=None, pop_size=None,
                batch_fitness=None, novelty=0.0):
        cfg, rng = self.cfg, self._rng
        generations = generations if generations is not None else cfg.generations
        pop_size = pop_size if pop_size else cfg.pop_size
        stride = max(1, generations // 40)
        pop = rng.standard_normal((pop_size, genome_size)) * 0.1
        if seed_pop is not None:                 # inject structured candidates
            pop[:min(len(seed_pop), pop_size)] = seed_pop[:pop_size]
        best_score, best_ind = -1.0, pop[0].copy()
        top_k = max(2, int(pop_size * cfg.elite_frac))
        ms = np.full(len(pop), 0.1)              # per-individual mutation scale
        best_ms = 0.1

        for gen in range(generations):
            if self.stopped:
                break
            scores = (np.asarray(batch_fitness(pop), dtype=float)
                      if batch_fitness is not None
                      else np.array([fitness_fn(ind) for ind in pop]))
            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                best_score, best_ind = float(scores[idx]), pop[idx].copy()
                best_ms = float(ms[idx])

            # population novelty bonus: genomes that differ from the rest get
            # a fitness BOOST at selection time (the champion above is still
            # tracked on raw fitness — novelty shapes who gets to breed)
            sel_scores = scores
            if novelty > 0 and len(pop) > 1:
                sub = pop[:, ::max(1, genome_size // 512)]   # gene subsample
                dist = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).mean(-1))
                nov = dist.mean(1)
                nov = nov / max(float(nov.max()), 1e-12)     # 0..1
                sel_scores = scores * (1.0 + novelty * nov)

            if gen % stride == 0 or gen == generations - 1:
                self.emit({"type": "node_gen", "id": node_id, "gen": gen,
                           "best": best_score, "mean": float(scores.mean())})
                if node_id.startswith("encoder"):   # persist for the runs panel
                    self._enc_curves.setdefault(node_id, []).append(
                        {"gen": gen, "best": round(best_score, 4),
                         "mean": round(float(scores.mean()), 4)})

            order = np.argsort(sel_scores)[-top_k:]
            elites, elite_ms = pop[order], ms[order]
            new_pop, new_ms = [best_ind.copy()], [best_ms]
            for _ in range(pop_size - 1):
                i1, i2 = rng.choice(len(elites), 2, replace=False)
                mask = rng.random(genome_size) > 0.5
                child = np.where(mask, elites[i1], elites[i2])
                cms = 0.1
                if self.sa_mutation:
                    # ES self-adaptation: inherit a parent's step size and
                    # log-perturb it — step size evolves with the solution
                    cms = float(np.clip(
                        elite_ms[i1 if rng.random() < 0.5 else i2] *
                        np.exp(0.3 * rng.standard_normal()), 0.005, 0.5))
                if rng.random() < cfg.mutation_rate:
                    scale = np.abs(child) * cms + 0.1 * cms   # relative — GENREG style
                    child = child + rng.standard_normal(genome_size) * scale
                new_pop.append(child)
                new_ms.append(cms)
            pop, ms = np.array(new_pop), np.array(new_ms)

        return best_ind

    # -- GPU-batched fitness (device == "gpu") -------------------------------
    def _batch_linear_acc(self, ctx, labels, k, mode="acc"):
        """Whole-population fitness for a router/leaf on CUDA: genomes
        (P, dim·k + k) → accuracy (P,) in one batched einsum. Returns None
        when GPU compute is off, so callers fall back to the scalar loop."""
        t = self._torch
        if t is None:
            return None
        dim = ctx.shape[1]
        dev = "cuda"
        ctx_t = t.tensor(ctx, dtype=t.float32, device=dev)
        lab_t = t.tensor(np.asarray(labels), dtype=t.long, device=dev)
        n = len(labels)
        # non-stationarity: bf is called once per generation, so a stateful
        # closure that rotates a common half-sample each call gives every
        # generation a fresh (but within-generation fair) fitness landscape
        resample = self.node_resample and n > 4000
        state = {"gen": 0}

        def bf(pop):
            if resample:
                rs = np.random.default_rng(((self.seed or 0) + 1) * 7919 + state["gen"])
                state["gen"] += 1
                sel = t.tensor(rs.choice(n, size=n // 2, replace=False), device=dev)
                ctx_use, lab_use = ctx_t[sel], lab_t[sel]
            else:
                ctx_use, lab_use = ctx_t, lab_t
            n_use = ctx_use.shape[0]
            chunk = max(1, int(6.4e7 // max(n_use * k, 1)))   # ≤ ~256 MB of scores
            g = np.asarray(pop, dtype=np.float32)
            out = np.empty(len(g))
            with t.no_grad():
                for i in range(0, len(g), chunk):
                    gt = t.tensor(g[i:i + chunk], device=dev)
                    W = gt[:, :dim * k].reshape(-1, dim, k)
                    b = gt[:, dim * k:]
                    scores = t.einsum("nd,pdk->pnk", ctx_use, W) + b[:, None, :]
                    if mode == "prob":
                        # GEOMETRIC mean prob of the target (inverse
                        # perplexity): smooth like mean-prob but a confident
                        # constant predictor scores ~0 instead of winning
                        lp = t.log_softmax(scores, dim=-1)
                        tgt = lp.gather(-1, lab_use[None, :, None]
                                        .expand(scores.shape[0], -1, -1)).squeeze(-1)
                        val = t.exp(tgt.clamp(min=-20.0).mean(1))
                    else:
                        val = (scores.argmax(-1) == lab_use[None]).float().mean(1)
                    out[i:i + chunk] = val.cpu().numpy()
            return out
        return bf

    def _batch_encoder_fitness(self, Xs, yi, counts, n_classes, positions,
                               use_time=False, use_div=False, fitness_mode="nc"):
        """Whole-population encoder fitness on CUDA (fixed context_dim):
        replicates ContextEncoder.encode (embed lookup + positions → mix →
        tanh) and the nearest-centroid accuracy, batched over genomes, with
        the time/diversity penalties applied when enabled. Returns None when
        GPU compute is off or dims are being evolved (ragged genomes)."""
        t = self._torch
        if t is None:
            return None
        cfg = self.cfg
        V, E, d = cfg.vocab_size, cfg.embed_dim, cfg.context_dim
        win = cfg.context_window
        emb_len, rows = V * E, win * E
        deep = self.encoder_depth == 2
        hidden = (self.encoder_hidden or d) if deep else 0
        n = len(yi)
        dev = "cuda"
        Xs_t = t.tensor(np.asarray(Xs), dtype=t.long, device=dev)      # (n, win)
        yi_t = t.tensor(np.asarray(yi), dtype=t.long, device=dev)
        cnt_t = t.tensor(np.asarray(counts), dtype=t.float32, device=dev)  # (k, 1)
        pos_t = t.tensor(positions, dtype=t.float32, device=dev)       # (win, E)
        budget, div_budget = self.encoder_time_budget, self.encoder_diversity_budget
        chunk = max(1, int(4.8e7 // max(n * rows, 1)))  # bound the flat tensor
        rot_state = {"gen": 0}

        def bf(pop):
            perm_t = None
            if fitness_mode == "ridge" and self.encoder_split_rotate:
                # rotate the fit/val split each generation — selection can't
                # slowly overfit a fixed validation half
                rs = np.random.default_rng(((self.seed or 0) + 1) * 104729
                                           + rot_state["gen"])
                rot_state["gen"] += 1
                perm_t = t.tensor(rs.permutation(n), device=dev)
            g = np.asarray(pop, dtype=np.float32)
            out = np.empty(len(g))
            with t.no_grad():
                for i in range(0, len(g), chunk):
                    gt = t.tensor(g[i:i + chunk], device=dev)
                    p = gt.shape[0]
                    emb = gt[:, :emb_len].reshape(p, V, E)
                    flat = (emb[:, Xs_t] + pos_t[None, None]).reshape(p, n, rows)
                    if deep:
                        i0 = emb_len + rows * hidden
                        W1 = gt[:, emb_len:i0].reshape(p, rows, hidden)
                        b1 = gt[:, i0:i0 + hidden]
                        i1 = i0 + hidden
                        W2 = gt[:, i1:i1 + hidden * d].reshape(p, hidden, d)
                        b2 = gt[:, i1 + hidden * d:]
                        h = t.tanh(t.bmm(flat, W1) + b1[:, None, :])
                        ctx = t.tanh(t.bmm(h, W2) + b2[:, None, :])    # (p, n, d)
                    else:
                        W = gt[:, emb_len:emb_len + rows * d].reshape(p, rows, d)
                        b = gt[:, emb_len + rows * d:]
                        ctx = t.tanh(t.bmm(flat, W) + b[:, None, :])   # (p, n, d)
                    if fitness_mode == "ridge":
                        # held-out ridge readout accuracy, batched:
                        # fit on one half, score on the other
                        n1 = n // 2
                        ctx_r = ctx[:, perm_t] if perm_t is not None else ctx
                        yi_r = yi_t[perm_t] if perm_t is not None else yi_t
                        ones = t.ones(p, n, 1, device=dev)
                        Xa = t.cat([ctx_r, ones], -1)
                        X1, X2 = Xa[:, :n1], Xa[:, n1:]
                        XtX = t.bmm(X1.transpose(1, 2), X1)
                        XtX += (1e-3 * n1) * t.eye(d + 1, device=dev)[None]
                        B = t.zeros(p, n_classes, d + 1, device=dev)
                        B.index_add_(1, yi_r[:n1], X1)
                        Wr = t.linalg.solve(XtX, B.transpose(1, 2))
                        pred = t.bmm(X2, Wr).argmax(-1)
                        val = (pred == yi_r[None, n1:]).float().mean(1)
                    else:
                        C = t.zeros(p, n_classes, d, device=dev)
                        C.index_add_(1, yi_t, ctx)
                        C = C / t.clamp(cnt_t[None], min=1.0)
                        scores = (t.bmm(ctx, C.transpose(1, 2))
                                  - 0.5 * (C * C).sum(-1)[:, None, :])
                        val = (scores.argmax(-1) == yi_t[None]).float().mean(1)
                    if use_time:
                        active = (gt.abs() > 0.01).float().mean(1)
                        val = val / (1.0 + active / budget)
                    if use_div:
                        xc = ctx - ctx.mean(1, keepdim=True)
                        sd = xc.std(1)                                  # (p, d)
                        live = sd > 1e-8
                        R = t.bmm(xc.transpose(1, 2), xc) / max(n - 1, 1)
                        R = R / t.clamp(sd[:, :, None] * sd[:, None, :], min=1e-12)
                        pair = (live[:, :, None] & live[:, None, :]).float()
                        pair = pair * (1.0 - t.eye(d, device=dev)[None])
                        m = (R.abs() * pair).sum((1, 2)) / t.clamp(pair.sum((1, 2)), min=1.0)
                        frac = live.float().mean(1)
                        red = m * frac + (1.0 - frac)      # dead dims → redundant
                        val = val / (1.0 + red / div_budget)
                    out[i:i + chunk] = val.cpu().numpy()
            return out
        return bf

    # -- encoder pre-evolution (blueprint step 1) ----------------------------
    def _cooc_embedding_seed(self, X, y):
        """Byte-embedding seed with linguistic geometry: PCA of each byte's
        preceding-context profile down to embed_dim, so bytes that appear in
        similar contexts START near each other. Without this the embedding
        table is a neutral plateau — any re-coding can be compensated by the
        mixer, so evolution never organizes it (verified: effective rank
        stays ≈ embed_dim even at embed_dim 6)."""
        cfg = self.cfg
        F = self._byte_context_features(X, y)          # (V, ≤2V)
        Fc = F - F.mean(0)
        if cfg.vocab_size > 1024:
            # word mode: full SVD of (V, 2V) is minutes at V≥2048 — a Gaussian
            # random projection keeps the co-occurrence neighborhoods (JL) at
            # a fraction of the cost, which is all the seed needs
            G = self._rng.standard_normal((Fc.shape[1], cfg.embed_dim))
            E = Fc @ (G / np.sqrt(Fc.shape[1]))
            k = cfg.embed_dim
        else:
            _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
            k = min(cfg.embed_dim, Vt.shape[0])
            E = Fc @ Vt[:k].T                          # (V, k)
        if k < cfg.embed_dim:
            E = np.hstack([E, np.zeros((cfg.vocab_size, cfg.embed_dim - k))])
        norms = np.linalg.norm(E, axis=1)
        return E / max(float(norms.mean()), 1e-9)      # mean row norm ≈ 1

    def _heuristic_encoder_genome(self, emb_seed=None):
        """Structured seed for the encoder GA: the mixer gets identity blocks
        for the most recent positions (so the last byte(s) survive the
        projection verbatim — bigram information from generation 0) plus small
        recency-decayed random projections for older positions. Without this,
        a random projection is nearly context-blind and routers collapse to
        majority routes."""
        cfg, rng = self.cfg, self._rng
        emb = (emb_seed if emb_seed is not None
               else rng.standard_normal((cfg.vocab_size, cfg.embed_dim)) * 0.4)

        def mixer(target_dim):
            W = rng.standard_normal(
                (cfg.context_window * cfg.embed_dim, target_dim)) * 0.01
            n_blocks = min(target_dim // cfg.embed_dim, cfg.context_window)
            for k in range(n_blocks):                 # k=0 → most recent byte
                pos = cfg.context_window - 1 - k
                blk = np.eye(cfg.embed_dim) * (1.0 if k == 0 else 0.6 ** k)
                W[pos * cfg.embed_dim:(pos + 1) * cfg.embed_dim,
                  k * cfg.embed_dim:(k + 1) * cfg.embed_dim] = blk
            for pos in range(cfg.context_window - n_blocks):
                decay = 0.5 ** (cfg.context_window - n_blocks - pos)
                rows = slice(pos * cfg.embed_dim, (pos + 1) * cfg.embed_dim)
                W[rows, :] += rng.standard_normal(
                    (cfg.embed_dim, target_dim)) * 0.03 * decay
            return W

        if self.encoder_depth == 2:
            # stage 1 = the usual identity-block mixer into the hidden layer;
            # stage 2 ≈ identity (slightly >1 to counter the double tanh) so
            # the deep encoder STARTS equal to the 1-stage form — evolution
            # can only gain from the extra nonlinearity
            hidden = self.encoder_hidden or cfg.context_dim
            W1 = mixer(hidden)
            W2 = (np.eye(hidden, cfg.context_dim) * 1.2 +
                  rng.standard_normal((hidden, cfg.context_dim)) * 0.01)
            return np.concatenate([emb.ravel(), W1.ravel(), np.zeros(hidden),
                                   W2.ravel(), np.zeros(cfg.context_dim)])
        W = mixer(cfg.context_dim)
        return np.concatenate([emb.ravel(), W.ravel(),
                               np.zeros(cfg.context_dim)])

    def _evolve_encoder(self, encoder, X, y):
        """Evolve the full encoder genome. Fitness: nearest-centroid accuracy
        of the *next byte* from the encoded context — a closed-form (gradient-
        free) proxy for "can a linear router separate these contexts". """
        cfg = self.cfg
        n = min(self.encoder_samples, len(y))
        idx = self._rng.choice(len(y), size=n, replace=False)
        Xs, ys = X[idx], y[idx]
        classes = np.unique(ys)
        cls_index = np.full(cfg.vocab_size, -1, dtype=int)
        cls_index[classes] = np.arange(len(classes))
        yi = cls_index[ys]
        counts = np.bincount(yi, minlength=len(classes)).astype(float)[:, None]

        probes = {}                      # scratch encoders keyed by context_dim

        deep = isinstance(encoder, DeepContextEncoder)

        def encode(genome, d=None):
            d = int(d if d is not None else cfg.context_dim)
            p = probes.get(d)
            if p is None:
                c2 = dataclasses.replace(cfg, context_dim=d)
                p = (DeepContextEncoder(c2, hidden=encoder.hidden) if deep
                     else ContextEncoder(c2))
                probes[d] = p
            p.unpack_genome(genome)
            return p.encode(Xs)                          # (n, d)

        def encode_embed(genome, e):
            # probe keyed by EMBED dim (positions are deterministic sinusoids,
            # so probe and final encoder agree exactly)
            key = ("e", int(e))
            p = probes.get(key)
            if p is None:
                p = ContextEncoder(dataclasses.replace(cfg, embed_dim=int(e)))
                probes[key] = p
            p.unpack_genome(genome)
            return p.encode(Xs)

        def acc_of(ctx):
            C = np.zeros((len(classes), ctx.shape[1]))
            np.add.at(C, yi, ctx)
            C /= np.maximum(counts, 1.0)
            scores = ctx @ C.T - 0.5 * np.sum(C * C, axis=1)[None, :]
            return float(np.mean(np.argmax(scores, axis=1) == yi))

        n_fit = n // 2                      # ridge mode: fit half, score half

        def ridge_of(ctx):
            # held-out accuracy of a closed-form ridge readout — the same
            # family as the leaf that will consume this encoding, and immune
            # to centroid overfit (scored on the unseen half)
            dd = ctx.shape[1]
            X1 = np.hstack([ctx[:n_fit], np.ones((n_fit, 1))])
            X2 = np.hstack([ctx[n_fit:], np.ones((n - n_fit, 1))])
            XtX = X1.T @ X1 + (1e-3 * n_fit) * np.eye(dd + 1)
            XtY = np.zeros((len(classes), dd + 1))
            np.add.at(XtY, yi[:n_fit], X1)
            try:
                W = np.linalg.solve(XtX, XtY.T)
            except np.linalg.LinAlgError:
                return 0.0
            return float(np.mean((X2 @ W).argmax(1) == yi[n_fit:]))

        # residual fitness: encoder is scored by how much a ridge head on its
        # encoding improves COUNTS — trained on onehot(y) − P_big(y|w), scored
        # as held-out accuracy of P_big + head (full vocab, no routing cap)
        residual_ok = (self.encoder_fitness == "residual"
                       and self._vocab is not None)
        if self.encoder_fitness == "residual" and not residual_ok:
            self.emit({"type": "status",
                       "message": "residual encoder fitness needs word mode — using ridge"})
            self.encoder_fitness = "ridge"
        if residual_ok:
            Pb = bigram_prior(cfg.vocab_size)
            Pb1 = Pb[Xs[:n_fit, -1]]                       # (n_fit, V)
            Pb2 = Pb[Xs[n_fit:, -1]]
            R1 = -Pb1.astype(np.float32)
            R1[np.arange(n_fit), ys[:n_fit]] += 1.0        # residual targets
            y2_res = ys[n_fit:]

        def residual_of(ctx):
            dd = ctx.shape[1]
            X1 = np.hstack([ctx[:n_fit], np.ones((n_fit, 1))]).astype(np.float32)
            X2 = np.hstack([ctx[n_fit:], np.ones((n - n_fit, 1))]).astype(np.float32)
            XtX = (X1.T @ X1).astype(np.float64) + (1e-2 * n_fit) * np.eye(dd + 1)
            try:
                Wh = np.linalg.solve(XtX, (X1.T @ R1).astype(np.float64)).astype(np.float32)
            except np.linalg.LinAlgError:
                return 0.0
            s = Pb2 + X2 @ Wh
            s[:, 0] = -1e9
            return float(np.mean(s.argmax(1) == y2_res))

        score_of = (residual_of if residual_ok
                    else ridge_of if self.encoder_fitness == "ridge" else acc_of)

        def fitness(genome, d=None):
            return score_of(encode(genome, d))

        THRESH = 0.01
        budget = self.encoder_time_budget
        div_budget = self.encoder_diversity_budget

        def active_frac(genome):
            return float(np.mean(np.abs(genome) > THRESH))

        def redundancy(ctx):
            # mean |off-diagonal correlation| across the encoder's output
            # dims — 0 = every head encodes something different, 1 = all
            # heads carry the same signal. Dead (constant) dims count as
            # fully redundant so collapse can't game the penalty.
            sd = ctx.std(axis=0)
            keep = sd > 1e-8
            k, d = int(keep.sum()), ctx.shape[1]
            if k < 2:
                return 1.0
            R = np.corrcoef(ctx[:, keep].T)
            m = float(np.mean(np.abs(R[np.triu_indices(k, 1)])))
            return (m * (k / d)) + (1.0 - k / d)         # dead dims → redundant

        def fitness_constrained(genome, d=None):
            # the enabled constraints fold into the original fitness — the
            # encoder must be accurate AND cheap/diverse from generation 0
            ctx = encode(genome, d)
            val = score_of(ctx)
            if self.encoder_time_constrained:
                val /= (1.0 + active_frac(genome) / budget)
            if self.encoder_diversity:
                val /= (1.0 + redundancy(ctx) / div_budget)
            return val

        self.emit({"type": "node_start", "id": "encoder", "kind": "encoder",
                   "samples": int(n), "coverage": 100.0,
                   "tokens": cfg.vocab_size})
        epop = self.encoder_pop_size or cfg.pop_size   # 0 = inherit shared size
        emb_seed = (self._cooc_embedding_seed(X, y)
                    if self.encoder_seed_embeddings else None)
        if emb_seed is not None:
            self.emit({"type": "status",
                       "message": "embedding table seeded from byte co-occurrence geometry"})
        heuristic = self._heuristic_encoder_genome(emb_seed)
        seeds = [heuristic, encoder.pack_genome()]
        for _ in range(min(6, epop - 2)):
            seeds.append(heuristic +
                         self._rng.standard_normal(heuristic.size) *
                         (np.abs(heuristic) * 0.1 + 0.01))
        if self.encoder_time_constrained:
            # make the sparse region reachable from gen 0: magnitude-pruned
            # variants of the heuristic join the seed population
            mags_h = np.abs(heuristic)
            for q in (0.3, 0.5, 0.7):
                pruned = heuristic.copy()
                pruned[mags_h < np.quantile(mags_h, q)] = 0.0
                seeds.append(pruned)
        constrained = self.encoder_time_constrained or self.encoder_diversity
        fit_fn = fitness_constrained if constrained else fitness
        start_dim = cfg.context_dim
        start_embed = cfg.embed_dim
        if (self.encoder_evolve_dims or self.encoder_evolve_embed) and deep:
            self.emit({"type": "status",
                       "message": "evolve-dims/embed is not supported with the 2-stage encoder — using fixed dims"})
            self.encoder_evolve_dims = False
            self.encoder_evolve_embed = False
        if self.encoder_evolve_embed and self.encoder_evolve_dims:
            self.emit({"type": "status",
                       "message": "evolve-embed and evolve-dims together not supported — evolving embed only"})
            self.encoder_evolve_dims = False
        if self.encoder_evolve_embed:
            def fit_embed(genome, e):
                ctx = encode_embed(genome, e)
                val = score_of(ctx)
                if self.encoder_time_constrained:
                    val /= (1.0 + active_frac(genome) / budget)
                if self.encoder_diversity:
                    val /= (1.0 + redundancy(ctx) / div_budget)
                return val

            # MIXED-DIM seeding: one-column-at-a-time growth has no fitness
            # gradient (measured: 64 → 66 in 100 gens), so seed the population
            # with REAL heuristic genomes at several capacity levels and let
            # selection compare whole levels head-to-head from generation 0.
            e_cap = max(start_embed * 4, 256)
            levels = sorted({max(8, start_embed // 2), start_embed,
                             min(start_embed * 2, e_cap), min(start_embed * 4, e_cap)})
            seed_pairs = []
            for e_lvl in levels:
                saved_cfg = self.cfg
                self.cfg = dataclasses.replace(cfg, embed_dim=e_lvl)
                try:
                    es = (self._cooc_embedding_seed(X, y)
                          if self.encoder_seed_embeddings else None)
                    h_e = self._heuristic_encoder_genome(es)
                finally:
                    self.cfg = saved_cfg
                seed_pairs.append((h_e, e_lvl))
            self.emit({"type": "status",
                       "message": f"evolve-embed: population seeded at dims {levels}"})
            best, best_embed = self._evolve_encoder_embed(
                fit_embed, "encoder", self.encoder_generations, seed_pairs,
                start_embed, pop_size=epop)
            if best_embed != cfg.embed_dim:
                cfg.embed_dim = best_embed       # everything downstream at this width
            encoder = ContextEncoder(cfg)
        elif self.encoder_evolve_dims:
            best, best_dim = self._evolve_encoder_dims(
                fit_fn, "encoder", self.encoder_generations, seeds, start_dim,
                pop_size=epop)
            if best_dim != cfg.context_dim:
                cfg.context_dim = best_dim       # routers/leaves build at this width
                encoder = ContextEncoder(cfg)
        else:
            best = self._evolve(encoder.genome_size, fit_fn, "encoder",
                                generations=self.encoder_generations,
                                seed_pop=np.array(seeds), pop_size=epop,
                                novelty=(self.encoder_novelty_strength
                                         if self.encoder_novelty else 0.0),
                                # residual fitness has no GPU batch kernel —
                                # fall back to the per-genome python loop
                                batch_fitness=None if residual_ok
                                else self._batch_encoder_fitness(
                                    Xs, yi, counts, len(classes),
                                    encoder.positions,
                                    use_time=self.encoder_time_constrained,
                                    use_div=self.encoder_diversity,
                                    fitness_mode=self.encoder_fitness))
        encoder.unpack_genome(best)
        # score via the FINAL encoder (correct for evolved embed/context dims;
        # positions are deterministic, so this equals the probe's encoding)
        ctx_best = encoder.encode(Xs)
        acc = score_of(ctx_best)                  # raw (unconstrained) score
        self._persist_history("encoder", "encoder", acc)
        done_ev = {"type": "node_done", "id": "encoder", "kind": "encoder",
                   "acc": acc, "samples": int(n), "coverage": 100.0}
        parts = []
        if self.encoder_evolve_dims:
            done_ev["context_dim"] = cfg.context_dim
            parts.append(f"context dim {start_dim} → {cfg.context_dim}")
        if self.encoder_evolve_embed:
            done_ev["embed_dim"] = cfg.embed_dim
            parts.append(f"embed dim {start_embed} → {cfg.embed_dim}")
        if self.encoder_time_constrained:
            af = active_frac(best)
            done_ev["active_fraction"] = af
            parts.append(f"active weights {af*100:.0f}% (budget {budget})")
        if self.encoder_diversity:
            red = redundancy(ctx_best)
            done_ev["redundancy"] = red
            parts.append(f"head redundancy {red:.3f} (budget {div_budget})")
        self.emit(done_ev)
        if parts:
            self.emit({"type": "status",
                       "message": f"encoder evolved: acc {acc:.3f}, " + ", ".join(parts)})
        self._enc_info = {
            "nc_accuracy": round(acc, 4), "samples": int(n),
            "generations": self.encoder_generations,
            "pop_size": epop,
            "start_context_dim": start_dim, "context_dim": cfg.context_dim,
            "start_embed_dim": start_embed, "embed_dim": cfg.embed_dim,
            "evolve_dims": self.encoder_evolve_dims,
            "evolve_embed": self.encoder_evolve_embed,
            "time_constrained": self.encoder_time_constrained,
            "time_budget": self.encoder_time_budget if self.encoder_time_constrained else None,
            "active_fraction": round(active_frac(best), 4) if self.encoder_time_constrained else None,
            "diversity": self.encoder_diversity,
            "diversity_budget": div_budget if self.encoder_diversity else None,
            "redundancy": round(redundancy(ctx_best), 4) if self.encoder_diversity else None,
            "novelty": self.encoder_novelty,
            "novelty_strength": self.encoder_novelty_strength if self.encoder_novelty else None,
            "seeded_embeddings": self.encoder_seed_embeddings,
            "fitness_mode": self.encoder_fitness,
        }

        # ── phase 2: speed/time constraint ──
        # The champion must now do it *cheaply*: fitness × 1/(1 + active/budget)
        # where active = fraction of genes with |w| > threshold (effective
        # multiplies — for a fixed-shape dense encoder, sparsity IS the
        # evolvable notion of speed). Population is seeded with the phase-1
        # champion plus magnitude-pruned variants of it so the sparse region
        # of the landscape is reachable from generation 0.
        if self.encoder_speed_generations > 0 and not self.stopped:
            def fitness_speed(genome):   # phase 2 is pure time pressure
                return fitness(genome) / (1.0 + active_frac(genome) / budget)

            self.emit({"type": "node_start", "id": "encoder-speed",
                       "kind": "encoder", "samples": int(n), "coverage": 100.0,
                       "tokens": cfg.vocab_size})
            mags = np.abs(best)
            seeds2 = [best.copy()]
            for q in (0.3, 0.5, 0.7, 0.85):
                pruned = best.copy()
                pruned[mags < np.quantile(mags, q)] = 0.0
                seeds2.append(pruned)
            best2 = self._evolve(encoder.genome_size, fitness_speed,
                                 "encoder-speed",
                                 generations=self.encoder_speed_generations,
                                 seed_pop=np.array(seeds2), pop_size=epop,
                                 batch_fitness=self._batch_encoder_fitness(
                                     Xs, yi, counts, len(classes),
                                     encoder.positions, use_time=True))
            encoder.unpack_genome(best2)
            acc2, af = fitness(best2), active_frac(best2)
            self._persist_history("encoder-speed", "encoder", acc2)
            self.emit({"type": "node_done", "id": "encoder-speed",
                       "kind": "encoder", "acc": acc2, "samples": int(n),
                       "coverage": 100.0, "active_fraction": af})
            self.emit({"type": "status",
                       "message": (f"encoder speed phase: acc {acc:.3f} → {acc2:.3f} "
                                   f"({acc2-acc:+.3f}), active weights {af*100:.0f}%")})
            self._enc_info.update({
                "speed_generations": self.encoder_speed_generations,
                "speed_nc_accuracy": round(acc2, 4),
                "speed_active_fraction": round(af, 4),
                "speed_time_budget": self.encoder_time_budget,
            })
            acc = acc2

        encoder.freeze()
        return encoder

    def _evolve_encoder_embed(self, fitness_fn, node_id, generations,
                              seed_pairs, start_embed, pop_size=None):
        """Variable EMBED-dim GA: each individual carries its own embed_dim e.
        Genome (shallow encoder): [emb V×e | W (window·e)×D | b D]. Growing
        appends one embedding column AND its mixer row in every position
        block; shrinking drops the lowest-magnitude column + its rows.
        Bounds [8, max(4×start, 256)]. fitness_fn takes (genome, e). CPU.
        `seed_pairs` is a list of (genome, e) — mixed-dim seeding lets
        selection compare capacity levels directly from generation 0."""
        cfg, rng = self.cfg, self._rng
        pop_size = pop_size if pop_size else cfg.pop_size
        V, win, D = cfg.vocab_size, cfg.context_window, cfg.context_dim
        e_min, e_max = 8, max(start_embed * 4, 256)
        stride = max(1, generations // 40)
        top_k = max(2, int(pop_size * cfg.elite_frac))

        def split(g, e):
            emb = g[:V * e].reshape(V, e)
            W = g[V * e:V * e + win * e * D].reshape(win, e, D)
            b = g[V * e + win * e * D:]
            return emb, W, b

        def join(emb, W, b):
            return np.concatenate([emb.ravel(), W.ravel(), b])

        def grow(g, e):
            emb, W, b = split(g, e)
            emb2 = np.hstack([emb, rng.standard_normal((V, 1)) * 0.05])
            W2 = np.concatenate([W, rng.standard_normal((win, 1, D)) * 0.01], axis=1)
            return join(emb2, W2, b), e + 1

        def shrink(g, e):
            emb, W, b = split(g, e)
            mag = np.abs(emb).sum(0) + np.abs(W).sum(axis=(0, 2))
            keep = np.arange(e) != int(np.argmin(mag))
            return join(emb[:, keep], W[:, keep, :], b), e - 1

        pop = [(g.copy(), int(e)) for g, e in seed_pairs]
        i = 0
        while len(pop) < pop_size:      # noisy copies, cycling the dim levels
            g, e = seed_pairs[i % len(seed_pairs)]
            pop.append((g + rng.standard_normal(g.size) *
                        (np.abs(g) * 0.1 + 0.01), int(e)))
            i += 1
        pop = pop[:pop_size]
        best_score, best_g, best_e = -1.0, pop[0][0].copy(), pop[0][1]

        for gen in range(generations):
            if self.stopped:
                break
            scores = np.array([fitness_fn(g, e) for g, e in pop])
            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                best_score = float(scores[idx])
                best_g, best_e = pop[idx][0].copy(), pop[idx][1]

            if gen % stride == 0 or gen == generations - 1:
                mean_e = float(np.mean([e for _, e in pop]))
                self.emit({"type": "node_gen", "id": node_id, "gen": gen,
                           "best": best_score, "mean": float(scores.mean()),
                           "dim": best_e, "mean_dim": round(mean_e, 1)})
                self._enc_curves.setdefault(node_id, []).append(
                    {"gen": gen, "best": round(best_score, 4),
                     "mean": round(float(scores.mean()), 4),
                     "dim": best_e, "mean_dim": round(mean_e, 1)})

            elites = [pop[i] for i in np.argsort(scores)[-top_k:]]
            new_pop = [(best_g.copy(), best_e)]
            for _ in range(pop_size - 1):
                g1, e1 = elites[int(rng.integers(len(elites)))]
                g2, e2 = elites[int(rng.integers(len(elites)))]
                child, e = g1.copy(), e1
                # crossover in the shared embed subspace (child keeps e1)
                em = min(e1, e2)
                emb_c, W_c, b_c = split(child, e)
                emb_2, W_2, b_2 = split(g2, e2)
                m = rng.random((V, em)) > 0.5
                emb_c[:, :em][m] = emb_2[:, :em][m]
                mw = rng.random((win, em, D)) > 0.5
                W_c[:, :em, :][mw] = W_2[:, :em, :][mw]
                mb = rng.random(D) > 0.5
                b_c[mb] = b_2[mb]
                child = join(emb_c, W_c, b_c)
                if rng.random() < 0.25:            # structural mutation
                    can_grow, can_shrink = e < e_max, e > e_min
                    if can_grow and (not can_shrink or rng.random() < 0.5):
                        child, e = grow(child, e)
                    elif can_shrink:
                        child, e = shrink(child, e)
                if rng.random() < cfg.mutation_rate:
                    scale = np.abs(child) * 0.1 + 0.01
                    child = child + rng.standard_normal(child.size) * scale
                new_pop.append((child, e))
            pop = new_pop

        return best_g, best_e

    def _evolve_encoder_dims(self, fitness_fn, node_id, generations,
                             seed_genomes, start_dim, pop_size=None):
        """Variable-dimension GA for the encoder: each individual carries its
        own context_dim; mutation can grow (append a mixer column) or shrink
        (drop the lowest-magnitude column). Bounds [4, 4×start_dim]. EEC note:
        size only *changes direction* under pressure — pure accuracy fitness
        tends to grow toward the cap; pair with the time/diversity constraint
        to reward smaller encoders. fitness_fn must accept (genome, dim)."""
        cfg, rng = self.cfg, self._rng
        pop_size = pop_size if pop_size else cfg.pop_size
        emb_len = cfg.vocab_size * cfg.embed_dim
        rows = cfg.context_window * cfg.embed_dim
        d_min, d_max = 4, min(start_dim * 4, 1024)
        stride = max(1, generations // 40)
        top_k = max(2, int(pop_size * cfg.elite_frac))

        def split(g, d):                 # views into the flat genome
            emb = g[:emb_len]
            W = g[emb_len:emb_len + rows * d].reshape(rows, d)
            b = g[emb_len + rows * d:]
            return emb, W, b

        def grow(g, d):
            emb, W, b = split(g, d)
            col = rng.standard_normal((rows, 1)) * 0.01
            return (np.concatenate([emb, np.hstack([W, col]).ravel(),
                                    np.append(b, 0.0)]), d + 1)

        def shrink(g, d):
            emb, W, b = split(g, d)
            j = int(np.argmin(np.abs(W).sum(axis=0) + np.abs(b)))
            keep = np.arange(d) != j
            return np.concatenate([emb, W[:, keep].ravel(), b[keep]]), d - 1

        pop = [(sg.copy(), start_dim) for sg in seed_genomes]
        while len(pop) < pop_size:
            g = rng.standard_normal(emb_len + rows * start_dim + start_dim) * 0.1
            pop.append((g, start_dim))
        pop = pop[:pop_size]
        best_score, best_g, best_d = -1.0, pop[0][0].copy(), start_dim

        for gen in range(generations):
            if self.stopped:
                break
            scores = np.array([fitness_fn(g, d) for g, d in pop])
            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                best_score = float(scores[idx])
                best_g, best_d = pop[idx][0].copy(), pop[idx][1]

            if gen % stride == 0 or gen == generations - 1:
                mean_dim = float(np.mean([d for _, d in pop]))
                self.emit({"type": "node_gen", "id": node_id, "gen": gen,
                           "best": best_score, "mean": float(scores.mean()),
                           "dim": best_d, "mean_dim": round(mean_dim, 1)})
                self._enc_curves.setdefault(node_id, []).append(
                    {"gen": gen, "best": round(best_score, 4),
                     "mean": round(float(scores.mean()), 4),
                     "dim": best_d, "mean_dim": round(mean_dim, 1)})

            elites = [pop[i] for i in np.argsort(scores)[-top_k:]]
            new_pop = [(best_g.copy(), best_d)]
            for _ in range(pop_size - 1):
                g1, d1 = elites[int(rng.integers(len(elites)))]
                g2, d2 = elites[int(rng.integers(len(elites)))]
                child, d = g1.copy(), d1
                # crossover in the shared subspace (child keeps parent 1's dim)
                dm = min(d1, d2)
                emb_c, W_c, b_c = split(child, d)
                emb_2, W_2, b_2 = split(g2, d2)
                m = rng.random(emb_len) > 0.5
                emb_c[m] = emb_2[m]
                mw = rng.random((rows, dm)) > 0.5
                W_c[:, :dm][mw] = W_2[:, :dm][mw]
                mb = rng.random(dm) > 0.5
                b_c[:dm][mb] = b_2[:dm][mb]
                if rng.random() < 0.25:            # structural mutation
                    can_grow, can_shrink = d < d_max, d > d_min
                    if can_grow and (not can_shrink or rng.random() < 0.5):
                        child, d = grow(child, d)
                    elif can_shrink:
                        child, d = shrink(child, d)
                if rng.random() < cfg.mutation_rate:
                    scale = np.abs(child) * 0.1 + 0.01
                    child = child + rng.standard_normal(child.size) * scale
                new_pop.append((child, d))
            pop = new_pop

        return best_g, best_d

    def _nc_seed(self, ctx, labels, n_classes):
        """Closed-form nearest-centroid weights as a GA seed for a node.

        score_c = x·μ_c − ½‖μ_c‖² is the linear form of a nearest-centroid
        classifier — no gradients, just class means. Seeding the population
        with it means evolution refines a working solution instead of trying
        to find one from random noise in a few thousand evaluations.
        Classes with no samples get a large negative bias so they are never
        predicted over an observed class.
        """
        C = np.zeros((n_classes, ctx.shape[1]))
        np.add.at(C, labels, ctx)
        cnt = np.bincount(labels, minlength=n_classes).astype(float)
        C /= np.maximum(cnt, 1.0)[:, None]
        bias = -0.5 * np.sum(C * C, axis=1)
        bias[cnt == 0] = -1e6
        genome = np.concatenate([C.T.ravel(), bias])   # weights (dim, k) order

        seeds = [genome]
        if self.ridge_seed:
            # closed-form ridge/least-squares to one-hot targets (bias via an
            # augmented ones-column) — same no-gradient family as the NC seed
            # (moment statistics + one linear solve), but it sits at the
            # actual linear optimum for this node's local data
            try:
                n, d = ctx.shape
                X1 = np.hstack([ctx, np.ones((n, 1))])
                XtX = X1.T @ X1 + (1e-3 * n) * np.eye(d + 1)
                XtY = np.zeros((n_classes, d + 1))
                np.add.at(XtY, labels, X1)
                W1 = np.linalg.solve(XtX, XtY.T)          # (d+1, k)
                seeds.append(np.concatenate([W1[:d].ravel(), W1[d]]))
            except np.linalg.LinAlgError:
                pass
        for _ in range(min(4, self.cfg.pop_size - 1)):
            seeds.append(genome + self._rng.standard_normal(genome.size) *
                         (np.abs(genome) * 0.05 + 0.005))
        return np.array(seeds)

    # -- bottom-up walk (mirrors tree_lm._train_node_recursive) -------------
    def _train_node(self, node, contexts, targets, total):
        if self.stopped:
            return
        # set-membership mask — token sets may be arbitrary (clustered split),
        # not just contiguous ID ranges
        member = np.zeros(self.cfg.vocab_size, dtype=bool)
        member[node.token_set] = True
        mask = member[targets]
        ctx_local, tgt_local = contexts[mask], targets[mask]

        if len(ctx_local) == 0:
            self.emit({"type": "node_done", "id": node.node_id, "acc": None,
                       "samples": 0, "kind": "leaf" if node.is_leaf else "router"})
            node.freeze()
            return

        coverage = len(ctx_local) / total * 100

        if node.is_leaf:
            local_of = np.full(self.cfg.vocab_size, -1, dtype=int)
            local_of[node.token_set] = np.arange(len(node.token_set))
            local_targets = local_of[tgt_local]
            self.emit({"type": "node_start", "id": node.node_id, "kind": "leaf",
                       "samples": int(len(ctx_local)), "coverage": coverage,
                       "tokens": len(node.token_set)})
            if len(node.token_set) == 1:
                # A one-token specialist always predicts its token (argmax over a
                # single class) — every genome scores 1.0, so evolving it is a
                # no-op. Skip straight to the blueprint's post-evolve state.
                acc = 1.0
            else:
                smooth = self.node_fitness == "prob"

                def leaf_fitness(genome):
                    node.unpack_genome(genome)
                    if smooth:      # geometric mean prob (inverse perplexity)
                        s = ctx_local @ node.weights + node.bias
                        s = s - s.max(axis=1, keepdims=True)
                        lp = s - np.log(np.exp(s).sum(1, keepdims=True))
                        tgt = lp[np.arange(len(local_targets)), local_targets]
                        return float(np.exp(np.maximum(tgt, -20.0).mean()))
                    return float(np.mean(node.predict(ctx_local)[0] == local_targets))
                seeds = self._nc_seed(ctx_local, local_targets, len(node.token_set))
                node.unpack_genome(self._evolve(
                    node.genome_size, leaf_fitness, node.node_id, seed_pop=seeds,
                    batch_fitness=self._batch_linear_acc(
                        ctx_local, local_targets, len(node.token_set),
                        mode=self.node_fitness)))
                acc = float(np.mean(node.predict(ctx_local)[0] == local_targets))
                # Calibrate the frozen leaf's score scale so softmax sampling
                # is meaningful: ridge-seeded scores live in ~[0,1], making
                # temperature-0.8 sampling near-uniform (argmax accuracy was
                # fine, generated text was noise). Fold in the scalar α whose
                # mean top-1 softmax prob equals the leaf's measured accuracy
                # (monotone in α → bisection; argmax/accuracy unchanged).
                s = ctx_local @ node.weights + node.bias
                s = s - s.max(axis=1, keepdims=True)
                target = min(max(acc, 1.5 / s.shape[1]), 0.995)

                def top_prob(alpha):
                    e = np.exp(np.maximum(s * alpha, -60.0))
                    return float((e.max(axis=1) / e.sum(axis=1)).mean())

                lo_a, hi_a = 1e-3, 1e4
                if top_prob(hi_a) > target:      # else leave scale as-is
                    for _ in range(40):
                        mid = (lo_a * hi_a) ** 0.5
                        if top_prob(mid) < target:
                            lo_a = mid
                        else:
                            hi_a = mid
                    alpha = (lo_a * hi_a) ** 0.5
                    node.weights = node.weights * alpha
                    node.bias = node.bias * alpha
            node.prediction_accuracy = acc
            node.freeze()
            self._done += 1
            self._persist_history(node.node_id, "leaf", acc)
            self.emit({"type": "node_done", "id": node.node_id, "kind": "leaf",
                       "acc": acc, "samples": int(len(ctx_local)),
                       "coverage": coverage, "progress": self._done})
            return

        for child in node.children:
            self._train_node(child, contexts, targets, total)
        if self.stopped:
            return

        # routing label = which child's token set holds the target
        child_of = np.full(self.cfg.vocab_size, -1, dtype=int)
        for ci, ch in enumerate(node.children):
            child_of[ch.token_set] = ci
        routing_labels = child_of[tgt_local]

        self.emit({"type": "node_start", "id": node.node_id, "kind": "router",
                   "samples": int(len(ctx_local)), "coverage": coverage,
                   "tokens": len(node.token_set)})

        smooth = self.node_fitness == "prob"

        def router_fitness(genome):
            node.unpack_genome(genome)
            if smooth:
                s = ctx_local @ node.weights + node.bias
                s = s - s.max(axis=1, keepdims=True)
                lp = s - np.log(np.exp(s).sum(1, keepdims=True))
                tgt = lp[np.arange(len(routing_labels)), routing_labels]
                return float(np.exp(np.maximum(tgt, -20.0).mean()))
            return float(np.mean(node.route(ctx_local) == routing_labels))

        seeds = self._nc_seed(ctx_local, routing_labels, len(node.children))
        node.unpack_genome(self._evolve(
            node.genome_size, router_fitness, node.node_id, seed_pop=seeds,
            batch_fitness=self._batch_linear_acc(
                ctx_local, routing_labels, len(node.children),
                mode=self.node_fitness)))
        acc = float(np.mean(node.route(ctx_local) == routing_labels))
        node.routing_accuracy = acc
        node.freeze()
        self._done += 1
        self._persist_history(node.node_id, "router", acc)
        self.emit({"type": "node_done", "id": node.node_id, "kind": "router",
                   "acc": acc, "samples": int(len(ctx_local)),
                   "coverage": coverage, "progress": self._done})

    # -- full run ------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception as exc:                     # never kill the ws thread
            self._persist_finalize("error")
            self.emit({"type": "error", "message": str(exc)})

    def _run(self):
        cfg = self.cfg
        self._rng = np.random.default_rng(self.seed)
        self._done = 0
        t_start = time.time()
        self._persist_create()
        self.emit({"type": "run", "id": self.run_id})

        self._resolve_device()

        # a saved encoder fixes the encoder dims — adopt them before sampling
        pre_encoder = None
        if self.encoder_id:
            try:
                pre_encoder = load_encoder(self.encoder_id)
                ec = pre_encoder.config
                if ec.vocab_size != cfg.vocab_size:
                    raise ValueError(f"encoder vocab {ec.vocab_size} ≠ run vocab "
                                     f"{cfg.vocab_size} — token level mismatch")
                cfg.embed_dim, cfg.context_window = ec.embed_dim, ec.context_window
                cfg.context_dim = ec.context_dim
                self.emit({"type": "status",
                           "message": (f"using saved encoder {self.encoder_id} "
                                       f"(dim {ec.context_dim} · window {ec.context_window} · "
                                       f"embed {ec.embed_dim}) — skipping encoder evolution")})
            except Exception as exc:
                self.emit({"type": "status",
                           "message": (f"could not load encoder {self.encoder_id} "
                                       f"({exc}) — evolving fresh instead")})
                pre_encoder = None

        self.emit({"type": "status", "message": "loading corpus…"})
        if self.token_mode == "word":
            tokens, source, self._vocab = load_word_tokens(cfg.vocab_size)
        else:
            tokens, source = load_corpus()
            self._vocab = None

        n_test = max(200, self.max_samples // 8)
        want = self.max_samples + n_test
        if self._vocab is not None:
            # never TRAIN to predict <unk> (id 0) — it's the most frequent
            # "word" at low coverage and would dominate every prediction.
            # It stays in the CONTEXT windows. Oversample to cover the drop.
            X, y = sample_windows(tokens, int(want * 1.6) + 64,
                                  cfg.context_window, self._rng)
            keep = y != 0
            X, y = X[keep][:want], y[keep][:want]
        else:
            X, y = sample_windows(tokens, want, cfg.context_window, self._rng)
        n_test = min(n_test, max(1, len(y) // 5))   # tiny fallback corpus safety
        X_train, y_train = X[:-n_test], y[:-n_test]
        X_test, y_test = X[-n_test:], y[-n_test:]
        self.emit({"type": "corpus", "source": source,
                   "corpus_bytes": int(len(tokens)),
                   "train_samples": int(len(y_train)),
                   "test_samples": int(len(y_test))})

        if self.cluster_tokens:
            self.emit({"type": "status",
                       "message": "clustering tokens by context co-occurrence…"})
            self._byte_F = self._byte_context_features(X_train, y_train)
        root = self._build_root()
        encoder = pre_encoder if pre_encoder is not None else self._make_encoder()
        nodes, total_params = self._tree_manifest(root, encoder)
        self.emit({"type": "tree", "nodes": nodes, "total_params": int(total_params),
                   "config": {"branching_factor": cfg.branching_factor,
                              "context_window": cfg.context_window,
                              "embed_dim": cfg.embed_dim, "context_dim": cfg.context_dim,
                              "min_leaf_tokens": cfg.min_leaf_tokens,
                              "pop_size": cfg.pop_size, "generations": cfg.generations}})

        if self.encoder_generations > 0 and pre_encoder is None:
            self.emit({"type": "status",
                       "message": "evolving context encoder (blueprint step 1)…"})
            dim0 = cfg.context_dim
            encoder = self._evolve_encoder(encoder, X_train, y_train)
            if self.stopped:
                self._persist_finalize("stopped", root=root, encoder=encoder)
                self.emit({"type": "stopped", "trained_nodes": 0,
                           "run_id": self.run_id})
                return
            if cfg.context_dim != dim0:
                # dims evolved — router/leaf genome sizes depend on
                # context_dim, so rebuild the (untrained) tree at the new
                # width and re-emit the manifest for the icicle/params tiles
                self.emit({"type": "status",
                           "message": (f"context dim evolved {dim0} → "
                                       f"{cfg.context_dim}; rebuilding routing tree…")})
                root = self._build_root()
                nodes, total_params = self._tree_manifest(root, encoder)
                self.emit({"type": "tree", "nodes": nodes,
                           "total_params": int(total_params),
                           "config": {"branching_factor": cfg.branching_factor,
                                      "context_window": cfg.context_window,
                                      "embed_dim": cfg.embed_dim,
                                      "context_dim": cfg.context_dim,
                                      "min_leaf_tokens": cfg.min_leaf_tokens,
                                      "pop_size": cfg.pop_size,
                                      "generations": cfg.generations}})

        self.emit({"type": "status", "message": "encoding contexts…"})
        contexts = self._encode_chunked(encoder, X_train)

        self.emit({"type": "status", "message": "evolving tree (bottom-up freeze-and-stack)…"})
        self._train_node(root, contexts, y_train, len(y_train))

        if self.stopped:
            # persist the partial model too — untrained nodes keep random init
            self._persist_finalize("stopped", root=root, encoder=encoder)
            self.emit({"type": "stopped", "trained_nodes": self._done,
                       "run_id": self.run_id})
            return

        # held-out evaluation + bigram baseline
        self.emit({"type": "status", "message": "evaluating on held-out data…"})
        t0 = time.time()
        preds = tree_predict_token(root, encoder, X_test)
        elapsed = max(time.time() - t0, 1e-6)

        V = cfg.vocab_size
        counts = np.zeros((V, V), dtype=np.int32)
        np.add.at(counts, (X_train[:, -1], y_train), 1)
        bigram_preds = counts.argmax(axis=1)[X_test[:, -1]]

        info = {
            "token_mode": self.token_mode,
            "vocab_size": int(V),
            "accuracy": float(np.mean(preds == y_test)),
            "train_accuracy": None,
            "bigram_accuracy": float(np.mean(bigram_preds == y_test)),
            "tokens_per_sec": float(len(y_test) / elapsed),
            "unique_predictions": int(len(np.unique(preds))),
            "target_entropy": float(_entropy(y_test)),
            "total_params": int(total_params),
            "train_seconds": round(time.time() - t_start, 1),
        }
        self.emit({"type": "eval", **info})

        _register_model(root, encoder, cfg, info,
                        run_id=self.run_id, run_dir=self.run_dir,
                        vocab=self._vocab)
        path = self._persist_finalize("finished", info=info, root=root, encoder=encoder)
        self.emit({"type": "done", "saved": path, "run_id": self.run_id,
                   "seconds": round(time.time() - t_start, 1)})

    @staticmethod
    def _encode_chunked(encoder, X, chunk=4096):
        # encoder.encode materializes (batch, ctx*embed) — chunk to bound memory
        return np.vstack([encoder.encode(X[i:i + chunk])
                          for i in range(0, len(X), chunk)])


# --------------------------------------------------------------------------
# Standalone encoder training — evolves and saves a reusable encoder without
# building a tree. Persists to runs/encoder/<id>/ (shows up as an "encoder"
# tab on the runs dashboard); tree runs reference it via encoder_id.
# --------------------------------------------------------------------------
class EncoderTrainer(TreeLMTrainer):

    def _persist_create(self):
        ts = datetime.datetime.now()
        cfg_dict = self._cfg_dict()
        cfg_dict["environment"] = "encoder"
        h = hashlib.sha1(json.dumps(cfg_dict, sort_keys=True, default=str)
                         .encode()).hexdigest()[:6]
        self.run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}-encoder-{h}"
        self.run_dir = os.path.join(ENC_DIR, self.run_id)
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"id": self.run_id, "environment": "encoder",
                           "created": ts.isoformat(timespec="seconds"),
                           "config": cfg_dict,
                           "started": {"population": self.encoder_pop_size or cfg_dict["pop_size"],
                                       "generations": self.encoder_generations,
                                       "notes": self.notes},
                           "status": "running"}, f, indent=2)
            open(os.path.join(self.run_dir, "history.jsonl"), "w").close()
        except OSError:
            self.run_dir = None

    def _run(self):
        cfg = self.cfg
        self._rng = np.random.default_rng(self.seed)
        self._done = 0
        t_start = time.time()
        self.notes = str(self.notes or "standalone encoder")
        self._persist_create()
        self.emit({"type": "run", "id": self.run_id})
        self._resolve_device()

        self.emit({"type": "status", "message": "loading corpus…"})
        if self.token_mode == "word":
            tokens, source, self._vocab = load_word_tokens(cfg.vocab_size)
        else:
            tokens, source = load_corpus()
            self._vocab = None
        X, y = sample_windows(tokens, self.max_samples, cfg.context_window,
                              self._rng)
        if self._vocab is not None:        # don't train the encoder on <unk> targets
            keep = y != 0
            X, y = X[keep], y[keep]
        self.emit({"type": "corpus", "source": source,
                   "corpus_bytes": int(len(tokens)),
                   "train_samples": int(len(y)), "test_samples": 0})

        self.emit({"type": "status", "message": "evolving context encoder…"})
        encoder = self._evolve_encoder(self._make_encoder(), X, y)

        status = "stopped" if self.stopped else "finished"
        saved = None
        if self.run_dir is not None:
            try:
                np.savez_compressed(
                    os.path.join(self.run_dir, "encoder.npz"),
                    genome=encoder.pack_genome(),
                    dims=np.array([cfg.vocab_size, cfg.embed_dim,
                                   cfg.context_window, cfg.context_dim]),
                    kind=np.array([2 if isinstance(encoder, DeepContextEncoder) else 1]),
                    hidden=np.array([getattr(encoder, "hidden", 0)]))
                saved = os.path.join(self.run_dir, "encoder.npz")
            except Exception:
                saved = None
            enc_block = dict(self._enc_info)
            enc_block.update({"embed_dim": cfg.embed_dim,
                              "context_window": cfg.context_window,
                              "curves": self._enc_curves})
            summary = {
                "id": self.run_id, "environment": "encoder", "status": status,
                "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                "gen": self._done,
                "best": {"score": self._enc_info.get("nc_accuracy")},
                "checkpoint": "encoder.npz" if saved else None,
                "encoder": enc_block,
                "seconds": round(time.time() - t_start, 1),
            }
            try:
                with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                cpath = os.path.join(self.run_dir, "config.json")
                with open(cpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["status"] = status
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except (OSError, ValueError):
                pass

        self.emit({"type": "encoders", "list": list_encoders()})
        if self.stopped:
            self.emit({"type": "stopped", "trained_nodes": self._done,
                       "run_id": self.run_id, "encoder_only": True})
        else:
            self.emit({"type": "done", "saved": saved, "run_id": self.run_id,
                       "encoder_only": True,
                       "seconds": round(time.time() - t_start, 1)})


# --------------------------------------------------------------------------
# Job hub — training survives WebSocket disconnects (page navigation).
# The active trainer/sweeper emits into the hub; sockets subscribe on attach
# and get a replayable snapshot (journal) so the UI rebuilds mid-run.
# --------------------------------------------------------------------------
def _notify_board(ev):
    """End-of-run alarm: post terminal job events to the shared Agent panel.
    Best-effort — the board must never be able to break training."""
    try:
        import agent_board
        agent_board.post_run_event("tree-lm", ev)
    except Exception:
        pass


class JobHub:
    JOURNAL_CAP = 50000

    def __init__(self):
        self._lock = threading.Lock()
        self._trainer = None
        self._thread = None
        self._journal = []        # replayable events (all but per-gen ticks)
        self._cur_gens = []       # node_gen of the node currently evolving
        self._subs = []

    def running(self):
        with self._lock:
            th = self._thread
        return bool(th is not None and th.is_alive())

    def subscribe(self, fn):
        with self._lock:
            self._subs.append(fn)

    def unsubscribe(self, fn):
        with self._lock:
            if fn in self._subs:
                self._subs.remove(fn)

    def snapshot(self):
        with self._lock:
            return list(self._journal) + list(self._cur_gens)

    def _emit(self, ev):
        with self._lock:
            t = ev.get("type")
            if t == "node_gen":
                self._cur_gens.append(ev)
            elif t == "node_start":
                self._cur_gens = []
                self._journal.append(ev)
            elif t == "status":
                # only the latest status matters on replay
                self._journal = [e for e in self._journal
                                 if e.get("type") != "status"]
                self._journal.append(ev)
            else:
                self._journal.append(ev)
            if len(self._journal) > self.JOURNAL_CAP:
                self._journal = self._journal[-self.JOURNAL_CAP:]
            subs = list(self._subs)
        if ev.get("type") in ("done", "sweep_done", "stopped", "error"):
            _notify_board(ev)             # Agent-panel alarm on job end/crash
        for fn in subs:
            try:
                fn(ev)
            except Exception:
                self.unsubscribe(fn)      # dead socket

    def start(self, msg, cls):
        with self._lock:
            old_tr, old_th = self._trainer, self._thread
        if old_tr is not None:
            old_tr.stop()
        if old_th is not None:
            old_th.join(timeout=5.0)
        trainer = cls(msg, self._emit)
        th = threading.Thread(target=trainer.run, name="tree-lm-job", daemon=True)
        with self._lock:
            self._trainer, self._thread = trainer, th
            self._journal, self._cur_gens = [], []
        th.start()

    def stop(self):
        with self._lock:
            tr = self._trainer
        if tr is not None:
            tr.stop()


HUB = JobHub()


# --------------------------------------------------------------------------
# Config sweep — user-defined grid: lock params to the sidebar value or list
# values to test; all combinations run; the sweep persists as its own run
# --------------------------------------------------------------------------
SWEEP_KEYS = {                            # sweepable params → short label
    "routing_layers": "layers", "branching_factor": "bf",
    "context_dim": "dim", "embed_dim": "emb", "context_window": "win",
    "max_samples": "n", "pop_size": "pop", "generations": "gens",
    "encoder_generations": "enc", "encoder_speed_generations": "spd",
    "encoder_time_budget": "budget", "mutation_rate": "mut",
}
MAX_SWEEP_CANDIDATES = 24


class TreeSweeper:
    """Runs a user-configured grid sequentially. Base config = the sidebar
    values; msg["sweep"] maps param → list of values to test (everything else
    stays locked to the base). Each candidate is a normal TreeLMTrainer run
    (persists to the runs store, streams to the live panels); the sweep itself
    also persists as a run (runs/tree/<id>-sweep-…) whose summary carries the
    ranked results, so it can be reviewed on the /runs dashboard."""

    def __init__(self, msg: dict, emit):
        self.base = {k: v for k, v in msg.items() if k not in ("op", "sweep")}
        self.base.setdefault("seed", 1234)   # same data/init across candidates
        self.emit = emit
        self._stop = threading.Event()
        self._current = None
        self.spec = {}
        for key, vals in (msg.get("sweep") or {}).items():
            if key not in SWEEP_KEYS or not isinstance(vals, list):
                continue
            cast = float if key in ("encoder_time_budget", "mutation_rate") else int
            clean = []
            for v in vals:
                try:
                    clean.append(cast(v))
                except (TypeError, ValueError):
                    continue
            if clean:
                self.spec[key] = clean[:8]

    def stop(self):
        self._stop.set()
        cur = self._current
        if cur is not None:
            cur.stop()

    def _candidates(self):
        import itertools
        keys = list(self.spec)
        combos = list(itertools.islice(
            itertools.product(*(self.spec[k] for k in keys)),
            MAX_SWEEP_CANDIDATES + 1))
        truncated = len(combos) > MAX_SWEEP_CANDIDATES
        out = []
        for combo in combos[:MAX_SWEEP_CANDIDATES]:
            ov = dict(zip(keys, combo))
            name = " · ".join(f"{SWEEP_KEYS[k]}={ov[k]:g}" for k in keys)
            out.append((name, ov))
        return out, truncated

    # -- sweep persistence (its own entry in the runs store) -----------------
    def _persist_create(self):
        ts = datetime.datetime.now()
        h = hashlib.sha1(json.dumps([self.base, self.spec], sort_keys=True,
                                    default=str).encode()).hexdigest()[:6]
        self.run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}-sweep-{h}"
        self.run_dir = os.path.join(MODEL_DIR, self.run_id)
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            cfg = dict(self.base)
            cfg["environment"] = "tree"
            cfg["sweep"] = self.spec
            with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"id": self.run_id, "environment": "tree",
                           "created": ts.isoformat(timespec="seconds"),
                           "config": cfg,
                           "started": {"notes": "config sweep: " + ", ".join(
                               f"{k}∈{v}" for k, v in self.spec.items())},
                           "status": "running"}, f, indent=2)
            open(os.path.join(self.run_dir, "history.jsonl"), "w").close()
        except OSError:
            self.run_dir = None

    def _persist_result(self, i, r):
        if self.run_dir is None:
            return
        try:
            with open(os.path.join(self.run_dir, "history.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"gen": i, "node": r["name"],
                                    "fitness": {"best": round(r["accuracy"], 4),
                                                "mean": round(r["bigram"], 4)}}) + "\n")
        except OSError:
            pass

    def _persist_finalize(self, status, results):
        if self.run_dir is None:
            return
        best = results[0] if results else None
        summary = {
            "id": self.run_id, "environment": "tree", "status": status,
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
            "gen": len(results),
            "best": ({"score": round(best["accuracy"], 4),
                      "base": round(best["bigram"], 4)} if best else None),
            "checkpoint": None,
            "sweep_results": results,
        }
        try:
            with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            cpath = os.path.join(self.run_dir, "config.json")
            with open(cpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = status
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except (OSError, ValueError):
            pass

    # -- run ------------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.emit({"type": "error", "message": str(exc)})

    def _run(self):
        if not self.spec:
            self.emit({"type": "error",
                       "message": "sweep: check at least one parameter and give it values"})
            return
        cands, truncated = self._candidates()
        self._persist_create()
        self.emit({"type": "sweep_start", "total": len(cands),
                   "id": self.run_id, "truncated": truncated,
                   "names": [n for n, _ in cands]})
        results = []
        for i, (name, overrides) in enumerate(cands):
            if self._stop.is_set():
                break
            self.emit({"type": "sweep_progress", "i": i, "name": name,
                       "total": len(cands)})
            cfg_msg = dict(self.base)
            cfg_msg.update(overrides)
            cfg_msg["notes"] = f"sweep {self.run_id} · {name}"
            holder = {}

            def inner_emit(ev):
                if ev.get("type") in ("eval", "done"):
                    holder[ev["type"]] = ev
                self.emit(ev)

            trainer = TreeLMTrainer(cfg_msg, inner_emit)
            self._current = trainer
            trainer.run()                  # synchronous — one candidate at a time
            self._current = None
            if "eval" in holder:
                r = {"i": i, "name": name, "overrides": overrides,
                     "accuracy": holder["eval"]["accuracy"],
                     "bigram": holder["eval"]["bigram_accuracy"],
                     "seconds": holder.get("done", {}).get("seconds"),
                     "run_id": holder.get("done", {}).get("run_id")}
                results.append(r)
                self._persist_result(i, r)
                self.emit({"type": "sweep_result", **r})
        results.sort(key=lambda r: r["accuracy"], reverse=True)
        status = "stopped" if self._stop.is_set() else "finished"
        self._persist_finalize(status, results)
        self.emit({"type": "sweep_done", "results": results, "id": self.run_id,
                   "stopped": self._stop.is_set()})
