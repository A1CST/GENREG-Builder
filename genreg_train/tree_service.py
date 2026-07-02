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
import hashlib
import json
import os
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


def _register_model(root, encoder, config, info, run_id=None, run_dir=None):
    with _model_lock:
        _model.update(root=root, encoder=encoder, config=config, info=info,
                      run_id=run_id, run_dir=run_dir)


def generate_text(prompt: str, length: int = 300, temperature: float = 0.0,
                  seed=None) -> str:
    """Autoregressive byte generation with the last in-memory trained tree."""
    with _model_lock:
        root, encoder, cfg = _model["root"], _model["encoder"], _model["config"]
    if root is None:
        raise RuntimeError("no trained model — run training first")
    return _generate(root, encoder, cfg, prompt, length, temperature, seed)


def infer_run(model_path: str, cfg_dict: dict, prompt: str = "the ",
              length: int = 400, temperature: float = 0.8):
    """Replay hook for the /runs dashboard: load a saved tree run and
    generate a text sample from its frozen model."""
    import io
    cfg, _, _ = parse_config(cfg_dict)
    # read into memory: np.load keeps the file handle open inside load_tree,
    # which locks model.npz on Windows for as long as the app lives
    with open(model_path, "rb") as fh:
        buf = io.BytesIO(fh.read())
    root, encoder = load_tree(buf, cfg)
    text = _generate(root, encoder, cfg, prompt, length, temperature, None)
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
    if root is None:
        raise RuntimeError("no trained model — run training first")

    length = max(1, min(int(length), 200))
    rng = np.random.default_rng(seed)
    tokens = list(prompt.encode("utf-8")) or [ord(" ")]
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
                              "sample": _token_sample(ch.token_set, 8)}
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
            "context": bytes(ctx_bytes).decode("utf-8", errors="replace"),
            "byte": int(byte),
            "path": path,
            "leaf": {
                "id": node.node_id, "t0": min(node.token_set),
                "t1": max(node.token_set) + 1, "tokens": len(node.token_set),
                "sample": _token_sample(node.token_set, 8),
                "chosen_byte": int(byte),
                "top": [{"byte": int(node.token_set[j]),
                         "score": round(float(s[j]), 4),
                         "prob": round(float(p[j]), 4)} for j in order],
            },
        })
        tokens.append(byte)

    text = bytes(tokens[n_prompt:]).decode("utf-8", errors="replace")
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


def _generate(root, encoder, cfg, prompt: str, length: int,
              temperature: float, seed) -> str:
    """Greedy routing through the tree (as in tree_lm inference); at the leaf,
    temperature > 0 samples from softmax(scores / T) over the leaf's token
    subset instead of argmax — pure argmax loops quickly on repeated n-grams.
    """
    length = max(1, min(int(length), 2000))
    rng = np.random.default_rng(seed)
    tokens = list(prompt.encode("utf-8")) or [ord(" ")]

    for _ in range(length):
        ctx = np.array(tokens[-cfg.context_window:], dtype=int)
        vec = encoder.encode(ctx)          # (1, context_dim)
        node = root
        while not node.is_leaf:
            node = node.children[int(node.route(vec)[0])]
        local_idx, scores = node.predict(vec)
        if temperature > 0 and len(node.token_set) > 1:
            s = scores[0] / max(temperature, 1e-6)
            s -= s.max()
            p = np.exp(s)
            p /= p.sum()
            choice = int(rng.choice(len(p), p=p))
        else:
            choice = int(local_idx[0])
        tokens.append(node.token_set[min(choice, len(node.token_set) - 1)])

    out = bytes(tokens[len(prompt.encode('utf-8')) or 1:])
    return out.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Config parsing (clamped so a browser can't wedge the server)
# --------------------------------------------------------------------------
def _token_sample(tokens, k=12):
    """Short printable preview of a token set, e.g. '␣ e t a …'."""
    out = []
    for t in tokens[:k]:
        if t == 32:
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
    bf = _num(msg, "branching_factor", 4, 2, 64)
    # routing_layers is the direct depth lever: leaf size = vocab / bf^layers.
    # 0 layers → min_leaf 256 → the root itself is one flat specialist over
    # all bytes (no routers). Falls back to explicit min_leaf_tokens.
    if msg.get("routing_layers") is not None:
        layers = _num(msg, "routing_layers", 2, 0, 8)
        min_leaf = 256 if layers == 0 else max(1, -(-256 // (bf ** layers)))
    else:
        min_leaf = _num(msg, "min_leaf_tokens", 16, 1, 256)
    cfg = TreeConfig(
        vocab_size=256,
        branching_factor=bf,
        context_window=_num(msg, "context_window", 16, 2, 64),
        embed_dim=_num(msg, "embed_dim", 32, 4, 128),
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
        # encoder pre-evolution (blueprint step 1); 0 generations = skip
        self.encoder_generations = _num(msg, "encoder_generations", 40, 0, 500)
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
        # evolve the encoder's output dimension count (context_dim) instead of
        # keeping it fixed: mutation can grow/shrink the mixer, start = the
        # sidebar Context dim, bounds [4, 4×start]. Off = fixed (default).
        self.encoder_evolve_dims = bool(msg.get("encoder_evolve_dims"))
        # cluster the vocabulary by context co-occurrence instead of splitting
        # into sequential byte-ID ranges — 'e', 't' and space land in different
        # branches because their preceding-context profiles differ
        self.cluster_tokens = bool(msg.get("cluster_tokens"))
        self._byte_F = None                 # context-profile features per byte
        # compute device: "gpu" batches whole-population fitness evaluation
        # on CUDA (torch); "cpu" keeps the numpy per-individual loop
        self.device = "gpu" if str(msg.get("device", "cpu")).lower() == "gpu" else "cpu"
        self._torch = None                  # resolved at run start
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
            "encoder_evolve_dims": self.encoder_evolve_dims,
            "cluster_tokens": self.cluster_tokens,
            "device": self.device,
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
                save_tree(root, encoder, os.path.join(self.run_dir, "model.npz"))
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
                batch_fitness=None):
        cfg, rng = self.cfg, self._rng
        generations = generations if generations is not None else cfg.generations
        pop_size = pop_size if pop_size else cfg.pop_size
        stride = max(1, generations // 40)
        pop = rng.standard_normal((pop_size, genome_size)) * 0.1
        if seed_pop is not None:                 # inject structured candidates
            pop[:min(len(seed_pop), pop_size)] = seed_pop[:pop_size]
        best_score, best_ind = -1.0, pop[0].copy()
        top_k = max(2, int(pop_size * cfg.elite_frac))

        for gen in range(generations):
            if self.stopped:
                break
            scores = (np.asarray(batch_fitness(pop), dtype=float)
                      if batch_fitness is not None
                      else np.array([fitness_fn(ind) for ind in pop]))
            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                best_score, best_ind = float(scores[idx]), pop[idx].copy()

            if gen % stride == 0 or gen == generations - 1:
                self.emit({"type": "node_gen", "id": node_id, "gen": gen,
                           "best": best_score, "mean": float(scores.mean())})
                if node_id.startswith("encoder"):   # persist for the runs panel
                    self._enc_curves.setdefault(node_id, []).append(
                        {"gen": gen, "best": round(best_score, 4),
                         "mean": round(float(scores.mean()), 4)})

            elites = pop[np.argsort(scores)[-top_k:]]
            new_pop = [best_ind.copy()]
            for _ in range(pop_size - 1):
                p1, p2 = elites[rng.choice(len(elites), 2, replace=False)]
                mask = rng.random(genome_size) > 0.5
                child = np.where(mask, p1, p2)
                if rng.random() < cfg.mutation_rate:
                    scale = np.abs(child) * 0.1 + 0.01   # relative mutation — GENREG style
                    child = child + rng.standard_normal(genome_size) * scale
                new_pop.append(child)
            pop = np.array(new_pop)

        return best_ind

    # -- GPU-batched fitness (device == "gpu") -------------------------------
    def _batch_linear_acc(self, ctx, labels, k):
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
        chunk = max(1, int(6.4e7 // max(n * k, 1)))     # ≤ ~256 MB of scores

        def bf(pop):
            g = np.asarray(pop, dtype=np.float32)
            out = np.empty(len(g))
            with t.no_grad():
                for i in range(0, len(g), chunk):
                    gt = t.tensor(g[i:i + chunk], device=dev)
                    W = gt[:, :dim * k].reshape(-1, dim, k)
                    b = gt[:, dim * k:]
                    scores = t.einsum("nd,pdk->pnk", ctx_t, W) + b[:, None, :]
                    acc = (scores.argmax(-1) == lab_t[None]).float().mean(1)
                    out[i:i + chunk] = acc.cpu().numpy()
            return out
        return bf

    def _batch_encoder_fitness(self, Xs, yi, counts, n_classes, positions,
                               use_time=False, use_div=False):
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
        n = len(yi)
        dev = "cuda"
        Xs_t = t.tensor(np.asarray(Xs), dtype=t.long, device=dev)      # (n, win)
        yi_t = t.tensor(np.asarray(yi), dtype=t.long, device=dev)
        cnt_t = t.tensor(np.asarray(counts), dtype=t.float32, device=dev)  # (k, 1)
        pos_t = t.tensor(positions, dtype=t.float32, device=dev)       # (win, E)
        budget, div_budget = self.encoder_time_budget, self.encoder_diversity_budget
        chunk = max(1, int(4.8e7 // max(n * rows, 1)))  # bound the flat tensor

        def bf(pop):
            g = np.asarray(pop, dtype=np.float32)
            out = np.empty(len(g))
            with t.no_grad():
                for i in range(0, len(g), chunk):
                    gt = t.tensor(g[i:i + chunk], device=dev)
                    p = gt.shape[0]
                    emb = gt[:, :emb_len].reshape(p, V, E)
                    W = gt[:, emb_len:emb_len + rows * d].reshape(p, rows, d)
                    b = gt[:, emb_len + rows * d:]
                    flat = (emb[:, Xs_t] + pos_t[None, None]).reshape(p, n, rows)
                    ctx = t.tanh(t.bmm(flat, W) + b[:, None, :])       # (p, n, d)
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
    def _heuristic_encoder_genome(self):
        """Structured seed for the encoder GA: the mixer gets identity blocks
        for the most recent positions (so the last byte(s) survive the
        projection verbatim — bigram information from generation 0) plus small
        recency-decayed random projections for older positions. Without this,
        a random projection is nearly context-blind and routers collapse to
        majority routes."""
        cfg, rng = self.cfg, self._rng
        emb = rng.standard_normal((cfg.vocab_size, cfg.embed_dim)) * 0.4
        W = rng.standard_normal(
            (cfg.context_window * cfg.embed_dim, cfg.context_dim)) * 0.01
        n_blocks = min(cfg.context_dim // cfg.embed_dim, cfg.context_window)
        for k in range(n_blocks):                     # k=0 → most recent byte
            pos = cfg.context_window - 1 - k
            blk = np.eye(cfg.embed_dim) * (1.0 if k == 0 else 0.6 ** k)
            W[pos * cfg.embed_dim:(pos + 1) * cfg.embed_dim,
              k * cfg.embed_dim:(k + 1) * cfg.embed_dim] = blk
        for pos in range(cfg.context_window - n_blocks):
            decay = 0.5 ** (cfg.context_window - n_blocks - pos)
            rows = slice(pos * cfg.embed_dim, (pos + 1) * cfg.embed_dim)
            W[rows, :] += rng.standard_normal(
                (cfg.embed_dim, cfg.context_dim)) * 0.03 * decay
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

        def encode(genome, d=None):
            d = int(d if d is not None else cfg.context_dim)
            p = probes.get(d)
            if p is None:
                p = ContextEncoder(dataclasses.replace(cfg, context_dim=d))
                probes[d] = p
            p.unpack_genome(genome)
            return p.encode(Xs)                          # (n, d)

        def acc_of(ctx):
            C = np.zeros((len(classes), ctx.shape[1]))
            np.add.at(C, yi, ctx)
            C /= np.maximum(counts, 1.0)
            scores = ctx @ C.T - 0.5 * np.sum(C * C, axis=1)[None, :]
            return float(np.mean(np.argmax(scores, axis=1) == yi))

        def fitness(genome, d=None):
            return acc_of(encode(genome, d))

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
            val = acc_of(ctx)
            if self.encoder_time_constrained:
                val /= (1.0 + active_frac(genome) / budget)
            if self.encoder_diversity:
                val /= (1.0 + redundancy(ctx) / div_budget)
            return val

        self.emit({"type": "node_start", "id": "encoder", "kind": "encoder",
                   "samples": int(n), "coverage": 100.0,
                   "tokens": cfg.vocab_size})
        epop = self.encoder_pop_size or cfg.pop_size   # 0 = inherit shared size
        heuristic = self._heuristic_encoder_genome()
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
        if self.encoder_evolve_dims:
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
                                batch_fitness=self._batch_encoder_fitness(
                                    Xs, yi, counts, len(classes),
                                    encoder.positions,
                                    use_time=self.encoder_time_constrained,
                                    use_div=self.encoder_diversity))
        encoder.unpack_genome(best)
        ctx_best = encode(best)
        acc = acc_of(ctx_best)                    # report raw accuracy either way
        self._persist_history("encoder", "encoder", acc)
        done_ev = {"type": "node_done", "id": "encoder", "kind": "encoder",
                   "acc": acc, "samples": int(n), "coverage": 100.0}
        parts = []
        if self.encoder_evolve_dims:
            done_ev["context_dim"] = cfg.context_dim
            parts.append(f"context dim {start_dim} → {cfg.context_dim}")
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
            "evolve_dims": self.encoder_evolve_dims,
            "time_constrained": self.encoder_time_constrained,
            "time_budget": self.encoder_time_budget if self.encoder_time_constrained else None,
            "active_fraction": round(active_frac(best), 4) if self.encoder_time_constrained else None,
            "diversity": self.encoder_diversity,
            "diversity_budget": div_budget if self.encoder_diversity else None,
            "redundancy": round(redundancy(ctx_best), 4) if self.encoder_diversity else None,
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
                def leaf_fitness(genome):
                    node.unpack_genome(genome)
                    return float(np.mean(node.predict(ctx_local)[0] == local_targets))
                seeds = self._nc_seed(ctx_local, local_targets, len(node.token_set))
                node.unpack_genome(self._evolve(
                    node.genome_size, leaf_fitness, node.node_id, seed_pop=seeds,
                    batch_fitness=self._batch_linear_acc(
                        ctx_local, local_targets, len(node.token_set))))
                acc = float(np.mean(node.predict(ctx_local)[0] == local_targets))
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

        def router_fitness(genome):
            node.unpack_genome(genome)
            return float(np.mean(node.route(ctx_local) == routing_labels))

        seeds = self._nc_seed(ctx_local, routing_labels, len(node.children))
        node.unpack_genome(self._evolve(
            node.genome_size, router_fitness, node.node_id, seed_pop=seeds,
            batch_fitness=self._batch_linear_acc(
                ctx_local, routing_labels, len(node.children))))
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

        self._torch = None
        if self.device == "gpu":
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

        self.emit({"type": "status", "message": "loading corpus…"})
        tokens, source = load_corpus()

        n_test = max(200, self.max_samples // 8)
        X, y = sample_windows(tokens, self.max_samples + n_test,
                              cfg.context_window, self._rng)
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
        encoder = ContextEncoder(cfg)
        nodes, total_params = self._tree_manifest(root, encoder)
        self.emit({"type": "tree", "nodes": nodes, "total_params": int(total_params),
                   "config": {"branching_factor": cfg.branching_factor,
                              "context_window": cfg.context_window,
                              "embed_dim": cfg.embed_dim, "context_dim": cfg.context_dim,
                              "min_leaf_tokens": cfg.min_leaf_tokens,
                              "pop_size": cfg.pop_size, "generations": cfg.generations}})

        if self.encoder_generations > 0:
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

        counts = np.zeros((256, 256), dtype=np.int64)
        np.add.at(counts, (X_train[:, -1], y_train), 1)
        bigram_preds = counts.argmax(axis=1)[X_test[:, -1]]

        info = {
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
                        run_id=self.run_id, run_dir=self.run_dir)
        path = self._persist_finalize("finished", info=info, root=root, encoder=encoder)
        self.emit({"type": "done", "saved": path, "run_id": self.run_id,
                   "seconds": round(time.time() - t_start, 1)})

    @staticmethod
    def _encode_chunked(encoder, X, chunk=4096):
        # encoder.encode materializes (batch, ctx*embed) — chunk to bound memory
        return np.vstack([encoder.encode(X[i:i + chunk])
                          for i in range(0, len(X), chunk)])


# --------------------------------------------------------------------------
# Job hub — training survives WebSocket disconnects (page navigation).
# The active trainer/sweeper emits into the hub; sockets subscribe on attach
# and get a replayable snapshot (journal) so the UI rebuilds mid-run.
# --------------------------------------------------------------------------
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
