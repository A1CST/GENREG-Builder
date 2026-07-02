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
                "children": [{"t0": min(ch.token_set), "t1": max(ch.token_set) + 1}
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
        # phase 2: continue evolving the champion under a time/Occam constraint
        # (fitness × 1/(1 + active_weights/budget)); 0 = off
        self.encoder_speed_generations = _num(msg, "encoder_speed_generations", 40, 0, 500)
        self.encoder_time_budget = _num(msg, "encoder_time_budget", 0.5, 0.05, 10.0, float)
        self.routing_layers = (_num(msg, "routing_layers", 2, 0, 8)
                               if msg.get("routing_layers") is not None else None)
        self.notes = str(msg.get("notes") or "tree-of-models text prediction")
        self.emit = emit
        self._stop = threading.Event()
        self.run_id = None
        self.run_dir = None
        self._acc_sum, self._acc_n = 0.0, 0

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
            "encoder_speed_generations": self.encoder_speed_generations,
            "encoder_time_budget": self.encoder_time_budget,
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

    # -- structure ---------------------------------------------------------
    def _tree_manifest(self, root, encoder):
        nodes, total_params = [], encoder.genome_size

        def walk(node, parent):
            nonlocal total_params
            total_params += node.genome_size
            nodes.append({
                "id": node.node_id, "parent": parent, "depth": node.depth,
                "leaf": node.is_leaf, "t0": min(node.token_set),
                "t1": max(node.token_set) + 1, "tokens": len(node.token_set),
            })
            for c in node.children:
                walk(c, node.node_id)

        walk(root, None)
        return nodes, total_params

    # -- GA (mirrors tree_lm._default_evolve, + events, stop, seeding) ------
    def _evolve(self, genome_size, fitness_fn, node_id,
                generations=None, seed_pop=None):
        cfg, rng = self.cfg, self._rng
        generations = generations if generations is not None else cfg.generations
        stride = max(1, generations // 40)
        pop = rng.standard_normal((cfg.pop_size, genome_size)) * 0.1
        if seed_pop is not None:                 # inject structured candidates
            pop[:len(seed_pop)] = seed_pop
        best_score, best_ind = -1.0, pop[0].copy()
        top_k = max(2, int(cfg.pop_size * cfg.elite_frac))

        for gen in range(generations):
            if self.stopped:
                break
            scores = np.array([fitness_fn(ind) for ind in pop])
            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                best_score, best_ind = float(scores[idx]), pop[idx].copy()

            if gen % stride == 0 or gen == generations - 1:
                self.emit({"type": "node_gen", "id": node_id, "gen": gen,
                           "best": best_score, "mean": float(scores.mean())})

            elites = pop[np.argsort(scores)[-top_k:]]
            new_pop = [best_ind.copy()]
            for _ in range(cfg.pop_size - 1):
                p1, p2 = elites[rng.choice(len(elites), 2, replace=False)]
                mask = rng.random(genome_size) > 0.5
                child = np.where(mask, p1, p2)
                if rng.random() < cfg.mutation_rate:
                    scale = np.abs(child) * 0.1 + 0.01   # relative mutation — GENREG style
                    child = child + rng.standard_normal(genome_size) * scale
                new_pop.append(child)
            pop = np.array(new_pop)

        return best_ind

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

        probe = ContextEncoder(cfg)      # scratch instance for fitness evals

        def fitness(genome):
            probe.unpack_genome(genome)
            ctx = probe.encode(Xs)                       # (n, context_dim)
            C = np.zeros((len(classes), ctx.shape[1]))
            np.add.at(C, yi, ctx)
            C /= np.maximum(counts, 1.0)
            scores = ctx @ C.T - 0.5 * np.sum(C * C, axis=1)[None, :]
            return float(np.mean(np.argmax(scores, axis=1) == yi))

        self.emit({"type": "node_start", "id": "encoder", "kind": "encoder",
                   "samples": int(n), "coverage": 100.0,
                   "tokens": cfg.vocab_size})
        heuristic = self._heuristic_encoder_genome()
        seeds = [heuristic, encoder.pack_genome()]
        for _ in range(min(6, cfg.pop_size - 2)):
            seeds.append(heuristic +
                         self._rng.standard_normal(heuristic.size) *
                         (np.abs(heuristic) * 0.1 + 0.01))
        best = self._evolve(encoder.genome_size, fitness, "encoder",
                            generations=self.encoder_generations,
                            seed_pop=np.array(seeds))
        encoder.unpack_genome(best)
        acc = fitness(best)
        self._persist_history("encoder", "encoder", acc)
        self.emit({"type": "node_done", "id": "encoder", "kind": "encoder",
                   "acc": acc, "samples": int(n), "coverage": 100.0})

        # ── phase 2: speed/time constraint ──
        # The champion must now do it *cheaply*: fitness × 1/(1 + active/budget)
        # where active = fraction of genes with |w| > threshold (effective
        # multiplies — for a fixed-shape dense encoder, sparsity IS the
        # evolvable notion of speed). Population is seeded with the phase-1
        # champion plus magnitude-pruned variants of it so the sparse region
        # of the landscape is reachable from generation 0.
        if self.encoder_speed_generations > 0 and not self.stopped:
            THRESH = 0.01
            budget = self.encoder_time_budget

            def active_frac(genome):
                return float(np.mean(np.abs(genome) > THRESH))

            def fitness_speed(genome):
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
                                 seed_pop=np.array(seeds2))
            encoder.unpack_genome(best2)
            acc2, af = fitness(best2), active_frac(best2)
            self._persist_history("encoder-speed", "encoder", acc2)
            self.emit({"type": "node_done", "id": "encoder-speed",
                       "kind": "encoder", "acc": acc2, "samples": int(n),
                       "coverage": 100.0, "active_fraction": af})
            self.emit({"type": "status",
                       "message": (f"encoder speed phase: acc {acc:.3f} → {acc2:.3f} "
                                   f"({acc2-acc:+.3f}), active weights {af*100:.0f}%")})
            acc = acc2

        encoder.freeze()
        return acc

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
        lo, hi = min(node.token_set), max(node.token_set)
        mask = (targets >= lo) & (targets <= hi)   # token ranges are contiguous by construction
        ctx_local, tgt_local = contexts[mask], targets[mask]

        if len(ctx_local) == 0:
            self.emit({"type": "node_done", "id": node.node_id, "acc": None,
                       "samples": 0, "kind": "leaf" if node.is_leaf else "router"})
            node.freeze()
            return

        coverage = len(ctx_local) / total * 100

        if node.is_leaf:
            local_targets = tgt_local - lo
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
                node.unpack_genome(self._evolve(node.genome_size, leaf_fitness,
                                                node.node_id, seed_pop=seeds))
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

        # routing label = which child's contiguous range holds the target
        bounds = np.array([min(c.token_set) for c in node.children] + [hi + 1])
        routing_labels = np.searchsorted(bounds, tgt_local, side="right") - 1

        self.emit({"type": "node_start", "id": node.node_id, "kind": "router",
                   "samples": int(len(ctx_local)), "coverage": coverage,
                   "tokens": len(node.token_set)})

        def router_fitness(genome):
            node.unpack_genome(genome)
            return float(np.mean(node.route(ctx_local) == routing_labels))

        seeds = self._nc_seed(ctx_local, routing_labels, len(node.children))
        node.unpack_genome(self._evolve(node.genome_size, router_fitness,
                                        node.node_id, seed_pop=seeds))
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

        root = build_tree(cfg)
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
            self._evolve_encoder(encoder, X_train, y_train)
            if self.stopped:
                self._persist_finalize("stopped", root=root, encoder=encoder)
                self.emit({"type": "stopped", "trained_nodes": 0,
                           "run_id": self.run_id})
                return

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
