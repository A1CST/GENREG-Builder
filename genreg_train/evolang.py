"""EvoLang — the fresh start after the Tree LM / LM campaign was archived.

WHY THIS EXISTS (see documentation/EVOLANG_PIVOT.md). The old line kept
collapsing into n-gram lookup tables: compressing corpus statistics into a table
is exactly what *gradients* are for, and once we leaned on that the "model" was a
1990s bigram/trigram count table wearing a neural coat. We are not building
attention, we are not building an optimizer, and we are not distilling a table.
We are breeding an organism whose only tool is selection on a fitness landscape.

So EvoLang is deliberately tiny and honest:

  * Corpus   : one small, fixed string of English (below). No external data, no
               token tables, nothing precomputed from counts.
  * Genome   : a minuscule neural predictor. Context = the last K characters,
               each looked up in an *evolved* embedding table (evolved, not
               counted), summed with a learned positional weight, pushed through
               one tanh hidden layer, read out to V character logits. Every
               number in it is a gene. ~a few thousand floats.
  * Fitness  : soft & multiplicative — mean log-softmax probability the genome
               assigns to the *actual* next character over a minibatch of the
               corpus. Never argmax, never "did it get the token right"; we
               reward the whole distribution being shaped like the language.
  * Evolution: tournament selection + elitism, Gaussian mutation with a
               self-adaptive per-genome step (floored so it can't freeze), and
               mandatory energy homeostasis — the least-fit slice each
               generation is starved (culled & replaced by mutated survivors),
               kept in the GENREG 3-15% band.

That is the entire mechanism. No gradient ever touches a weight; the landscape
is the only lever. Streamed over WS /evolang, training survives navigation via
the shared JobHub (same pattern as diffuse_service.JobHub).
"""

import datetime
import hashlib
import json
import os
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")

# --------------------------------------------------------------------------
# The corpus — SWAPPED AGAIN 2026-07-08 (user directive: Wikipedia alone
# lacks real question/exclaim register — an intent-genome probe found every
# emotionally-loaded word, wow/amazing/hooray/alas, was out-of-vocabulary).
# Now a COMBINED corpus: Wikipedia (316MB) + Cornell Movie Dialogs (17.1MB
# dialogue, repeated 6x, ~24.4% of the final 421MB) for real turn-taking,
# real questions, real exclamations (corpora/combined/combined_corpus.txt,
# built by corpora/combined/build_combined_corpus.py). Every genome that
# depends on this corpus was retrained on it, including the intent genomes
# (punctuation-sequence, sent_type, sent_type_exclaim) and a fresh backward
# Order+Selection pair (genreg_train/run_retrain_combined.py, on the I2
# primary — see genomes.txt). Prior corpus history kept below, commented,
# for revert/comparison. We DON'T hold every training window in RAM; the
# flat character-id array is cached once and windows are sampled on the fly
# per generation. The vocabulary is a fixed small charset (not derived from
# the text) so the genome stays tiny: lowercase letters + space + basic
# punctuation, digits folded to '#', everything else folded to space.
# --------------------------------------------------------------------------
# CORPUS_PATH = os.path.join(ROOT, "project", "EEC-main", "engine", "corpus.txt")  # oldest: Gutenberg
# CORPUS_PATH = os.path.join(ROOT, "corpora", "wikipedia", "wiki_corpus.txt")  # prior: Wikipedia only
CORPUS_PATH = os.path.join(ROOT, "corpora", "combined", "combined_corpus.txt")

CHARS = " abcdefghijklmnopqrstuvwxyz.,;:'\"!?-#"
VOCAB = list(CHARS)
V = len(VOCAB)
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
_SPACE = STOI[" "]

# tiny fallback so the module still works if the Gutenberg dump is missing
_FALLBACK = ("the quick brown fox jumps over the lazy dog. "
             "she sells sea shells by the sea shore. ") * 4

_IDS = None            # cached flat char-id array (np.int16), lazily loaded
_IDS_LOCK = threading.Lock()


def _clean(text):
    """Lowercase + normalise smart punctuation to the fixed charset."""
    text = text.lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("—", "-"), ("–", "-"), ("\n", " "), ("\r", " "), ("\t", " ")):
        text = text.replace(a, b)
    return text


def encode(s):
    """Text -> list of char ids (unknown/other chars dropped; digits -> '#')."""
    out = []
    for c in _clean(s):
        if c in STOI:
            out.append(STOI[c])
        elif c.isdigit():
            out.append(STOI["#"])
    return out


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


def _build_ids():
    """Read + map the whole Gutenberg dump to a flat np.int16 id array once.
    Uses a full-unicode lookup table (proven in the archived LM loader): digits
    -> '#', whitespace -> space, anything outside the charset -> space."""
    try:
        with open(CORPUS_PATH, "rb") as fh:
            text = _clean(fh.read().decode("utf-8", errors="replace"))
    except OSError:
        text = _clean(_FALLBACK)
    lut = np.full(1114112, _SPACE, dtype=np.int16)     # default: fold to space
    for i, c in enumerate(CHARS):
        lut[ord(c)] = i
    for d in "0123456789":
        lut[ord(d)] = STOI["#"]
    codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    return lut[codes.astype(np.int64)].astype(np.int16)


def corpus_ids():
    """The cached flat char-id array (loads on first call)."""
    global _IDS
    if _IDS is None:
        with _IDS_LOCK:
            if _IDS is None:
                _IDS = _build_ids()
    return _IDS


def _corpus_preview(n=1600):
    """A short readable slice for the page — reads only a small chunk from disk
    (skipping front-matter) so import stays cheap; never ships the whole 49 MB.
    The full array loads lazily via corpus_ids() on the first training run."""
    try:
        with open(CORPUS_PATH, "rb") as fh:
            fh.seek(40000)                             # skip title-page boilerplate
            raw = fh.read(8000).decode("utf-8", errors="replace")
    except OSError:
        return _clean(_FALLBACK)
    txt = "".join(c for c in _clean(raw) if c in STOI or c.isdigit())
    return txt[:n] or _clean(_FALLBACK)


# CORPUS is a PREVIEW string (for display + generation seeding), not the whole
# corpus — training reads corpus_ids() directly.
CORPUS = _corpus_preview()


# guardrails (clamped from the browser config)
LIM = {
    "context":   (1, 12),
    "hidden":    (2, 64),
    "embed":     (2, 32),
    "pop":       (8, 800),
    "minibatch": (8, 512),
    "max_gens":  (5, 20000),
    "patience":  (10, 5000),
    "tournament": (2, 12),
}


def _clamp(v, lo, hi, default):
    try:
        return int(min(hi, max(lo, int(v))))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Novelty constraint (opt-in landscape lever).
#
# A pure REWARD (never a penalty) that fights the repetitive-collapse failure
# mode — the tiny char models love to settle into "the the and and the". For a
# genome, we sample a short passage and walk its words, maintaining a novelty
# scalar N in [0,1]:
#
#   * each step N decays by `decay` (0.005) — a word's novelty fades "after a
#     while", so it can pay out again later;
#   * a word pays  gain * cooldown_scale  where cooldown_scale ramps 0 -> 1 over
#     `cooldown` words since that word was last used. A word hammered every few
#     steps ("the") is always on cooldown -> ~0; a word not seen for a long time
#     ("normandy") is fully cold -> the full `gain` (0.15);
#   * N only ever gains from words (capped at 1.0) and only ever decays
#     passively — the genome is never penalised, matching "only gained".
#
# The final N is the genome's novelty, added to fitness as  weight * N.
# --------------------------------------------------------------------------
def novelty_scores(out_ids, gain, decay, cooldown):
    """(P,n_chars) sampled ids -> (P,) novelty in [0,1], one per genome."""
    P = out_ids.shape[0]
    cooldown = max(1.0, float(cooldown))
    scores = np.zeros(P, np.float32)
    for p in range(P):
        words = decode(out_ids[p]).split()
        N = 0.0
        last = {}
        for i, w in enumerate(words):
            N = max(0.0, N - decay)                       # passive fade
            since = i - last.get(w, -10 ** 9)             # steps since last use
            scale = min(1.0, since / cooldown)            # cooldown ramp 0->1
            N = min(1.0, N + gain * scale)                # only gained, capped
            last[w] = i
        scores[p] = N
    return scores


# --------------------------------------------------------------------------
# Population of tiny character predictors.
#
# One genome:  context (K ids) -> embed[K,E] -> weighted sum over positions ->
#              tanh(H) -> V logits.  Held as stacked arrays so the whole
#              population is scored in a few einsums.
# --------------------------------------------------------------------------
class LangPop:
    def __init__(self, pop, K, E, H, seed):
        rng = np.random.default_rng(seed)
        self.pop, self.K, self.E, self.H = pop, K, E, H
        self.emb = (rng.standard_normal((pop, V, E)) * 0.3).astype(np.float32)
        self.pos = (rng.standard_normal((pop, K)) * 0.3).astype(np.float32)
        self.W1 = (rng.standard_normal((pop, E, H)) * (1.0 / np.sqrt(E))).astype(np.float32)
        self.b1 = np.zeros((pop, H), np.float32)
        self.W2 = (rng.standard_normal((pop, H, V)) * (1.0 / np.sqrt(H))).astype(np.float32)
        self.b2 = np.zeros((pop, V), np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)     # self-adaptive mutation step

    def _forward_all(self, ctx):
        """ctx (N,K) int ids -> logits (P,N,V) for every genome."""
        # gather embeddings per genome: (P,N,K,E)
        emb = self.emb[:, ctx, :]                          # (P,N,K,E)
        ctxvec = np.einsum("pnke,pk->pne", emb, self.pos)  # positional weighted sum
        h = np.tanh(np.einsum("pne,peh->pnh", ctxvec, self.W1) + self.b1[:, None, :])
        logits = np.einsum("pnh,phv->pnv", h, self.W2) + self.b2[:, None, :]
        return logits

    @staticmethod
    def _log_softmax(logits):
        m = logits.max(axis=-1, keepdims=True)
        z = logits - m
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    def fitness_all(self, ctx, target):
        """Soft multiplicative fitness: mean log-prob of the true next char.
        Returns (P,) — higher (closer to 0) is better."""
        logp = self._log_softmax(self._forward_all(ctx))      # (P,N,V)
        idx = target[None, :, None]
        chosen = np.take_along_axis(logp, np.broadcast_to(idx, (self.pop, len(target), 1)), axis=2)
        return chosen[..., 0].mean(axis=1)                    # (P,)

    def forward_one(self, idx, ctx):
        """One genome, ctx (N,K) -> logits (N,V). For generation."""
        emb = self.emb[idx][ctx]                              # (N,K,E)
        ctxvec = np.einsum("nke,k->ne", emb, self.pos[idx])
        h = np.tanh(ctxvec @ self.W1[idx] + self.b1[idx])
        return h @ self.W2[idx] + self.b2[idx]

    def sample_batch(self, n_chars, seed_ids, rng, temp=0.8):
        """Sample n_chars from EVERY genome at once (P,n_chars) int ids, each
        seeded from the same corpus context. Used only for the novelty
        constraint — one batched forward per character step across the whole
        population, so it stays vectorised (no per-genome Python loop)."""
        P, K = self.pop, self.K
        seed = np.asarray(seed_ids[-K:], np.int64)
        if len(seed) < K:                                     # pad from corpus start
            seed = np.concatenate([np.zeros(K - len(seed), np.int64), seed])
        ctx = np.tile(seed, (P, 1))                           # (P,K)
        rows = np.arange(P)[:, None]
        out = np.empty((P, n_chars), np.int64)
        temp = max(0.05, float(temp))
        for t in range(n_chars):
            emb = self.emb[rows, ctx, :]                      # (P,K,E)
            cvec = np.einsum("pke,pk->pe", emb, self.pos)     # (P,E)
            h = np.tanh(np.einsum("pe,peh->ph", cvec, self.W1) + self.b1)   # (P,H)
            logits = (np.einsum("ph,phv->pv", h, self.W2) + self.b2) / temp  # (P,V)
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            cdf = (e / e.sum(axis=1, keepdims=True)).cumsum(axis=1)
            cdf[:, -1] = 1.0                                  # guard fp rounding
            nxt = (rng.random((P, 1)) < cdf).argmax(axis=1)   # (P,) categorical
            out[:, t] = nxt
            ctx = np.concatenate([ctx[:, 1:], nxt[:, None]], axis=1)
        return out

    def champion(self, idx):
        return (self.emb[idx].copy(), self.pos[idx].copy(), self.W1[idx].copy(),
                self.b1[idx].copy(), self.W2[idx].copy(), self.b2[idx].copy())

    # -- one GA generation: tournament + elitism + self-adaptive mutation + ----
    #    energy homeostasis (starve the least-fit band).
    def evolve_step(self, ctx, target, elite_frac, tourn_k, starve_frac,
                    self_adaptive, mut, rng, nov=None, seed_ids=None):
        base = self.fitness_all(ctx, target)                  # (P,) mean log-prob
        # Novelty constraint: a pure-reward MULTIPLICATIVE boost on the genome's
        # own soft fitness (never a penalty). A genome's probability-space
        # fitness is scaled by (1 + weight * novelty): novelty 0 → ×1 (no boost),
        # novelty 1 → ×(1+weight). Done in log-space via log1p so it stays in the
        # same units as `base` and the boost is always ≥ 0 (only ever gained).
        # Because it multiplies, the absolute lift is proportional to how well
        # the genome already predicts — the boost is scaled to that genome.
        if nov and nov.get("on"):
            out = self.sample_batch(nov["chars"], seed_ids, rng, temp=nov["temp"])
            novval = novelty_scores(out, nov["gain"], nov["decay"], nov["cooldown"])
            fit = base + np.log1p(nov["weight"] * novval)
        else:
            novval = np.zeros(self.pop, np.float32)
            fit = base
        order = np.argsort(fit)[::-1]                          # best first
        n_elite = max(1, int(round(self.pop * elite_frac)))
        n_starve = int(round(self.pop * starve_frac))         # energy homeostasis
        elite = order[:n_elite]

        # survivors eligible to parent = everyone except the starved tail
        alive = order[: self.pop - n_starve] if n_starve > 0 else order

        # children fill the non-elite slots via tournament selection among the alive
        n_child = self.pop - n_elite
        parents = np.empty(n_child, np.int64)
        for i in range(n_child):
            picks = alive[rng.integers(0, len(alive), size=tourn_k)]
            parents[i] = picks[np.argmax(fit[picks])]

        def take(a, sel):
            return a[sel].copy()

        newemb = take(self.emb, elite); newpos = take(self.pos, elite)
        nW1 = take(self.W1, elite); nb1 = take(self.b1, elite)
        nW2 = take(self.W2, elite); nb2 = take(self.b2, elite)
        nsig = take(self.sigma, elite)

        cemb = take(self.emb, parents); cpos = take(self.pos, parents)
        cW1 = take(self.W1, parents); cb1 = take(self.b1, parents)
        cW2 = take(self.W2, parents); cb2 = take(self.b2, parents)
        csig = take(self.sigma, parents)

        if self_adaptive:
            csig = csig * np.exp(0.2 * rng.standard_normal(n_child).astype(np.float32))
            csig = np.clip(csig, 5e-3, 0.4)                   # floor so it can't freeze
            step = csig
        else:
            step = np.full(n_child, np.float32(mut))

        def mutate(arr, s):
            noise = rng.standard_normal(arr.shape).astype(np.float32)
            shape = (len(s),) + (1,) * (arr.ndim - 1)
            return arr + noise * s.reshape(shape)

        cemb = mutate(cemb, step); cpos = mutate(cpos, step)
        cW1 = mutate(cW1, step); cb1 = mutate(cb1, step)
        cW2 = mutate(cW2, step); cb2 = mutate(cb2, step)

        self.emb = np.concatenate([newemb, cemb]); self.pos = np.concatenate([newpos, cpos])
        self.W1 = np.concatenate([nW1, cW1]); self.b1 = np.concatenate([nb1, cb1])
        self.W2 = np.concatenate([nW2, cW2]); self.b2 = np.concatenate([nb2, cb2])
        self.sigma = np.concatenate([nsig, csig])
        # genome 0 is the standing champion (elites came first)
        champ = order[0]
        return {"best": float(fit[champ]), "mean": float(fit.mean()),
                "worst": float(fit[order[-1]]),
                "base": float(base[champ]),          # champion's raw log-prob (for ppl)
                "novelty": float(novval[champ]),     # champion's novelty in [0,1]
                "nov_mean": float(novval.mean()),
                "starved": n_starve,
                "sigma": float(np.median(self.sigma))}


# --------------------------------------------------------------------------
# Global champion — kept so generate() works between/after runs.
# --------------------------------------------------------------------------
_CHAMP = {"g": None, "K": None}
_CHAMP_LOCK = threading.Lock()


def _set_champion(genome, K):
    with _CHAMP_LOCK:
        _CHAMP["g"], _CHAMP["K"] = genome, K


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def generate(prompt="", length=200, temp=0.8, seed=0):
    """Sample from the standing champion. Falls back to a message if untrained."""
    with _CHAMP_LOCK:
        g, K = _CHAMP["g"], _CHAMP["K"]
    if g is None:
        return "(no champion yet — start a run first)"
    emb, pos, W1, b1, W2, b2 = g
    rng = np.random.default_rng(seed or 12345)
    # seed context from the prompt (or the corpus start), padded/truncated to K
    ctx = encode(prompt)[-K:]
    if len(ctx) < K:
        ctx = encode(CORPUS)[:K - len(ctx)] + ctx
    out = []
    temp = max(0.05, float(temp))
    for _ in range(int(length)):
        c = np.array(ctx[-K:], np.int64)
        cvec = np.einsum("ke,k->e", emb[c], pos)
        h = np.tanh(cvec @ W1 + b1)
        logits = (h @ W2 + b2) / temp
        p = _softmax(logits)
        nxt = int(rng.choice(V, p=p))
        out.append(nxt)
        ctx.append(nxt)
    return decode(out)


# --------------------------------------------------------------------------
# Trainer — evolves the population, streams events, honours stop().
# --------------------------------------------------------------------------
class EvoTrainer:
    def __init__(self, msg, emit):
        self.emit = emit
        self._stop = threading.Event()
        self.K        = _clamp(msg.get("context", 5), *LIM["context"], 5)
        self.H        = _clamp(msg.get("hidden", 24), *LIM["hidden"], 24)
        self.E        = _clamp(msg.get("embed", 8), *LIM["embed"], 8)
        self.pop      = _clamp(msg.get("pop", 200), *LIM["pop"], 200)
        self.minibatch = _clamp(msg.get("minibatch", 128), *LIM["minibatch"], 128)
        self.max_gens = _clamp(msg.get("max_gens", 2000), *LIM["max_gens"], 2000)
        self.patience = _clamp(msg.get("patience", 400), *LIM["patience"], 400)
        self.tourn_k  = _clamp(msg.get("tournament", 4), *LIM["tournament"], 4)
        try:
            self.elite = min(0.4, max(0.02, float(msg.get("elite_frac", 0.1))))
        except (TypeError, ValueError):
            self.elite = 0.1
        try:                                     # energy homeostasis band 3-15%
            self.starve = min(0.15, max(0.03, float(msg.get("starve_frac", 0.08))))
        except (TypeError, ValueError):
            self.starve = 0.08
        try:
            self.mut = float(msg.get("mutation", 0.06))
        except (TypeError, ValueError):
            self.mut = 0.06
        self.self_adaptive = bool(msg.get("self_adaptive", True))

        # --- novelty constraint (opt-in, pure reward) -----------------------
        def _f(key, default, lo, hi):
            try:
                return min(hi, max(lo, float(msg.get(key, default))))
            except (TypeError, ValueError):
                return default
        self.nov = {
            "on":       bool(msg.get("novelty", False)),
            "gain":     _f("novelty_gain", 0.15, 0.0, 1.0),
            "decay":    _f("novelty_decay", 0.005, 0.0, 1.0),
            "cooldown": _f("novelty_cooldown", 40, 1, 500),
            "weight":   _f("novelty_weight", 0.5, 0.0, 10.0),
            "chars":    _clamp(msg.get("novelty_chars", 80), 20, 300, 80),
            "temp":     0.8,
        }
        # held-out validation: train samples only from the first (1-holdout) of
        # the corpus; ppl is also measured on windows from the reserved tail so
        # we can see the honest generalisation gap.
        self.holdout = _f("holdout_frac", 0.1, 0.0, 0.5)
        self.seed = _clamp(msg.get("seed", 1234), 0, 2 ** 31 - 1, 1234)
        self.raw = {k: v for k, v in msg.items() if k != "op"}
        self._runlog = None

    def stop(self):
        self._stop.set()

    # ---- run persistence (runs/evolang/<id>/) ------------------------------
    def _persist_start(self, started):
        try:
            ts = datetime.datetime.now()
            stamp = ts.strftime("%Y%m%d-%H%M%S")
            h = hashlib.sha1(json.dumps(self.raw, sort_keys=True, default=str).encode()).hexdigest()[:6]
            rid = f"{stamp}-evolang-{h}"
            d = os.path.join(RUNS_DIR, "evolang", rid)
            os.makedirs(d, exist_ok=True)
            cfg = dict(self.raw)
            cfg.update(population=self.pop, generations=self.max_gens, device="cpu")
            meta = {"id": rid, "environment": "evolang",
                    "created": ts.isoformat(timespec="seconds"),
                    "config": cfg, "started": started, "status": "running"}
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            open(os.path.join(d, "history.jsonl"), "w").close()
            self._runlog = {"id": rid, "dir": d}
        except OSError:
            self._runlog = None

    def _persist_gen(self, ev):
        if not self._runlog:
            return
        rec = {"gen": ev.get("gen"),
               "fitness": {"best": ev.get("best"), "mean": ev.get("mean")}}
        try:
            with open(os.path.join(self._runlog["dir"], "history.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _persist_done(self, done):
        if not self._runlog:
            return
        d = self._runlog["dir"]
        summary = {"id": self._runlog["id"], "environment": "evolang",
                   "status": done.get("reason", "finished"),
                   "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                   "gen": done.get("gen"),
                   "best": {"score": done.get("best"), "sample": done.get("sample")},
                   "checkpoint": None}
        try:
            with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            with open(os.path.join(d, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["status"] = summary["status"]
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except (OSError, ValueError):
            pass

    def run(self):
        try:
            self._run()
        except Exception as exc:                       # pragma: no cover
            self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    def _run(self):
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        Kd = self.K
        ids = corpus_ids()                    # flat char-id array (whole dump)
        n_pos = len(ids) - Kd - 1             # valid window start positions
        # windows are sampled ON THE FLY per generation — we never materialise
        # the ~49M (context -> next char) pairs.
        offs = np.arange(Kd + 1)
        n_train = max(1, int(n_pos * (1.0 - self.holdout)))       # train region
        # fixed held-out validation windows from the reserved tail
        vctx = vtgt = None
        if n_pos - n_train > self.minibatch:
            vrng = np.random.default_rng(self.seed + 99991)
            nval = min(4096, n_pos - n_train)
            vstarts = n_train + vrng.integers(0, n_pos - n_train, size=nval)
            vwin = ids[vstarts[:, None] + offs].astype(np.int64)
            vctx, vtgt = vwin[:, :Kd], vwin[:, Kd]

        seed_ids = ids[:Kd].astype(np.int64)                      # rollout seed
        started = {"vocab": V, "context": Kd, "hidden": self.H, "embed": self.E,
                   "pop": self.pop, "corpus_chars": int(len(ids)),
                   "holdout_frac": self.holdout, "minibatch": self.minibatch,
                   "starve_frac": self.starve, "elite_frac": self.elite,
                   "generations": self.max_gens, "novelty": self.nov["on"],
                   "novelty_cfg": {k: self.nov[k] for k in
                                   ("gain", "decay", "cooldown", "weight", "chars")},
                   "params_per_genome": int(V * self.E + Kd + self.E * self.H
                                            + self.H + self.H * V + V)}
        self._persist_start(started)
        self.emit({"type": "started", **started})

        popn = LangPop(self.pop, Kd, self.E, self.H, self.seed)
        best_ever = -1e9          # best TOTAL fitness (drives champion + plateau)
        best_base = -1e9          # best raw log-prob seen (for the honest ppl tile)
        best_val = -1e9           # best held-out log-prob seen
        val = None
        best_hist = []
        reason = "max_gens"
        gen = 0
        for gen in range(1, self.max_gens + 1):
            if self._stop.is_set():
                reason = "stopped"
                break
            starts = rng.integers(0, n_train, size=self.minibatch)
            win = ids[starts[:, None] + offs].astype(np.int64)     # (mb, K+1)
            ctx, tgt = win[:, :Kd], win[:, Kd]
            stats = popn.evolve_step(ctx, tgt, self.elite,
                                     self.tourn_k, self.starve, self.self_adaptive,
                                     self.mut, rng, nov=self.nov, seed_ids=seed_ids)
            if stats["best"] > best_ever:
                best_ever = stats["best"]
                _set_champion(popn.champion(0), Kd)   # publish champion live
            best_base = max(best_base, stats["base"])
            best_hist.append(stats["best"])
            # held-out ppl every 100 gens (genome 0 = standing champion post-sort)
            if vctx is not None and (gen % 100 == 0 or gen == 1):
                val = float(popn.fitness_all(vctx, vtgt)[0])
                best_val = max(best_val, val)
            # ppl comes from the RAW log-prob (base), never the novelty bonus
            ev = {"type": "gen", "gen": gen, "generations": self.max_gens,
                  "best": round(stats["best"], 5), "mean": round(stats["mean"], 5),
                  "best_ever": round(best_ever, 5),
                  "base": round(stats["base"], 5), "base_ever": round(best_base, 5),
                  "novelty": round(stats["novelty"], 4),
                  "nov_mean": round(stats["nov_mean"], 4),
                  "ppl": round(float(np.exp(-stats["base"])), 3),
                  "val_ppl": (round(float(np.exp(-val)), 3) if val is not None else None),
                  "starved": stats["starved"], "sigma": round(stats["sigma"], 4)}
            self._persist_gen(ev)
            self.emit(ev)
            # periodic live sample so the page shows the language emerging
            if gen % 25 == 0 or gen == 1:
                self.emit({"type": "sample", "gen": gen,
                           "text": generate("the ", 120, temp=0.7, seed=gen)})
            # plateau early-stop
            if len(best_hist) > self.patience:
                window = best_hist[-self.patience:]
                if (max(window) - max(best_hist[:-self.patience])) < 1e-4:
                    reason = "plateau"
                    break

        # final held-out eval on the standing champion
        if vctx is not None:
            best_val = max(best_val, float(popn.fitness_all(vctx, vtgt)[0]))
        final_sample = generate("the ", 200, temp=0.7, seed=7)
        done = {"type": "done",
                "reason": "stopped" if self._stop.is_set() else reason,
                "gen": gen, "best": round(best_ever, 5),
                "base": round(best_base, 5), "novelty_on": self.nov["on"],
                "ppl": round(float(np.exp(-best_base)), 3),
                "val_ppl": (round(float(np.exp(-best_val)), 3) if best_val > -1e8 else None),
                "sample": final_sample, "seconds": round(time.time() - t0, 1)}
        self._persist_done(done)
        self.emit(done)


# --------------------------------------------------------------------------
# Job hub — reuse the exact pattern from diffuse_service so training survives
# page navigation and end-of-run posts to the Agent panel.
# --------------------------------------------------------------------------
from genreg_train.diffuse_service import JobHub   # noqa: E402

HUB = JobHub(program="evolang")
