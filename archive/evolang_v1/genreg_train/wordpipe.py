"""WordPipe — the specialist-pipeline experiment (user's vision, 2026-07-06).

Not one genome that "does English." Decompose language into components and evolve
a specialist per component, each with a clean, unambiguous survival condition —
"the constraint IS the genome's entire reason for existing." This module builds
and gates the first three specialists, in order, so each is proven before the
next is built on it:

  GATE 1 — VOCABULARY (the speller). A char genome rewarded for emitting REAL
           words (lexicon coverage), scaffolded by char-prediction so it can
           bootstrap. Claim: raises word-validity above a plain char LM.

  GATE 2 — ORDER DISCRIMINATOR (the grammaticality signal). A genome evolved to
           tell REAL corpus word-order from SHUFFLED word-order. This is the
           "metaphorical, not statistical" grammar signal — a landscape, NOT an
           n-gram transition table. Claim: beats 50% on held-out windows.

  GATE 3 — ORDERER. A word-level generator over the valid vocab, scored by the
           FROZEN discriminator's "this looks like English" verdict (adversarial),
           NOT by next-word accuracy (which would rebuild the n-gram table).
           Claim: its generated word-order scores above random order.

Everything gradient-free: tournament + elitism + energy homeostasis + self-
adaptive mutation, numpy inference only. Reuses EvoLang's char substrate.
"""
import collections
import os
import time

import numpy as np

from genreg_train import evolang
from genreg_train.evolang import corpus_ids, decode, V as CHAR_V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Lexicon + word corpus (built from the corpus itself — a vocabulary, not a
# statistical model; carries no grammar or transition information).
# --------------------------------------------------------------------------
_LEX = None
_WORDCACHE = {}


def build_lexicon(min_count=8, min_len=2, max_len=15):
    """The set of valid words = tokens that appear >= min_count times in the
    corpus (filters OCR junk/typos). Cached. Returns (lexicon set, Counter)."""
    global _LEX
    if _LEX is None:
        text = decode(corpus_ids())
        counts = collections.Counter(text.split())
        lex = {w for w, c in counts.items()
               if c >= min_count and min_len <= len(w) <= max_len and w.isalpha()}
        _LEX = (lex, counts)
    return _LEX


def build_word_corpus(vocab_n=2000):
    """Word-tokenised corpus over the top `vocab_n` words. id 0 = <unk>.
    Returns (word_ids int32, vocab list, stoi). Cached per vocab_n."""
    if vocab_n in _WORDCACHE:
        return _WORDCACHE[vocab_n]
    lex, counts = build_lexicon()
    top = [w for w, _ in counts.most_common() if w in lex][:vocab_n]
    vocab = ["<unk>"] + top
    stoi = {w: i for i, w in enumerate(vocab)}
    text = decode(corpus_ids())
    ids = np.fromiter((stoi.get(w, 0) for w in text.split()), dtype=np.int32)
    _WORDCACHE[vocab_n] = (ids, vocab, stoi)
    return _WORDCACHE[vocab_n]


# --------------------------------------------------------------------------
# Shared GA mechanics (tournament + elitism + energy homeostasis + self-adaptive
# mutation). Works on a dict of stacked param arrays so every specialist reuses
# it — the machinery is identical, only the fitness (the reason to exist) differs.
# --------------------------------------------------------------------------
def ga_step(params, fit, rng, elite_frac=0.1, tourn_k=4, starve_frac=0.08):
    """One generation, in place on `params` (dict name -> (P, ...) array).
    `fit` is (P,) higher=better. Returns the champion index (0 after the step —
    elites are packed first). sigma is params['sigma'] (P,) self-adaptive step."""
    P = len(fit)
    order = np.argsort(fit)[::-1]
    n_elite = max(1, int(round(P * elite_frac)))
    n_starve = int(round(P * starve_frac))
    elite = order[:n_elite]
    alive = order[:P - n_starve] if n_starve > 0 else order
    n_child = P - n_elite
    parents = np.empty(n_child, np.int64)
    for i in range(n_child):
        picks = alive[rng.integers(0, len(alive), size=tourn_k)]
        parents[i] = picks[np.argmax(fit[picks])]

    sigma = params["sigma"]
    csig = sigma[parents] * np.exp(0.2 * rng.standard_normal(n_child).astype(np.float32))
    csig = np.clip(csig, 5e-3, 0.4)
    new = {}
    for name, arr in params.items():
        if name == "sigma":
            new["sigma"] = np.concatenate([sigma[elite].copy(), csig])
            continue
        keep = arr[elite].copy()
        child = arr[parents].copy()
        shape = (n_child,) + (1,) * (arr.ndim - 1)
        child += rng.standard_normal(child.shape).astype(np.float32) * csig.reshape(shape)
        new[name] = np.concatenate([keep, child])
    params.clear()
    params.update(new)
    return int(order[0])


def _softmax_last(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ==========================================================================
# GATE 1 — VOCABULARY genome (the speller)
# ==========================================================================
def coverage_scores(out_char_ids, lex, min_len=2):
    """(P, L) emitted char ids -> (P,) coverage = fraction of emitted characters
    that fall inside a REAL word (per the genome's own spacing). Punishes gibberish
    and trivial 1-char output; rewards densely emitting valid words. Order/repeat
    are NOT judged here — that's the orderer's job."""
    P = out_char_ids.shape[0]
    out = np.zeros(P, np.float32)
    for p in range(P):
        s = decode(out_char_ids[p])
        if not s:
            continue
        good = sum(len(t) for t in s.split() if len(t) >= min_len and t in lex)
        out[p] = good / len(s)
    return out


def run_speller(gens=1500, pop=200, K=4, H=48, E=12, minibatch=256,
                cov_weight=1.0, seed=1234, log=print):
    """Evolve a char genome whose reason to exist is emitting REAL words.
    Fitness = char-logprob (dense scaffold, lets it bootstrap) boosted
    multiplicatively by lexicon coverage: base + log1p(cov_weight * coverage).
    cov_weight=0 recovers a plain char LM (the control). Returns a result dict."""
    lex, _ = build_lexicon()
    ids = corpus_ids()
    n_pos = len(ids) - K - 1
    n_train = int(n_pos * 0.9)
    offs = np.arange(K + 1)
    seed_ids = ids[:K].astype(np.int64)
    rng = np.random.default_rng(seed)
    popn = evolang.LangPop(pop, K, E, H, seed)

    best_cov, best_base = 0.0, -1e9
    for gen in range(1, gens + 1):
        starts = rng.integers(0, n_train, size=minibatch)
        win = ids[starts[:, None] + offs].astype(np.int64)
        ctx, tgt = win[:, :K], win[:, K]
        base = popn.fitness_all(ctx, tgt)                       # (P,) char logprob
        if cov_weight > 0:
            gen_ids = popn.sample_batch(160, seed_ids, rng, temp=0.8)
            cov = coverage_scores(gen_ids, lex)
            fit = base + np.log1p(cov_weight * cov)
        else:
            cov = np.zeros(pop, np.float32)
            fit = base
        pdict = {"emb": popn.emb, "pos": popn.pos, "W1": popn.W1,
                 "b1": popn.b1, "W2": popn.W2, "b2": popn.b2, "sigma": popn.sigma}
        champ = ga_step(pdict, fit, rng)
        (popn.emb, popn.pos, popn.W1, popn.b1, popn.W2, popn.b2, popn.sigma) = (
            pdict["emb"], pdict["pos"], pdict["W1"], pdict["b1"],
            pdict["W2"], pdict["b2"], pdict["sigma"])
        best_base = max(best_base, float(base[champ]))
        best_cov = max(best_cov, float(cov[champ]))
        if gen % 250 == 0 or gen == 1:
            log(f"  [speller w{cov_weight}] gen {gen}: base={base[champ]:.3f} "
                f"cov={cov[champ]:.3f} best_cov={best_cov:.3f}")
    # champion sample + validity measure
    gen_ids = popn.sample_batch(400, seed_ids, rng, temp=0.7)
    sample = decode(gen_ids[0])
    toks = sample.split()
    valid = [t for t in toks if len(t) >= 2 and t in lex]
    pct_valid = len(valid) / max(1, len(toks))
    return {"cov_weight": cov_weight, "best_cov": round(best_cov, 4),
            "pct_valid_tokens": round(pct_valid, 4), "sample": sample[:200]}


# ==========================================================================
# Word-window helpers (for gates 2 & 3)
# ==========================================================================
def sample_windows(word_ids, n, W, rng, max_unk=1):
    """n windows of W consecutive words, rejecting unk-heavy windows so the
    order signal isn't drowned by <unk>. Returns (n, W) int64."""
    keep = np.empty((n, W), np.int64)
    got = 0
    while got < n:
        need = (n - got) * 2
        starts = rng.integers(0, len(word_ids) - W, size=need)
        cand = word_ids[starts[:, None] + np.arange(W)].astype(np.int64)
        ok = cand[(cand == 0).sum(axis=1) <= max_unk]
        take = min(len(ok), n - got)
        keep[got:got + take] = ok[:take]
        got += take
    return keep


def shuffle_rows(win, rng):
    """Permute the word order WITHIN each row (bag-of-words preserved, order
    destroyed) — so a discriminator must learn ORDER, not word identity."""
    keys = rng.random(win.shape)
    return np.take_along_axis(win, np.argsort(keys, axis=1), axis=1)


# ==========================================================================
# GATE 2 — ORDER DISCRIMINATOR (the grammaticality signal)
# ==========================================================================
class DiscPop:
    """Genome: word-window -> real/fake. emb(P,Vw,E) then a W*E -> H -> 1 MLP.
    Because fakes are within-window shuffles of reals, the ONLY way to win is to
    read word ORDER — this is a learned grammaticality landscape, not an n-gram
    transition table."""

    def __init__(self, pop, Vw, W, E, H, seed):
        rng = np.random.default_rng(seed)
        self.pop, self.Vw, self.W, self.E, self.H = pop, Vw, W, E, H
        self.emb = (rng.standard_normal((pop, Vw, E)) * 0.1).astype(np.float32)
        self.W1 = (rng.standard_normal((pop, W * E, H)) * (1 / np.sqrt(W * E))).astype(np.float32)
        self.b1 = np.zeros((pop, H), np.float32)
        self.W2 = (rng.standard_normal((pop, H, 1)) * (1 / np.sqrt(H))).astype(np.float32)
        self.b2 = np.zeros((pop, 1), np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def logits(self, win):                    # win (N,W) -> (P,N)
        e = self.emb[:, win, :]               # (P,N,W,E)
        P, N = e.shape[0], e.shape[1]
        flat = e.reshape(P, N, self.W * self.E)
        h = np.tanh(np.einsum("pnf,pfh->pnh", flat, self.W1) + self.b1[:, None, :])
        return np.einsum("pnh,pho->pn", h, self.W2) + self.b2

    def fit_acc(self, win, lab):              # lab (N,) 0/1 -> (fit(P,), acc(P,))
        p = 1.0 / (1.0 + np.exp(-self.logits(win)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        bce = -(lab[None] * np.log(p) + (1 - lab[None]) * np.log(1 - p)).mean(axis=1)
        acc = ((p > 0.5) == lab[None].astype(bool)).mean(axis=1)
        return -bce, acc

    def fit_margin(self, win, lab):
        """DENSE fitness: mean logit on REAL minus mean logit on SHUFFLED. Unlike
        accuracy (flat — every random genome ~0.5), this rewards ANY separation
        of the two pools, so evolution has a smooth gradient to climb from random
        init. Also returns held-out-style accuracy for reporting."""
        z = self.logits(win)                  # (P,N)
        real = z[:, lab.astype(bool)].mean(axis=1)
        fake = z[:, ~lab.astype(bool)].mean(axis=1)
        p = 1.0 / (1.0 + np.exp(-z))
        acc = ((p > 0.5) == lab[None].astype(bool)).mean(axis=1)
        return real - fake, acc

    def champion(self, idx):
        return (self.emb[idx].copy(), self.W1[idx].copy(), self.b1[idx].copy(),
                self.W2[idx].copy(), self.b2[idx].copy())


def disc_score(champ, win):
    """Frozen champion's P(real) on windows (N,W) -> (N,)."""
    emb, W1, b1, W2, b2 = champ
    e = emb[win]                              # (N,W,E)
    flat = e.reshape(len(win), -1)
    h = np.tanh(flat @ W1 + b1)
    return 1.0 / (1.0 + np.exp(-((h @ W2)[:, 0] + b2[0])))


def run_disc_on(sym_ids, n_sym, gens=1500, pop=200, W=5, E=10, H=48,
                minibatch=256, seed=1234, label="disc", fitness="margin", log=print):
    """Core: evolve a real-vs-shuffled ORDER discriminator over an arbitrary
    symbol stream `sym_ids` (word ids, or class ids) with `n_sym` symbols. Two
    levers per GENREG: SHRINK the space (small n_sym) AND shape the fitness
    (`margin` = dense/graded, climbs from random; `acc` = flat, doesn't).
    GATE: held-out accuracy must beat 50%."""
    n_train = int(len(sym_ids) * 0.9)
    rng = np.random.default_rng(seed)
    popn = DiscPop(pop, n_sym, W, E, H, seed)
    vr = np.random.default_rng(seed + 7)
    vreal = sample_windows(sym_ids[n_train:], 2048, W, vr, max_unk=W)  # classes: no unk filter
    vwin = np.concatenate([vreal, shuffle_rows(vreal, vr)])
    vlab = np.concatenate([np.ones(len(vreal)), np.zeros(len(vreal))]).astype(np.float32)
    fit_fn = popn.fit_margin if fitness == "margin" else popn.fit_acc

    best_acc, best_champ = 0.5, None
    for gen in range(1, gens + 1):
        real = sample_windows(sym_ids[:n_train], minibatch // 2, W, rng, max_unk=W)
        win = np.concatenate([real, shuffle_rows(real, rng)])
        lab = np.concatenate([np.ones(len(real)), np.zeros(len(real))]).astype(np.float32)
        fit, _ = fit_fn(win, lab)
        pdict = {"emb": popn.emb, "W1": popn.W1, "b1": popn.b1,
                 "W2": popn.W2, "b2": popn.b2, "sigma": popn.sigma}
        ga_step(pdict, fit, rng)
        (popn.emb, popn.W1, popn.b1, popn.W2, popn.b2, popn.sigma) = (
            pdict["emb"], pdict["W1"], pdict["b1"], pdict["W2"], pdict["b2"], pdict["sigma"])
        if gen % 100 == 0 or gen == 1:
            _, vacc = popn.fit_acc(vwin, vlab)
            if float(vacc[0]) > best_acc:
                best_acc = float(vacc[0]); best_champ = popn.champion(0)
            log(f"  [{label}] gen {gen}: val_acc={vacc[0]:.3f} best={best_acc:.3f}")
    n_params = n_sym * E + W * E * H + H + H + 1
    return {"best_val_acc": round(best_acc, 4), "champ": best_champ,
            "n_sym": n_sym, "emb_params": n_sym * E, "n_params": int(n_params), "W": W}


def run_discriminator(gens=1500, pop=200, vocab_n=4000, W=5, E=10, H=48,
                      minibatch=256, seed=1234, log=print):
    """Word-level order discriminator (the ballooned version — vocab_n symbols)."""
    ids, vocab, _ = build_word_corpus(vocab_n)
    r = run_disc_on(ids, len(vocab), gens, pop, W, E, H, minibatch, seed, "disc-word", log)
    r.update(vocab=vocab, vocab_n=vocab_n)
    return r


# --------------------------------------------------------------------------
# Category induction — collapse 4000 words to ~30 distributional classes so the
# order specialist's search space shrinks by ~100x (embedding n_class x E). This
# is HOW GENREG shrinks the space: syntax is categorical, so order over classes
# is the same signal in a space small enough for evolution to climb.
# --------------------------------------------------------------------------
_CLASSCACHE = {}


def induce_word_classes(n_classes=32, n_anchors=40, vocab_n=4000, seed=0):
    """Cluster the vocab into n_classes by their left/right anchor-word context
    (unsupervised distributional POS induction). Returns (word2class (Vw,),
    class_ids corpus, n_classes). Cached."""
    key = (n_classes, n_anchors, vocab_n)
    if key in _CLASSCACHE:
        return _CLASSCACHE[key]
    ids, vocab, _ = build_word_corpus(vocab_n)
    Vw = len(vocab)
    c, l, r = ids[1:-1], ids[:-2], ids[2:]
    feat = np.zeros((Vw, 2 * n_anchors), np.float32)
    ml = (l >= 1) & (l <= n_anchors)
    np.add.at(feat, (c[ml], l[ml] - 1), 1.0)               # left-neighbour anchor
    mr = (r >= 1) & (r <= n_anchors)
    np.add.at(feat, (c[mr], n_anchors + r[mr] - 1), 1.0)   # right-neighbour anchor
    feat /= (feat.sum(1, keepdims=True) + 1e-6)            # row-normalise
    # simple k-means over the REAL words only — <unk> (id 0) is ~1/3 of the corpus
    # and, left in, forms a mega-class the order genome over-emits and selection
    # fills with the class's top real words ("more"/"good"). Give it its own class.
    rng = np.random.default_rng(seed)
    real = np.arange(1, Vw)
    fr = feat[real]
    cen = fr[rng.choice(len(real), n_classes, replace=False)].copy()
    rl = np.zeros(len(real), np.int64)
    for _ in range(30):
        d = ((fr[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        rl = d.argmin(1)
        for j in range(n_classes):
            m = rl == j
            if m.any():
                cen[j] = fr[m].mean(0)
    word2class = np.zeros(Vw, np.int64)
    word2class[real] = rl
    word2class[0] = n_classes                              # reserved <unk> class
    nc = n_classes + 1
    class_ids = word2class[ids]
    _CLASSCACHE[key] = (word2class, class_ids, nc, vocab)
    return _CLASSCACHE[key]


def run_cat_discriminator(n_classes=32, gens=1500, pop=200, W=5, E=8, H=48,
                          seed=1234, log=print):
    """Order discriminator over induced CATEGORIES (the shrunk search space)."""
    word2class, class_ids, nc, vocab = induce_word_classes(n_classes)
    r = run_disc_on(class_ids, nc, gens, pop, W, E, H, 256, seed,
                    f"disc-cat{nc}", log)
    r.update(n_classes=nc, mode="category")
    return r


# ==========================================================================
# GATE 3 — ORDERER (word generator scored by the frozen discriminator)
# ==========================================================================
class OrderPop:
    """Word-level generator: prev C words -> next word over Vw. Same shape as the
    char substrate but over the valid-word vocab."""

    def __init__(self, pop, Vw, C, E, H, seed):
        rng = np.random.default_rng(seed)
        self.pop, self.Vw, self.C, self.E, self.H = pop, Vw, C, E, H
        self.emb = (rng.standard_normal((pop, Vw, E)) * 0.1).astype(np.float32)
        self.pos = (rng.standard_normal((pop, C)) * 0.3).astype(np.float32)
        self.W1 = (rng.standard_normal((pop, E, H)) * (1 / np.sqrt(E))).astype(np.float32)
        self.b1 = np.zeros((pop, H), np.float32)
        self.W2 = (rng.standard_normal((pop, H, Vw)) * (1 / np.sqrt(H))).astype(np.float32)
        self.b2 = np.zeros((pop, Vw), np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def gen_windows(self, W, seed_ctx, rng, temp=0.9):
        """Each genome autoregressively emits W words -> (P, W) word ids."""
        P, C = self.pop, self.C
        ctx = np.tile(np.asarray(seed_ctx[-C:], np.int64), (P, 1))
        rows = np.arange(P)[:, None]
        out = np.empty((P, W), np.int64)
        for t in range(W):
            cvec = np.einsum("pce,pc->pe", self.emb[rows, ctx, :], self.pos)
            h = np.tanh(np.einsum("pe,peh->ph", cvec, self.W1) + self.b1)
            logits = (np.einsum("ph,phv->pv", h, self.W2) + self.b2) / temp
            cdf = _softmax_last(logits).cumsum(axis=1)
            cdf[:, -1] = 1.0
            nxt = (rng.random((P, 1)) < cdf).argmax(axis=1)
            out[:, t] = nxt
            ctx = np.concatenate([ctx[:, 1:], nxt[:, None]], axis=1)
        return out

    def fitness_all(self, ctx, tgt):
        """Predictive fitness: mean log-prob of the true next symbol. DENSE and
        graded per position — the climbable landscape shape (same as the char/word
        LM), unlike the holistic real/fake discriminator bit."""
        emb = self.emb[:, ctx, :]                              # (P,N,C,E)
        cvec = np.einsum("pnce,pc->pne", emb, self.pos)
        h = np.tanh(np.einsum("pne,peh->pnh", cvec, self.W1) + self.b1[:, None, :])
        logits = np.einsum("pnh,phv->pnv", h, self.W2) + self.b2[:, None, :]
        m = logits.max(-1, keepdims=True)
        z = logits - m
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        idx = tgt[None, :, None]
        ch = np.take_along_axis(logp, np.broadcast_to(idx, (self.pop, len(tgt), 1)), axis=2)
        return ch[..., 0].mean(1)

    def champion(self, idx):
        return (self.emb[idx].copy(), self.pos[idx].copy(), self.W1[idx].copy(),
                self.b1[idx].copy(), self.W2[idx].copy(), self.b2[idx].copy())


def run_class_lm(n_classes=32, gens=800, pop=200, C=4, E=8, H=48, minibatch=256,
                 seed=1234, log=print):
    """The order/syntax specialist as a PREDICTOR: evolve a next-CLASS model
    (dense log-prob fitness) over induced POS-like classes. Reports held-out
    class perplexity vs the class-unigram baseline — beating unigram = it learned
    categorical word order (syntax) gradient-free."""
    w2c, cids, nc, vocab = induce_word_classes(n_classes)
    n_train = int(len(cids) * 0.9)
    offs = np.arange(C + 1)
    rng = np.random.default_rng(seed)
    popn = OrderPop(pop, nc, C, E, H, seed)
    # class-unigram baseline perplexity on held-out
    cnt = np.bincount(cids[:n_train], minlength=nc).astype(np.float64)
    pu = np.clip(cnt / cnt.sum(), 1e-9, 1.0)
    vr = np.random.default_rng(seed + 5)
    vs = vr.integers(0, len(cids) - n_train - C - 1, size=4096) + n_train
    vwin = cids[vs[:, None] + offs].astype(np.int64)
    vctx, vtgt = vwin[:, :C], vwin[:, C]
    uni_ppl = float(np.exp(-np.log(pu[vtgt]).mean()))

    best_val = -1e9
    for gen in range(1, gens + 1):
        starts = rng.integers(0, n_train - C - 1, size=minibatch)
        win = cids[starts[:, None] + offs].astype(np.int64)
        fit = popn.fitness_all(win[:, :C], win[:, C])
        pdict = {"emb": popn.emb, "pos": popn.pos, "W1": popn.W1, "b1": popn.b1,
                 "W2": popn.W2, "b2": popn.b2, "sigma": popn.sigma}
        ga_step(pdict, fit, rng)
        (popn.emb, popn.pos, popn.W1, popn.b1, popn.W2, popn.b2, popn.sigma) = (
            pdict["emb"], pdict["pos"], pdict["W1"], pdict["b1"],
            pdict["W2"], pdict["b2"], pdict["sigma"])
        if gen % 100 == 0 or gen == 1:
            v = float(popn.fitness_all(vctx, vtgt)[0])
            best_val = max(best_val, v)
            log(f"  [class-lm {nc}] gen {gen}: val_ppl={np.exp(-v):.3f} "
                f"(unigram {uni_ppl:.3f})")
    return {"n_classes": nc, "val_ppl": round(float(np.exp(-best_val)), 3),
            "unigram_ppl": round(uni_ppl, 3),
            "beats_unigram": float(np.exp(-best_val)) < uni_ppl,
            "champ": popn.champion(0), "C": C, "E": E, "H": H}


# ==========================================================================
# GATE 3 — COMPOSE the two proven specialists into text
#   order genome  -> emits a CLASS skeleton (grammatical order)
#   vocabulary    -> fills each class slot with a real word of that class
# ==========================================================================
def gen_class_seq(champ, C, length, seed_ctx, rng, temp=0.9, reranks=None, clause=None):
    """Autoregressively emit `length` CLASS ids from the order-genome champion.
    `reranks` = list of (classfeat[nc,K], M[K,K], gamma): pairwise constraint genomes
    (content-function alternation, agreement, …) lifted to per-class centroids in
    their own feature space, applied as a bias on the next-CLASS logits. Biases sum.
    `clause` = (B[nc]*Kc, Kc, gamma): a clause-template genome expanded to a validity
    bias tensor — given the last Kc-1 classes it biases the next class toward those
    that COMPLETE a real clause fragment. Both put the constraint on the ORDER skeleton
    itself, not just word selection within a slot."""
    emb, pos, W1, b1, W2, b2 = champ
    abias = None
    if reranks:
        abias = sum(gamma * (CF @ M @ CF.T) for CF, M, gamma in reranks)   # (nc, nc)[prev, cand]
    ctx = list(np.asarray(seed_ctx[-C:], np.int64))
    out = []
    for _ in range(length):
        c = np.asarray(ctx[-C:], np.int64)
        cvec = np.einsum("ce,c->e", emb[c], pos)
        h = np.tanh(cvec @ W1 + b1)
        logits = h @ W2 + b2
        if abias is not None:
            logits = logits + abias[ctx[-1]]
        if clause is not None:
            B, Kc, cg = clause
            if len(ctx) >= Kc - 1:
                b = B[tuple(int(x) for x in ctx[-(Kc - 1):])]   # (nc,) validity of each completion
                logits = logits + cg * (b - b.mean())           # zero-mean = pure rerank
        logits = logits / max(0.05, temp)
        p = _softmax_last(logits)
        nxt = int(rng.choice(len(p), p=p))
        out.append(nxt); ctx.append(nxt)
    return out


def build_class_words(n_classes, vocab_n=4000, cap=300):
    """class id -> (member word-ids, freq-probs) from corpus counts, unk excluded.
    The vocabulary component's knowledge: which real words belong to each class."""
    w2c, cids, nc, vocab = induce_word_classes(n_classes)
    ids, _, _ = build_word_corpus(vocab_n)
    counts = np.bincount(ids, minlength=len(vocab)).astype(np.float64)
    table = {}
    for cl in range(nc):
        members = [w for w in range(1, len(vocab)) if w2c[w] == cl]   # skip unk(0)
        if not members:
            continue
        members = sorted(members, key=lambda w: -counts[w])[:cap]
        fr = np.array([counts[w] for w in members], np.float64)
        table[cl] = (np.array(members), fr / fr.sum())
    return table, w2c, vocab, nc, cids


def _fill(seq_classes, table, rng):
    """Fill a class sequence with freq-weighted real words -> word-id list."""
    out = []
    for cl in seq_classes:
        if cl in table:
            m, fr = table[cl]
            out.append(int(rng.choice(m, p=fr)))
    return out


def run_gate3(n_classes=32, order_gens=2500, C=4, seed=1234, log=print):
    """Compose order genome + vocabulary. Decisive comparison — both use the SAME
    real-word fillers, so any difference is due to CLASS ORDER alone:
      * pipeline  : order genome picks the class sequence
      * unigram   : classes drawn by marginal frequency (no order genome)
      * real      : actual corpus (upper reference)
    Metric: fraction of adjacent word-pairs that occur in the real corpus
    (a local 'looks like English' score, used for EVALUATION only)."""
    log(f"training order genome ({n_classes} classes, C={C}, {order_gens} gens)…")
    res = run_class_lm(n_classes=n_classes, gens=order_gens, pop=200, C=C,
                       E=10, H=64, seed=seed, log=log)
    champ = res["champ"]
    table, w2c, vocab, nc, cids = build_class_words(n_classes)
    ids, _, _ = build_word_corpus(4000)
    rng = np.random.default_rng(seed + 1)

    # real-corpus adjacent-word-pair set (evaluation metric only)
    bigset = set(zip(ids[:-1].tolist(), ids[1:].tolist()))

    def hit(wseq):
        if len(wseq) < 2:
            return 0.0
        return float(np.mean([(wseq[i], wseq[i + 1]) in bigset
                              for i in range(len(wseq) - 1)]))

    def to_text(wseq):
        return " ".join(vocab[w] for w in wseq)

    # class marginal (for the unigram baseline)
    cfreq = np.bincount(cids, minlength=nc).astype(np.float64)
    cprob = cfreq / cfreq.sum()

    N = 400
    seed_ctx = cids[500:500 + C]
    pipe_hits, uni_hits, real_hits = [], [], []
    samples = {}
    for k in range(20):
        r2 = np.random.default_rng(seed + 100 + k)
        pipe_cls = gen_class_seq(champ, C, N, seed_ctx, r2, temp=0.8)
        pipe_w = _fill(pipe_cls, table, r2)
        uni_cls = list(r2.choice(nc, size=N, p=cprob))
        uni_w = _fill(uni_cls, table, r2)
        rstart = int(r2.integers(0, len(ids) - N))
        real_w = ids[rstart:rstart + N].tolist()
        pipe_hits.append(hit(pipe_w)); uni_hits.append(hit(uni_w)); real_hits.append(hit(real_w))
        if k == 0:
            samples = {"pipeline": to_text(pipe_w[:40]),
                       "unigram": to_text(uni_w[:40]),
                       "real": to_text([w for w in real_w[:40] if w != 0])}
    out = {"order_val_ppl": res["val_ppl"], "order_unigram_ppl": res["unigram_ppl"],
           "adj_pair_hit": {"pipeline": round(float(np.mean(pipe_hits)), 4),
                            "unigram_baseline": round(float(np.mean(uni_hits)), 4),
                            "real_corpus": round(float(np.mean(real_hits)), 4)},
           "samples": samples}
    log(f"adj-pair hit — pipeline {out['adj_pair_hit']['pipeline']} | "
        f"unigram {out['adj_pair_hit']['unigram_baseline']} | "
        f"real {out['adj_pair_hit']['real_corpus']}")
    log(f"  PIPELINE: {samples['pipeline']}")
    log(f"  UNIGRAM : {samples['unigram']}")
    log(f"  REAL    : {samples['real']}")
    return out


# ==========================================================================
# GATE 4 — WORD-SELECTION specialist (fill a class slot with the word that fits
# the neighbours, not a class-random one). Applies both lessons: the 4000-word
# representation is FIXED (distributional features, out of the search space —
# only a tiny bilinear head evolves) and the fitness is DENSE predictive
# (log-prob of the TRUE word among its class-mates via in-class negative
# sampling). Learns collocation ("strong tea", not "powerful tea") gradient-free.
# ==========================================================================
_FEATCACHE = {}


def word_features(vocab_n=4000, D=24, n_anchors=60):
    """Fixed D-dim distributional feature per word (SVD of left/right anchor
    co-occurrence). FIXED = not evolved, so the big representation is out of the
    search space. Returns (feat (Vw,D) float32, vocab)."""
    key = (vocab_n, D, n_anchors)
    if key in _FEATCACHE:
        return _FEATCACHE[key]
    ids, vocab, _ = build_word_corpus(vocab_n)
    Vw = len(vocab)
    c, l, r = ids[1:-1], ids[:-2], ids[2:]
    F = np.zeros((Vw, 2 * n_anchors), np.float64)
    ml = (l >= 1) & (l <= n_anchors)
    np.add.at(F, (c[ml], l[ml] - 1), 1.0)
    mr = (r >= 1) & (r <= n_anchors)
    np.add.at(F, (c[mr], n_anchors + r[mr] - 1), 1.0)
    F = np.log1p(F)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    emb = U[:, :D] * S[:D]
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    _FEATCACHE[key] = (emb.astype(np.float32), vocab)
    return _FEATCACHE[key]


class WordSelPop:
    """Genome = a bilinear compatibility head M (D x D) + a frequency-prior weight
    beta. score(prev, cand) = feat_prev^T M feat_cand + beta*logfreq(cand).
    ~D*D + 1 params — tiny, climbable."""

    def __init__(self, pop, D, seed):
        rng = np.random.default_rng(seed)
        self.pop, self.D = pop, D
        self.M = (rng.standard_normal((pop, D, D)) * (1.0 / np.sqrt(D))).astype(np.float32)
        self.beta = np.full(pop, 0.5, np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def scores(self, pf, cf, clogf):
        """pf (N,D) prev feats, cf (N,K1,D) candidate feats, clogf (N,K1) cand
        log-freq -> (P,N,K1) scores."""
        pm = np.einsum("nd,pde->pne", pf, self.M)          # (P,N,D)
        s = np.einsum("pne,nke->pnk", pm, cf)              # (P,N,K1)
        return s + self.beta[:, None, None] * clogf[None]

    def fitness(self, pf, cf, clogf):
        """Dense: mean log-prob of the TRUE candidate (index 0) among class-mates."""
        s = self.scores(pf, cf, clogf)
        m = s.max(-1, keepdims=True)
        z = s - m
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        return logp[:, :, 0].mean(1)                       # (P,)

    def champion(self, idx):
        return (self.M[idx].copy(), float(self.beta[idx]))


def _sel_pool(ids, table, w2c, feat, logfreq, n_pos, K, rng):
    """Precompute a fixed pool of (prev, [true + K in-class negatives]) positions
    once, so per-gen sampling is just array indexing. Returns prev_ids (n,),
    cand_ids (n,K+1) with the true word at column 0."""
    prev = np.empty(n_pos, np.int64)
    cand = np.empty((n_pos, K + 1), np.int64)
    got = 0
    while got < n_pos:
        idx = rng.integers(1, len(ids) - 1, size=(n_pos - got) * 3)
        for i in idx:
            pv, nx = int(ids[i]), int(ids[i + 1])
            if pv == 0 or nx == 0:
                continue
            cl = w2c[nx]
            if cl not in table:
                continue
            mem = table[cl][0]
            if len(mem) < K + 1:
                continue
            negs = rng.choice(mem, size=K, replace=False)
            prev[got] = pv
            cand[got, 0] = nx
            cand[got, 1:] = negs
            got += 1
            if got >= n_pos:
                break
    return prev, cand


def run_selection(n_classes=32, gens=1500, pop=200, D=24, K=7, minibatch=512,
                  seed=1234, log=print):
    """Evolve the word-selection specialist. GATE: held-out log-prob (and top-1-
    among-candidates accuracy) of the true word must beat the frequency baseline
    (which ignores the previous word)."""
    feat, vocab = word_features(4000, D)
    logfreq = np.log1p(np.bincount(build_word_corpus(4000)[0],
                                   minlength=len(vocab)).astype(np.float32))
    table, w2c, _, nc, _ = build_class_words(n_classes)
    ids, _, _ = build_word_corpus(4000)
    n_train = int(len(ids) * 0.9)
    rng = np.random.default_rng(seed)
    log("building position pools…")
    tp_prev, tp_cand = _sel_pool(ids[:n_train], table, w2c, feat, logfreq, 60000, K, rng)
    vp_prev, vp_cand = _sel_pool(ids[n_train:], table, w2c, feat, logfreq, 8000, K,
                                 np.random.default_rng(seed + 5))
    # frequency baseline held-out log-prob (score = beta*logfreq only, i.e. M=0)
    vcf, vclf = feat[vp_cand], logfreq[vp_cand]
    base_s = vclf                                          # pure frequency
    bm = base_s.max(-1, keepdims=True)
    base_lp = float((base_s - bm - np.log(np.exp(base_s - bm).sum(-1, keepdims=True)))[:, 0].mean())
    base_acc = float((vclf.argmax(1) == 0).mean())

    popn = WordSelPop(pop, D, seed)
    best_val, best_champ = -1e9, None
    for gen in range(1, gens + 1):
        sel = rng.integers(0, len(tp_prev), size=minibatch)
        pf, cf, clf = feat[tp_prev[sel]], feat[tp_cand[sel]], logfreq[tp_cand[sel]]
        fit = popn.fitness(pf, cf, clf)
        pdict = {"M": popn.M, "beta": popn.beta, "sigma": popn.sigma}
        ga_step(pdict, fit, rng)
        popn.M, popn.beta, popn.sigma = pdict["M"], pdict["beta"], pdict["sigma"]
        if gen % 100 == 0 or gen == 1:
            vf = float(popn.fitness(feat[vp_prev], vcf, vclf)[0])
            if vf > best_val:
                best_val = vf; best_champ = popn.champion(0)
            # top-1 accuracy of the champion
            s = popn.scores(feat[vp_prev], vcf, vclf)[0]
            acc = float((s.argmax(1) == 0).mean())
            log(f"  [select] gen {gen}: val_logprob={vf:.4f} (base {base_lp:.4f}) "
                f"top1={acc:.3f} (base {base_acc:.3f})")
    return {"val_logprob": round(best_val, 4), "base_logprob": round(base_lp, 4),
            "beats_freq_baseline": best_val > base_lp,
            "champ": best_champ, "D": D, "K": K}


def _apply_reranks(s, prev_w, mem, reranks):
    """Add each pairwise re-rank genome's score to the selection scores in place.
    `reranks` = list of (feats[Vw,K], M[K,K], gamma): agreement, content-function
    alternation, … Each adds gamma * feats[prev] @ M @ feats[mem]^T."""
    if reranks:
        for Ag, aM, gamma in reranks:
            s = s + gamma * ((Ag[prev_w] @ aM) @ Ag[mem].T)
    return s


def _fill_selected(prev_w, cl, table, feat, logfreq, champ, rng, temp=0.7, reranks=None, bonus=None):
    """Fill a class slot using the selection specialist: score class members by
    compatibility with the previous emitted word, sample from the softmax.
    `reranks`: optional list of pairwise genomes (agreement, alternation, semantic)
    that re-rank candidates by their relation to the previous word — see _apply_reranks.
    `bonus`: optional per-candidate additive score (aligned with table[cl][0]) for
    stateful constraints the caller computes — e.g. the repetition penalty."""
    M, beta = champ
    mem = table[cl][0]
    s = (feat[prev_w] @ M) @ feat[mem].T + beta * logfreq[mem]
    s = _apply_reranks(s, prev_w, mem, reranks)
    if bonus is not None:
        s = s + bonus
    s = s / max(0.05, temp)
    s -= s.max()
    p = np.exp(s); p /= p.sum()
    return int(rng.choice(mem, p=p))


def run_gate34(n_classes=32, order_gens=2500, sel_gens=1500, C=4, seed=1234, log=print):
    """Full 3-specialist pipeline: ORDER genome -> class skeleton; SELECTION genome
    -> fills each slot with a neighbour-compatible word (vs class-random). Measures
    whether context-aware selection raises local English-likeness."""
    log("=== training ORDER genome ===")
    ores = run_class_lm(n_classes=n_classes, gens=order_gens, pop=200, C=C,
                        E=10, H=64, seed=seed, log=log)
    log("=== training SELECTION genome ===")
    sres = run_selection(n_classes=n_classes, gens=sel_gens, pop=200, D=24, K=7,
                         seed=seed, log=log)
    ochamp, schamp = ores["champ"], sres["champ"]
    table, w2c, vocab, nc, cids = build_class_words(n_classes)
    feat, _ = word_features(4000, 24)
    ids, _, _ = build_word_corpus(4000)
    logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
    bigset = set(zip(ids[:-1].tolist(), ids[1:].tolist()))

    def hit(ws):
        return float(np.mean([(ws[i], ws[i + 1]) in bigset for i in range(len(ws) - 1)])) if len(ws) > 1 else 0.0

    rnd_hits, sel_hits = [], []
    samples = {}
    N = 400
    for k in range(20):
        r2 = np.random.default_rng(seed + 200 + k)
        cls_seq = gen_class_seq(ochamp, C, N, cids[500:500 + C], r2, temp=0.8)
        # class-random fill (Gate-3 pipeline)
        rnd_w = _fill(cls_seq, table, r2)
        # selection fill (Gate-4): condition each pick on the previous emitted word
        sel_w, prev = [], int(ids[499])
        for cl in cls_seq:
            if cl not in table:
                continue
            w = _fill_selected(prev, cl, table, feat, logfreq, schamp, r2)
            sel_w.append(w); prev = w
        rnd_hits.append(hit(rnd_w)); sel_hits.append(hit(sel_w))
        if k == 0:
            samples = {"class_random_fill": " ".join(vocab[w] for w in rnd_w[:40]),
                       "selection_fill": " ".join(vocab[w] for w in sel_w[:40])}
    out = {"order_val_ppl": ores["val_ppl"],
           "select_val_logprob": sres["val_logprob"], "select_base": sres["base_logprob"],
           "adj_pair_hit": {"class_random_fill": round(float(np.mean(rnd_hits)), 4),
                            "selection_fill": round(float(np.mean(sel_hits)), 4)},
           "samples": samples}
    log(f"adj-pair hit — class-random {out['adj_pair_hit']['class_random_fill']} "
        f"-> selection {out['adj_pair_hit']['selection_fill']}")
    log(f"  RANDOM-FILL   : {samples['class_random_fill']}")
    log(f"  SELECTION-FILL: {samples['selection_fill']}")
    return out


# ==========================================================================
# GATE 5 — PUNCTUATION / SENTENCE-BOUNDARY specialist. The output is currently
# one endless run-on. This genome decides, per position, whether a sentence ends
# — dense per-position binary prediction (log-prob of the true boundary), tiny
# space (class embed + sentence-position). Turns the stream into sentences.
# ==========================================================================
_BOUNDCACHE = {}


def build_boundary_corpus(n_classes=32, vocab_n=4000):
    """Word stream + per-position (class, sentence-position, is-boundary). Same
    tokenisation as build_word_corpus (so it lines up with the trained genomes);
    boundary = the token carried trailing . ! or ?. Cached."""
    key = (n_classes, vocab_n)
    if key in _BOUNDCACHE:
        return _BOUNDCACHE[key]
    _, vocab, stoi = build_word_corpus(vocab_n)
    w2c, _, nc, _ = induce_word_classes(n_classes)
    text = decode(corpus_ids())
    toks = text.split()
    ids = np.fromiter((stoi.get(t, 0) for t in toks), np.int32, len(toks))
    bound = np.fromiter((1 if t and t[-1] in ".!?" else 0 for t in toks), np.int8, len(toks))
    cls = w2c[ids].astype(np.int64)
    posn = np.empty(len(bound), np.float32)
    c = 0
    for i in range(len(bound)):
        posn[i] = c
        c = 0 if bound[i] else c + 1
    _BOUNDCACHE[key] = (ids, cls, bound.astype(np.float32), posn, nc)
    return _BOUNDCACHE[key]


class BoundaryPop:
    """Genome: (class, sentence-position) -> P(sentence ends here). Tiny."""

    def __init__(self, pop, nc, E, H, seed):
        rng = np.random.default_rng(seed)
        self.pop, self.nc, self.E, self.H = pop, nc, E, H
        self.emb = (rng.standard_normal((pop, nc, E)) * 0.3).astype(np.float32)
        self.W1 = (rng.standard_normal((pop, E + 1, H)) * (1 / np.sqrt(E + 1))).astype(np.float32)
        self.b1 = np.zeros((pop, H), np.float32)
        self.W2 = (rng.standard_normal((pop, H, 1)) * (1 / np.sqrt(H))).astype(np.float32)
        self.b2 = np.zeros((pop, 1), np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def logit(self, cls, posn):                            # (N,),(N,) -> (P,N)
        e = self.emb[:, cls, :]                            # (P,N,E)
        x = np.concatenate([e, np.broadcast_to(posn / 20.0, (self.pop, len(posn)))[..., None]], -1)
        h = np.tanh(np.einsum("pnf,pfh->pnh", x, self.W1) + self.b1[:, None, :])
        return np.einsum("pnh,pho->pn", h, self.W2) + self.b2

    def fitness(self, cls, posn, y):
        p = 1.0 / (1.0 + np.exp(-np.clip(self.logit(cls, posn), -30, 30)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return (y[None] * np.log(p) + (1 - y[None]) * np.log(1 - p)).mean(1)   # (P,) mean logprob

    def champion(self, idx):
        return (self.emb[idx].copy(), self.W1[idx].copy(), self.b1[idx].copy(),
                self.W2[idx].copy(), self.b2[idx].copy())


def boundary_prob(champ, cl, pos):
    """Frozen champion's P(boundary) for a single (class, position)."""
    emb, W1, b1, W2, b2 = champ
    x = np.concatenate([emb[cl], [pos / 20.0]])
    h = np.tanh(x @ W1 + b1)
    return float(1.0 / (1.0 + np.exp(-np.clip((h @ W2)[0] + b2[0], -30, 30))))


# ---- COMMA / internal-punctuation specialist (same shape as BoundaryPop) -----
_COMMACACHE = {}


def build_comma_corpus(n_classes=32, vocab_n=4000):
    """Word stream + per-position (class, clause-position, is-comma). clause-
    position resets on any comma/clause punctuation, so 'deeper into the clause'
    is a feature. Same tokenisation as build_word_corpus. Cached."""
    key = (n_classes, vocab_n)
    if key in _COMMACACHE:
        return _COMMACACHE[key]
    _, vocab, stoi = build_word_corpus(vocab_n)
    w2c, _, nc, _ = induce_word_classes(n_classes)
    toks = decode(corpus_ids()).split()
    ids = np.fromiter((stoi.get(t, 0) for t in toks), np.int32, len(toks))
    comma = np.fromiter((1 if t and t[-1] == "," else 0 for t in toks), np.int8, len(toks))
    reset = np.fromiter((1 if t and t[-1] in ",.!?;:" else 0 for t in toks), np.int8, len(toks))
    cls = w2c[ids].astype(np.int64)
    posn = np.empty(len(comma), np.float32)
    c = 0
    for i in range(len(comma)):
        posn[i] = c
        c = 0 if reset[i] else c + 1
    _COMMACACHE[key] = (ids, cls, comma.astype(np.float32), posn, nc)
    return _COMMACACHE[key]


def run_comma(n_classes=32, gens=1000, pop=200, E=8, H=32, minibatch=1024,
              seed=1234, log=print):
    """Evolve the comma specialist (reuses BoundaryPop). GATE: held-out log-prob
    beats the base-rate baseline. Uses `comma_prob` (== boundary_prob) at gen."""
    ids, cls, comma, posn, nc = build_comma_corpus(n_classes)
    n_train = int(len(ids) * 0.9)
    rng = np.random.default_rng(seed)
    popn = BoundaryPop(pop, nc, E, H, seed)
    base = float(comma[:n_train].mean())
    vsel = rng.integers(0, len(ids) - n_train, size=8192) + n_train
    vcls, vpos, vy = cls[vsel], posn[vsel], comma[vsel]
    base_lp = float((vy * np.log(base) + (1 - vy) * np.log(1 - base)).mean())
    best_val, best_champ = -1e9, None
    for gen in range(1, gens + 1):
        s = rng.integers(0, n_train, size=minibatch)
        fit = popn.fitness(cls[s], posn[s], comma[s])
        pd = {"emb": popn.emb, "W1": popn.W1, "b1": popn.b1, "W2": popn.W2, "b2": popn.b2, "sigma": popn.sigma}
        ga_step(pd, fit, rng)
        (popn.emb, popn.W1, popn.b1, popn.W2, popn.b2, popn.sigma) = (
            pd["emb"], pd["W1"], pd["b1"], pd["W2"], pd["b2"], pd["sigma"])
        if gen % 100 == 0 or gen == 1:
            v = float(popn.fitness(vcls, vpos, vy)[0])
            if v > best_val:
                best_val = v; best_champ = popn.champion(0)
            log(f"  [comma] gen {gen}: val_logprob={v:.4f} (base {base_lp:.4f})")
    return {"val_logprob": round(best_val, 4), "base_logprob": round(base_lp, 4),
            "beats_baseline": best_val > base_lp, "champ": best_champ,
            "comma_rate": round(base, 4)}


def run_boundary(n_classes=32, gens=1200, pop=200, E=8, H=32, minibatch=1024,
                 seed=1234, log=print):
    """Evolve the sentence-boundary specialist. GATE: held-out log-prob must beat
    the base-rate baseline (predicting the constant boundary frequency)."""
    ids, cls, bound, posn, nc = build_boundary_corpus(n_classes)
    n_train = int(len(ids) * 0.9)
    rng = np.random.default_rng(seed)
    popn = BoundaryPop(pop, nc, E, H, seed)
    base = float(bound[:n_train].mean())                   # boundary rate
    vsel = rng.integers(0, len(ids) - n_train, size=8192) + n_train
    vcls, vpos, vy = cls[vsel], posn[vsel], bound[vsel]
    base_lp = float((vy * np.log(base) + (1 - vy) * np.log(1 - base)).mean())

    best_val, best_champ = -1e9, None
    for gen in range(1, gens + 1):
        s = rng.integers(0, n_train, size=minibatch)
        fit = popn.fitness(cls[s], posn[s], bound[s])
        pdict = {"emb": popn.emb, "W1": popn.W1, "b1": popn.b1,
                 "W2": popn.W2, "b2": popn.b2, "sigma": popn.sigma}
        ga_step(pdict, fit, rng)
        (popn.emb, popn.W1, popn.b1, popn.W2, popn.b2, popn.sigma) = (
            pdict["emb"], pdict["W1"], pdict["b1"], pdict["W2"], pdict["b2"], pdict["sigma"])
        if gen % 100 == 0 or gen == 1:
            v = float(popn.fitness(vcls, vpos, vy)[0])
            if v > best_val:
                best_val = v; best_champ = popn.champion(0)
            log(f"  [bound] gen {gen}: val_logprob={v:.4f} (base {base_lp:.4f})")
    return {"val_logprob": round(best_val, 4), "base_logprob": round(base_lp, 4),
            "beats_baseline": best_val > base_lp, "champ": best_champ,
            "boundary_rate": round(base, 4), "E": E}


def run_gate5(n_classes=32, order_gens=3000, sel_gens=2500, bound_gens=1000,
              C=4, seed=1234, log=print):
    """The full 4-specialist pipeline: ORDER (class skeleton) + SELECTION (context
    word choice) + BOUNDARY (sentence segmentation). Generates real sentences and
    checks the sentence-length distribution against the corpus (~18.6 words)."""
    log("=== ORDER genome ===")
    ores = run_class_lm(n_classes=n_classes, gens=order_gens, pop=200, C=C, E=10, H=64, seed=seed, log=log)
    log("=== SELECTION genome ===")
    sres = run_selection(n_classes=n_classes, gens=sel_gens, pop=200, D=24, K=7, seed=seed, log=log)
    log("=== BOUNDARY genome ===")
    bres = run_boundary(n_classes=n_classes, gens=bound_gens, pop=200, seed=seed, log=log)

    table, w2c, vocab, nc, cids = build_class_words(n_classes)
    feat, _ = word_features(4000, 24)
    ids, _, _ = build_word_corpus(4000)
    logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
    ochamp, schamp, bchamp = ores["champ"], sres["champ"], bres["champ"]

    rng = np.random.default_rng(seed + 9)
    text_parts, sent_lens = [], []
    cur = 0
    cls_seq = gen_class_seq(ochamp, C, 1600, cids[500:500 + C], rng, temp=0.8)
    prev = int(ids[499])
    for cl in cls_seq:
        if cl not in table:
            continue
        w = _fill_selected(prev, cl, table, feat, logfreq, schamp, rng); prev = w
        text_parts.append(vocab[w])
        cur += 1
        if rng.random() < boundary_prob(bchamp, cl, cur) or cur >= 45:
            text_parts.append(".")
            sent_lens.append(cur); cur = 0
    text = " ".join(text_parts).replace(" .", ".")
    # capitalise sentence starts for readability
    import re
    text = re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    real_mean = float(1.0 / build_boundary_corpus(n_classes)[2].mean())
    out = {"order_val_ppl": ores["val_ppl"], "select_val_logprob": sres["val_logprob"],
           "boundary_val_logprob": bres["val_logprob"],
           "gen_sent_len_mean": round(float(np.mean(sent_lens)), 1) if sent_lens else None,
           "real_sent_len_mean": round(real_mean, 1),
           "sample": text[:600]}
    log(f"gen sentence-len mean {out['gen_sent_len_mean']} vs real {out['real_sent_len_mean']}")
    log(f"SAMPLE:\n{out['sample']}")
    return out


# ==========================================================================
# GATE 6 — CONTEXT genome (recurrent evolved state = MEMORY of what was said).
# The local genomes have no memory (order=last few classes, selection=last word).
# This carries a running state h_t = tanh(A h_{t-1} + B f(w_t)) over the whole
# emitted stream and outputs a query into feature space that reweights the next
# word. Trained on next-word prediction, so it learns the corpus's actual (non-
# repetitive) distribution — repetition drops because the model finally matches
# real text, not via a decode-time patch. Fixed word features keep the space
# small; only A, B, U (+beta) evolve (~1700 params). Dense per-step fitness.
# ==========================================================================
class ContextPop:
    def __init__(self, pop, D, S, seed):
        rng = np.random.default_rng(seed)
        self.pop, self.D, self.S = pop, D, S
        self.A = (rng.standard_normal((pop, S, S)) * (0.5 / np.sqrt(S))).astype(np.float32)
        self.B = (rng.standard_normal((pop, D, S)) * (1 / np.sqrt(D))).astype(np.float32)
        self.U = (rng.standard_normal((pop, S, D)) * (1 / np.sqrt(S))).astype(np.float32)
        self.beta = np.full(pop, 0.5, np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def fitness(self, sf, cf, clf):
        """sf (B,L,D) sequence feats; cf (B,L-1,K1,D) candidate feats (true@0);
        clf (B,L-1,K1) cand log-freq -> (P,) mean log-prob of true next word."""
        P = self.pop; B, L, _ = sf.shape
        h = np.zeros((P, B, self.S), np.float32)
        tot = np.zeros((P, B), np.float32)
        for t in range(L - 1):
            h = np.tanh(np.einsum("pbs,pst->pbt", h, self.A)
                        + np.einsum("bd,pds->pbs", sf[:, t], self.B))
            q = np.einsum("pbs,psd->pbd", h, self.U)               # (P,B,D)
            s = np.einsum("pbd,bkd->pbk", q, cf[:, t]) + self.beta[:, None, None] * clf[:, t][None]
            m = s.max(-1, keepdims=True)
            lp = s - m - np.log(np.exp(s - m).sum(-1, keepdims=True))
            tot += lp[:, :, 0]
        return (tot / (L - 1)).mean(1)

    def champion(self, idx):
        return (self.A[idx].copy(), self.B[idx].copy(), self.U[idx].copy(), float(self.beta[idx]))


# ==========================================================================
# TRACK A — BIDIRECTIONAL SELECTION. Selection saw only the previous word. Word
# choice depends on BOTH neighbours ("the ___ ran"). Here: score a candidate
# against the previous WORD (left) and the next CLASS (right — known from the
# order skeleton at generation). Purely local (±1). Two bilinear heads (~1150
# params), dense predictive fitness. A direct upgrade to the proven selector.
# ==========================================================================
def class_centroids(n_classes=32, D=24):
    """Fixed mean feature vector per class (the 'shape' of the next slot)."""
    feat, vocab = word_features(4000, D)
    w2c, _, nc, _ = induce_word_classes(n_classes)
    C = np.zeros((nc, D), np.float32)
    for cl in range(nc):
        members = np.where(w2c[:len(vocab)] == cl)[0]
        if len(members):
            C[cl] = feat[members].mean(0)
    return C


class BiSelPop:
    def __init__(self, pop, D, seed):
        rng = np.random.default_rng(seed)
        self.pop, self.D = pop, D
        self.ML = (rng.standard_normal((pop, D, D)) * (1 / np.sqrt(D))).astype(np.float32)
        self.MR = (rng.standard_normal((pop, D, D)) * (1 / np.sqrt(D))).astype(np.float32)
        self.beta = np.full(pop, 0.5, np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def scores(self, pf, cf, ncf, clf):
        """pf (N,D) prev-word feat; cf (N,K1,D) cand feat; ncf (N,D) next-class
        centroid; clf (N,K1) cand log-freq -> (P,N,K1)."""
        left = np.einsum("pne,nke->pnk", np.einsum("nd,pde->pne", pf, self.ML), cf)
        qR = np.einsum("pde,ne->pnd", self.MR, ncf)
        right = np.einsum("nkd,pnd->pnk", cf, qR)
        return left + right + self.beta[:, None, None] * clf[None]

    def fitness(self, pf, cf, ncf, clf):
        s = self.scores(pf, cf, ncf, clf)
        m = s.max(-1, keepdims=True)
        lp = s - m - np.log(np.exp(s - m).sum(-1, keepdims=True))
        return lp[:, :, 0].mean(1)

    def champion(self, idx):
        return (self.ML[idx].copy(), self.MR[idx].copy(), float(self.beta[idx]))


def _bisel_pool(ids, table, w2c, n_pos, K, rng):
    """Positions with (prev word, next class, [true + K in-class negatives])."""
    prev = np.empty(n_pos, np.int64)
    ncls = np.empty(n_pos, np.int64)
    cand = np.empty((n_pos, K + 1), np.int64)
    got = 0
    while got < n_pos:
        for i in rng.integers(1, len(ids) - 2, size=(n_pos - got) * 3):
            pv, w, nx = int(ids[i - 1]), int(ids[i]), int(ids[i + 1])
            if pv == 0 or w == 0:
                continue
            cl = w2c[w]
            if cl not in table or len(table[cl][0]) < K + 1:
                continue
            prev[got] = pv; ncls[got] = w2c[nx]
            cand[got, 0] = w; cand[got, 1:] = rng.choice(table[cl][0], size=K, replace=False)
            got += 1
            if got >= n_pos:
                break
    return prev, ncls, cand


def run_biselection(n_classes=32, gens=1500, pop=200, D=24, K=7, minibatch=512,
                    seed=1234, log=print):
    """Evolve bidirectional selection. GATE: held-out log-prob beats the uni-
    directional (prev-word-only) selector."""
    feat, vocab = word_features(4000, D)
    logfreq = np.log1p(np.bincount(build_word_corpus(4000)[0], minlength=len(vocab)).astype(np.float32))
    cents = class_centroids(n_classes, D)
    table, w2c, _, nc, _ = build_class_words(n_classes)
    ids, _, _ = build_word_corpus(4000)
    n_train = int(len(ids) * 0.9)
    rng = np.random.default_rng(seed)
    log("building bidirectional pools…")
    tp_prev, tp_nc, tp_cand = _bisel_pool(ids[:n_train], table, w2c, 60000, K, rng)
    vp_prev, vp_nc, vp_cand = _bisel_pool(ids[n_train:], table, w2c, 8000, K, np.random.default_rng(seed + 5))
    vcf, vclf, vncf = feat[vp_cand], logfreq[vp_cand], cents[vp_nc]
    bs = vclf; bm = bs.max(-1, keepdims=True)
    base_lp = float((bs - bm - np.log(np.exp(bs - bm).sum(-1, keepdims=True)))[:, 0].mean())

    popn = BiSelPop(pop, D, seed)
    best_val, best_champ = -1e9, None
    for gen in range(1, gens + 1):
        s = rng.integers(0, len(tp_prev), size=minibatch)
        fit = popn.fitness(feat[tp_prev[s]], feat[tp_cand[s]], cents[tp_nc[s]], logfreq[tp_cand[s]])
        pdict = {"ML": popn.ML, "MR": popn.MR, "beta": popn.beta, "sigma": popn.sigma}
        ga_step(pdict, fit, rng)
        popn.ML, popn.MR, popn.beta, popn.sigma = pdict["ML"], pdict["MR"], pdict["beta"], pdict["sigma"]
        if gen % 100 == 0 or gen == 1:
            v = float(popn.fitness(feat[vp_prev], vcf, vncf, vclf)[0])
            if v > best_val:
                best_val = v; best_champ = popn.champion(0)
            log(f"  [bisel] gen {gen}: val_logprob={v:.4f} (freq base {base_lp:.4f})")
    return {"val_logprob": round(best_val, 4), "base_logprob": round(base_lp, 4),
            "champ": best_champ, "D": D}


def _fill_bisel(prev_w, cl, next_cls, table, feat, logfreq, cents, champ, rng, temp=0.7,
                reranks=None, bonus=None):
    ML, MR, beta = champ
    mem = table[cl][0]
    cf = feat[mem]
    s = (feat[prev_w] @ ML) @ cf.T + cf @ (MR @ cents[next_cls]) + beta * logfreq[mem]
    s = _apply_reranks(s, prev_w, mem, reranks)
    if bonus is not None:
        s = s + bonus
    s = s / temp; s -= s.max(); p = np.exp(s); p /= p.sum()
    return int(rng.choice(mem, p=p))


def run_gate7(n_classes=32, order_gens=2000, sel_gens=1500, C=4, seed=1234, log=print):
    """Compare unidirectional vs BIDIRECTIONAL selection on the same order
    skeleton. Metric: adjacent-word-pair corpus-hit (local English-likeness)."""
    import collections
    ores = run_class_lm(n_classes=n_classes, gens=order_gens, pop=200, C=C, E=10, H=64, seed=seed, log=log)
    log("=== unidirectional selection ===")
    ures = run_selection(n_classes=n_classes, gens=sel_gens, pop=200, D=24, K=7, seed=seed, log=log)
    log("=== BIDIRECTIONAL selection ===")
    bres = run_biselection(n_classes=n_classes, gens=sel_gens, pop=200, D=24, K=7, seed=seed, log=log)
    table, w2c, vocab, nc, cids = build_class_words(n_classes)
    feat, _ = word_features(4000, 24)
    cents = class_centroids(n_classes, 24)
    ids, _, _ = build_word_corpus(4000)
    logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
    bigset = set(zip(ids[:-1].tolist(), ids[1:].tolist()))

    def measure(kind):
        rng = np.random.default_rng(seed + 9)
        cls_seq = gen_class_seq(ores["champ"], C, 1500, cids[500:500 + C], rng, temp=0.8)
        words, prev = [], int(ids[499])
        for j, cl in enumerate(cls_seq):
            if cl not in table:
                continue
            nxt = next((cls_seq[k] for k in range(j + 1, len(cls_seq)) if cls_seq[k] in table), cl)
            if kind == "uni":
                w = _fill_selected(prev, cl, table, feat, logfreq, ures["champ"], rng)
            else:
                w = _fill_bisel(prev, cl, nxt, table, feat, logfreq, cents, bres["champ"], rng)
            words.append(w); prev = w
        cnt = collections.Counter(words)
        adj = float(np.mean([(words[i], words[i + 1]) in bigset for i in range(len(words) - 1)]))
        return {"adj_pair_hit": round(adj, 4), "distinct_ratio": round(len(cnt) / len(words), 4),
                "sample": " ".join(vocab[w] for w in words[:45])}

    uni, bi = measure("uni"), measure("bi")
    log(f"uni-selection : adj {uni['adj_pair_hit']} distinct {uni['distinct_ratio']}")
    log(f"  {uni['sample']}")
    log(f"BIDIRECTIONAL : adj {bi['adj_pair_hit']} distinct {bi['distinct_ratio']}")
    log(f"  {bi['sample']}")
    return {"uni_val_logprob": ures["val_logprob"], "bi_val_logprob": bres["val_logprob"],
            "uni": uni, "bi": bi}


# ==========================================================================
# TRACK A — CHUNK / PHRASE genome. English is formulaic: much of local fluency
# is fixed collocations ("of the", "he was", "in the world"). Instead of placing
# every word independently, emit frequent REAL phrases as atomic units when the
# order skeleton's upcoming class pattern matches a real phrase's class pattern.
# The phrase's internal adjacencies are 100% real by construction — instant local
# fluency. A phrase LEXICON (like the word lexicon), indexed by class-tuple.
# ==========================================================================
_CHUNKCACHE = {}


def build_chunk_index(n_classes=32, vocab_n=4000, min_count=25):
    """Frequent 2- & 3-word phrases (non-unk), indexed by their class-tuple.
    Returns {L: {class_tuple: (chunk_ids array (m,L), prob array (m,))}}."""
    key = (n_classes, vocab_n, min_count)
    if key in _CHUNKCACHE:
        return _CHUNKCACHE[key]
    ids, vocab, _ = build_word_corpus(vocab_n)
    w2c, _, nc, _ = induce_word_classes(n_classes)
    V = len(vocab)
    index = {}
    for L in (2, 3):
        cols = [ids[i:len(ids) - (L - 1 - i)] for i in range(L)]     # aligned columns
        good = np.ones(len(cols[0]), bool)
        for c in cols:
            good &= (c != 0)
        code = np.zeros(len(cols[0]), np.int64)
        for c in cols:
            code = code * V + c
        code = code[good]
        uniq, cnt = np.unique(code, return_counts=True)
        keep = uniq[cnt >= min_count]; kc = cnt[cnt >= min_count]
        # decode back to L word ids
        words = np.empty((len(keep), L), np.int64)
        rem = keep.copy()
        for i in range(L - 1, -1, -1):
            words[:, i] = rem % V; rem //= V
        buckets = {}
        for row, c in zip(words, kc):
            ct = tuple(int(w2c[w]) for w in row)
            buckets.setdefault(ct, []).append((row, int(c)))
        index[L] = {ct: (np.array([r for r, _ in v]),
                         np.array([c for _, c in v], np.float64) / sum(c for _, c in v))
                    for ct, v in buckets.items()}
    _CHUNKCACHE[key] = index
    return index


def gen_chunked(ochamp, C, n, seed_ctx, table, feat, logfreq, cents, schamp,
                chunk_index, vocab, w2c, rng, chunk_prob=0.6, bidir=True):
    """Generate over the class skeleton, emitting a real PHRASE whenever the
    upcoming class pattern matches one (prob chunk_prob), else a single selected
    word. Returns a list of word ids."""
    cls_seq = gen_class_seq(ochamp, C, n, seed_ctx, rng, temp=0.8)
    words, prev, j = [], None, 0
    while j < len(cls_seq):
        cl = cls_seq[j]
        if cl not in table:
            j += 1; continue
        hit = False
        for L in (3, 2):
            if j + L <= len(cls_seq):
                ct = tuple(int(x) for x in cls_seq[j:j + L])
                bucket = chunk_index.get(L, {}).get(ct)
                if bucket is not None and rng.random() < chunk_prob:
                    rows, p = bucket
                    choice = rows[rng.choice(len(rows), p=p)]
                    words.extend(int(x) for x in choice); prev = int(choice[-1])
                    j += L; hit = True; break
        if not hit:
            if bidir and schamp is not None:
                nxt = next((cls_seq[k] for k in range(j + 1, len(cls_seq)) if cls_seq[k] in table), cl)
                w = _fill_bisel(prev if prev is not None else int(cls_seq[0]), cl, nxt,
                                table, feat, logfreq, cents, schamp, rng) if prev is not None \
                    else int(rng.choice(table[cl][0], p=table[cl][1]))
            else:
                w = int(rng.choice(table[cl][0], p=table[cl][1]))
            words.append(w); prev = w; j += 1
    return words


def run_gate_chunks(n_classes=32, order_gens=2000, sel_gens=1500, C=4, seed=1234, log=print):
    """Does chunk-aware generation (emit real phrases) beat word-by-word fill on
    local English-likeness? Same order skeleton; word-fill vs chunk-fill."""
    import collections
    ores = run_class_lm(n_classes=n_classes, gens=order_gens, pop=200, C=C, E=10, H=64, seed=seed, log=log)
    bres = run_biselection(n_classes=n_classes, gens=sel_gens, pop=200, D=24, K=7, seed=seed, log=log)
    log("building chunk index…")
    index = build_chunk_index(n_classes)
    table, w2c, vocab, nc, cids = build_class_words(n_classes)
    feat, _ = word_features(4000, 24); cents = class_centroids(n_classes, 24)
    ids, _, _ = build_word_corpus(4000)
    logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
    bigset = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
    n2 = sum(len(v) for v in index.get(2, {}).values())
    n3 = sum(len(v) for v in index.get(3, {}).values())
    log(f"chunk index: {n2} bigram-phrases, {n3} trigram-phrases")

    def measure(chunk_prob):
        rng = np.random.default_rng(seed + 9)
        words = gen_chunked(ores["champ"], C, 1500, cids[500:500 + C], table, feat, logfreq,
                            cents, bres["champ"], index if chunk_prob > 0 else {}, vocab, w2c,
                            rng, chunk_prob=chunk_prob)
        cnt = collections.Counter(words)
        adj = float(np.mean([(words[i], words[i + 1]) in bigset for i in range(len(words) - 1)]))
        return {"chunk_prob": chunk_prob, "adj_pair_hit": round(adj, 4),
                "distinct_ratio": round(len(cnt) / len(words), 4),
                "sample": " ".join(vocab[w] for w in words[:50])}

    word_only, chunked = measure(0.0), measure(0.7)
    log(f"word-fill : adj {word_only['adj_pair_hit']} distinct {word_only['distinct_ratio']}")
    log(f"  {word_only['sample']}")
    log(f"chunk-fill: adj {chunked['adj_pair_hit']} distinct {chunked['distinct_ratio']}")
    log(f"  {chunked['sample']}")
    return {"word_only": word_only, "chunked": chunked}


def _ctx_pool(ids, table, w2c, n_seq, L, K, rng):
    """Fixed pool of contiguous sequences with IN-CLASS candidate sets per next-
    word position (true@0). In-class negatives make frequency a weak baseline, so
    the test measures whether MEMORY helps (as selection's in-class test did)."""
    seq = np.empty((n_seq, L), np.int64)
    cand = np.empty((n_seq, L - 1, K + 1), np.int64)
    got = 0
    while got < n_seq:
        for st in rng.integers(0, len(ids) - L, size=(n_seq - got) * 2):
            s = ids[st:st + L]
            cc = np.empty((L - 1, K + 1), np.int64); ok = True
            for j in range(1, L):
                nx = int(s[j]); cl = w2c[nx]
                if cl not in table or len(table[cl][0]) < K + 1:
                    ok = False; break
                cc[j - 1, 0] = nx
                cc[j - 1, 1:] = rng.choice(table[cl][0], size=K, replace=False)
            if not ok:
                continue
            seq[got] = s; cand[got] = cc; got += 1
            if got >= n_seq:
                break
    return seq.astype(np.int64), cand


def run_context(n_classes=32, gens=1500, pop=200, D=24, S=24, L=12, K=7,
                minibatch=128, seed=1234, log=print):
    """Evolve the recurrent context genome (in-class negatives). GATE: held-out
    next-word log-prob beats the memoryless frequency baseline."""
    feat, vocab = word_features(4000, D)
    ids, _, _ = build_word_corpus(4000)
    logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
    table, w2c, _, _, _ = build_class_words(n_classes)
    n_train = int(len(ids) * 0.9)
    rng = np.random.default_rng(seed)
    log("building context sequence pools…")
    tr_seq, tr_cand = _ctx_pool(ids[:n_train], table, w2c, 20000, L, K, rng)
    v_seq, v_cand = _ctx_pool(ids[n_train:], table, w2c, 2000, L, K, np.random.default_rng(seed + 5))
    vsf, vcf, vclf = feat[v_seq], feat[v_cand], logfreq[v_cand]
    bs = vclf; bm = bs.max(-1, keepdims=True)
    base_lp = float((bs - bm - np.log(np.exp(bs - bm).sum(-1, keepdims=True)))[:, :, 0].mean())

    popn = ContextPop(pop, D, S, seed)
    best_val, best_champ = -1e9, None
    for gen in range(1, gens + 1):
        s = rng.integers(0, len(tr_seq), size=minibatch)
        fit = popn.fitness(feat[tr_seq[s]], feat[tr_cand[s]], logfreq[tr_cand[s]])
        pdict = {"A": popn.A, "B": popn.B, "U": popn.U, "beta": popn.beta, "sigma": popn.sigma}
        ga_step(pdict, fit, rng)
        popn.A, popn.B, popn.U, popn.beta, popn.sigma = (pdict[k] for k in ("A", "B", "U", "beta", "sigma"))
        if gen % 100 == 0 or gen == 1:
            v = float(popn.fitness(vsf, vcf, vclf)[0])
            if v > best_val:
                best_val = v; best_champ = popn.champion(0)
            log(f"  [context] gen {gen}: val_logprob={v:.4f} (base {base_lp:.4f})")
    return {"val_logprob": round(best_val, 4), "base_logprob": round(base_lp, 4),
            "beats_baseline": best_val > base_lp, "champ": best_champ, "D": D, "S": S}


def run_gate6(n_classes=32, order_gens=2000, sel_gens=1500, ctx_gens=1500,
              C=4, S=24, gamma=1.0, seed=1234, log=print):
    """Full test: does the context genome REDUCE repetition in generation? Train
    order + selection + context; generate WITHOUT and WITH context (state query
    reweights each pick); compare distinct-ratio / top-word share against the
    corpus's own distinct-ratio (the target for 'not repetitive')."""
    import collections
    ores = run_class_lm(n_classes=n_classes, gens=order_gens, pop=200, C=C, E=10, H=64, seed=seed, log=log)
    sres = run_selection(n_classes=n_classes, gens=sel_gens, pop=200, D=24, K=7, seed=seed, log=log)
    cres = run_context(n_classes=n_classes, gens=ctx_gens, pop=200, D=24, S=S, seed=seed, log=log)
    table, w2c, vocab, nc, cids = build_class_words(n_classes)
    feat, _ = word_features(4000, 24)
    ids, _, _ = build_word_corpus(4000)
    logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
    ochamp, schamp, cchamp = ores["champ"], sres["champ"], cres["champ"]
    A, Bm, U, cbeta = cchamp

    def generate(use_ctx):
        rng = np.random.default_rng(seed + 9)
        cls_seq = gen_class_seq(ochamp, C, 1500, cids[500:500 + C], rng, temp=0.8)
        words, prev = [], int(ids[499])
        h = np.zeros(S, np.float32)
        for cl in cls_seq:
            if cl not in table:
                continue
            mem = table[cl][0]
            s = (feat[prev] @ schamp[0]) @ feat[mem].T + schamp[1] * logfreq[mem]
            if use_ctx:
                q = h @ U
                s = s + gamma * (feat[mem] @ q)
            s = s / 0.7; s -= s.max(); p = np.exp(s); p /= p.sum()
            w = int(rng.choice(mem, p=p))
            words.append(w); prev = w
            h = np.tanh(h @ A + feat[w] @ Bm)                 # advance context state
        cnt = collections.Counter(words)
        return {"distinct_ratio": round(len(cnt) / len(words), 4),
                "top_word_share": round(cnt.most_common(1)[0][1] / len(words), 4),
                "top5": [(vocab[w], c) for w, c in cnt.most_common(5)],
                "sample": " ".join(vocab[w] for w in words[:45])}

    # corpus reference distinct-ratio over a matched 1500-word slice (non-unk)
    ref = ids[10000:10000 + 4000]; ref = ref[ref != 0][:1500]
    corpus_distinct = round(len(set(ref.tolist())) / len(ref), 4)
    no_ctx, with_ctx = generate(False), generate(True)
    out = {"context_val_logprob": cres["val_logprob"], "context_base": cres["base_logprob"],
           "corpus_distinct_ratio": corpus_distinct,
           "no_context": no_ctx, "with_context": with_ctx}
    log(f"corpus distinct-ratio target: {corpus_distinct}")
    log(f"NO context : distinct {no_ctx['distinct_ratio']} top_share {no_ctx['top_word_share']} top {no_ctx['top5'][0]}")
    log(f"  {no_ctx['sample']}")
    log(f"WITH context: distinct {with_ctx['distinct_ratio']} top_share {with_ctx['top_word_share']} top {with_ctx['top5'][0]}")
    log(f"  {with_ctx['sample']}")
    return out


def run_classcount(n_classes, order_gens=2000, sel_gens=1500, C=4, seed=1234, log=print):
    """Train order + selection at a given class count and measure whether MORE
    classes improve the output: less repetition (higher distinct-word ratio, lower
    top-word share) and higher local English-likeness (adj-pair hit). Finer classes
    should split the dominant <unk>-polluted mega-class that drives 'more/good'."""
    import collections
    ores = run_class_lm(n_classes=n_classes, gens=order_gens, pop=200, C=C, E=10, H=64, seed=seed, log=log)
    sres = run_selection(n_classes=n_classes, gens=sel_gens, pop=200, D=24, K=7, seed=seed, log=log)
    table, w2c, vocab, nc, cids = build_class_words(n_classes)
    feat, _ = word_features(4000, 24)
    ids, _, _ = build_word_corpus(4000)
    logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
    bigset = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
    rng = np.random.default_rng(seed + 9)
    cls_seq = gen_class_seq(ores["champ"], C, 1500, cids[500:500 + C], rng, temp=0.8)
    words, prev = [], int(ids[499])
    for cl in cls_seq:
        if cl not in table:
            continue
        w = _fill_selected(prev, cl, table, feat, logfreq, sres["champ"], rng); prev = w
        words.append(w)
    cnt = collections.Counter(words)
    adj = float(np.mean([(words[i], words[i + 1]) in bigset for i in range(len(words) - 1)]))
    top = cnt.most_common(5)
    return {"n_classes": nc, "order_val_ppl": ores["val_ppl"], "order_unigram": ores["unigram_ppl"],
            "adj_pair_hit": round(adj, 4),
            "distinct_ratio": round(len(cnt) / len(words), 4),
            "top_word_share": round(top[0][1] / len(words), 4),
            "top5": [(vocab[w], c) for w, c in top],
            "sample": " ".join(vocab[w] for w in words[:45])}


def run_orderer(disc_result, gens=1200, pop=200, C=4, E=10, H=48, n_win=6,
                seed=1234, log=print):
    """Evolve a word generator whose fitness is the FROZEN discriminator's
    P(real) on its output — adversarial grammaticality, NOT next-word accuracy.
    Reports generated order-score vs the real and shuffled reference levels."""
    champ = disc_result["champ"]
    vocab = disc_result["vocab"]
    W = disc_result["W"]
    ids, _, _ = build_word_corpus(disc_result["vocab_n"])
    Vw = len(vocab)
    rng = np.random.default_rng(seed)
    # reference levels: what the discriminator says about REAL and SHUFFLED order
    ref = sample_windows(ids, 4096, W, np.random.default_rng(seed + 3))
    real_ref = float(disc_score(champ, ref).mean())
    shuf_ref = float(disc_score(champ, shuffle_rows(ref, rng)).mean())
    log(f"  [orderer] discriminator says: REAL order={real_ref:.3f}  "
        f"SHUFFLED order={shuf_ref:.3f} (these bracket the orderer)")

    popn = OrderPop(pop, Vw, C, E, H, seed)
    seed_ctx = ids[100:100 + C].astype(np.int64)
    best_score = 0.0
    for gen in range(1, gens + 1):
        gw = popn.gen_windows(W * n_win, seed_ctx, rng)          # (P, W*n_win)
        wins = gw.reshape(pop * n_win, W)                        # windows to score
        s = disc_score(champ, wins).reshape(pop, n_win).mean(axis=1)  # (P,) fitness
        pdict = {"emb": popn.emb, "pos": popn.pos, "W1": popn.W1, "b1": popn.b1,
                 "W2": popn.W2, "b2": popn.b2, "sigma": popn.sigma}
        ga_step(pdict, s.astype(np.float32), rng)
        (popn.emb, popn.pos, popn.W1, popn.b1, popn.W2, popn.b2, popn.sigma) = (
            pdict["emb"], pdict["pos"], pdict["W1"], pdict["b1"],
            pdict["W2"], pdict["b2"], pdict["sigma"])
        best_score = max(best_score, float(s[0]))
        if gen % 200 == 0 or gen == 1:
            log(f"  [orderer] gen {gen}: order_score={s[0]:.3f} best={best_score:.3f}")
    # champion sample as words
    gw = popn.gen_windows(24, seed_ctx, rng, temp=0.8)
    words = " ".join(vocab[i] for i in gw[0])
    return {"real_ref": round(real_ref, 4), "shuffled_ref": round(shuf_ref, 4),
            "best_order_score": round(best_score, 4), "sample_words": words}

