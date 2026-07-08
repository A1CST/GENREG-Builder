"""Shared scaffolding for the constraint-genome battery (genomes.txt roadmap).

Most WordPipe constraint genomes are the SAME shape: a fixed per-word feature vector
(grammatical, function-type, or distributional — kept OUT of evolution) and a tiny
evolved bilinear head M that scores an adjacent/near pair. They differ only in (a) the
features, (b) which positive pairs, and (c) how negatives are drawn. This module
factors that shared training loop so each genome file is just its feature function +
its pair/negative policy. Agreement and alternation predate this helper and keep their
own inline copies; new pairwise genomes use `train_pairwise`.

A pairwise genome exposes its champion M[NF,NF]; the service re-ranks word selection by
feat[prev] @ M @ feat[cand] and re-ranks the ORDER skeleton by lifting M to per-class
centroids (classfeat @ M @ classfeat.T). No transition table is ever stored.
"""
import numpy as np

from genreg_train import wordpipe as wp


class BilinearPop:
    """Genome = a bilinear head M (NF x NF). score(prev, cand) = f(prev)^T M f(cand)."""

    def __init__(self, pop, nf, seed, scale=None):
        rng = np.random.default_rng(seed)
        s = scale if scale is not None else (1.0 / nf)
        self.M = (rng.standard_normal((pop, nf, nf)) * s).astype(np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def score(self, pf, cf):                       # (N,NF),(N,NF) -> (P,N)
        return np.einsum("pne,ne->pn", np.einsum("nd,pde->pne", pf, self.M), cf)

    def fitness(self, pf_r, cf_r, pf_n, cf_n):
        sr = self.score(pf_r, cf_r); sn = self.score(pf_n, cf_n)
        pr = np.clip(1 / (1 + np.exp(-sr)), 1e-6, 1 - 1e-6)
        pn = np.clip(1 / (1 + np.exp(-sn)), 1e-6, 1 - 1e-6)
        acc = ((sr > 0).mean(1) + (sn < 0).mean(1)) / 2
        return (np.log(pr) + np.log(1 - pn)).mean(1), acc

    def champion(self, idx):
        return self.M[idx].copy()


def train_pairwise(G, pos_a, pos_b, neg_pool, name="pairwise",
                   gens=2500, pop=200, minibatch=1024, seed=7, log=print):
    """Train a bilinear discriminator: positive pairs (pos_a[i], pos_b[i]) vs hard
    negatives (same prev, candidate resampled from `neg_pool`). Only the relation the
    features expose separates them, so the genome learns the RULE, not a lookup.

    G          : (Vw, NF) fixed feature matrix
    pos_a/pos_b: int arrays of positive (prev, cand) word ids
    neg_pool   : int array to draw negative candidates from (e.g. the real next-word
                 marginal, or the content-word set)
    """
    G = np.ascontiguousarray(G, np.float32)
    nf = G.shape[1]
    n = len(pos_a)
    n_train = int(n * 0.9)
    rng = np.random.default_rng(seed)

    def batch(lo, hi, mb, rr):
        idx = rr.integers(lo, hi, size=mb)
        pv, nx = pos_a[idx], pos_b[idx]
        neg = neg_pool[rr.integers(0, len(neg_pool), size=mb)]
        return G[pv], G[nx], G[pv], G[neg]

    vr = np.random.default_rng(seed + 3)
    vpr, vcr, vpn, vcn = batch(n_train, n, min(8000, n - n_train), vr)
    popn = BilinearPop(pop, nf, seed)
    best_acc, champ = 0.0, None
    for gen in range(1, gens + 1):
        pr, cr, pn, cn = batch(0, n_train, minibatch, rng)
        fit, _ = popn.fitness(pr, cr, pn, cn)
        pd = {"M": popn.M, "sigma": popn.sigma}
        wp.ga_step(pd, fit, rng)
        popn.M, popn.sigma = pd["M"], pd["sigma"]
        if gen % 200 == 0 or gen == 1:
            _, acc = popn.fitness(vpr, vcr, vpn, vcn)
            if float(acc[0]) > best_acc:
                best_acc = float(acc[0]); champ = popn.champion(0)
            log(f"  [{name}] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4)}


def score_pair_M(M, feats, vocab, w1, w2):
    """Generic bilinear pair score given a feature function `feats(list)->array`."""
    stoi = {w: i for i, w in enumerate(vocab)}
    if w1 in stoi and w2 in stoi:
        return float(feats([w1])[0] @ M @ feats([w2])[0])
    return float(feats([w1])[0] @ M @ feats([w2])[0])


# --------------------------------------------------------------------------
# Windowed class-sequence discriminators (verb-argument, pronoun-reference, …):
# score a K-class window and tell a real one from one corrupted at a chosen slot.
# The genome is a class embedding + a one-hidden-layer validity head; it expands
# to a bias tensor for O(1) lookup, biasing the ORDER skeleton's next-class logits.
# --------------------------------------------------------------------------
class WindowPop:
    def __init__(self, pop, nc, K, E, Hd, seed):
        rng = np.random.default_rng(seed)
        self.K, self.E = K, E
        self.emb = (rng.standard_normal((pop, nc, E)) * 0.3).astype(np.float32)
        self.W1 = (rng.standard_normal((pop, K * E, Hd)) * (1.0 / np.sqrt(K * E))).astype(np.float32)
        self.b1 = np.zeros((pop, Hd), np.float32)
        self.w2 = (rng.standard_normal((pop, Hd)) * (1.0 / np.sqrt(Hd))).astype(np.float32)
        self.b2 = np.zeros(pop, np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def score(self, win):
        P = self.emb.shape[0]; Nn = win.shape[0]
        x = self.emb[:, win, :].reshape(P, Nn, self.K * self.E)
        h = np.tanh(np.einsum("pnx,pxh->pnh", x, self.W1) + self.b1[:, None, :])
        return np.einsum("pnh,ph->pn", h, self.w2) + self.b2[:, None]

    def fitness(self, wr, wn):
        sr = self.score(wr); sn = self.score(wn)
        pr = np.clip(1 / (1 + np.exp(-sr)), 1e-6, 1 - 1e-6)
        pn = np.clip(1 / (1 + np.exp(-sn)), 1e-6, 1 - 1e-6)
        acc = ((sr > 0).mean(1) + (sn < 0).mean(1)) / 2
        return (np.log(pr) + np.log(1 - pn)).mean(1), acc

    def champion(self, idx):
        return (self.emb[idx].copy(), self.W1[idx].copy(), self.b1[idx].copy(),
                self.w2[idx].copy(), self.b2[idx].copy())


def window_bias_tensor(champ, nc, K, E):
    idx = np.indices((nc,) * K).reshape(K, -1).T
    emb, W1, b1, w2, b2 = champ
    x = emb[idx.astype(np.int64)].reshape(idx.shape[0], K * E)
    s = np.tanh(x @ W1 + b1) @ w2 + b2
    return s.reshape((nc,) * K).astype(np.float32)


class UnaryPop:
    """Genome = a linear scorer w over per-word features. score(word) = feat(word)·w.
    For positional constraints (is this a plausible sentence opener/closer?)."""

    def __init__(self, pop, nf, seed):
        rng = np.random.default_rng(seed)
        self.w = (rng.standard_normal((pop, nf)) * (1.0 / np.sqrt(nf))).astype(np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def score(self, F):                            # (N,nf) -> (P,N)
        return self.w @ F.T

    def fitness(self, Fr, Fn):
        sr = self.score(Fr); sn = self.score(Fn)
        pr = np.clip(1 / (1 + np.exp(-sr)), 1e-6, 1 - 1e-6)
        pn = np.clip(1 / (1 + np.exp(-sn)), 1e-6, 1 - 1e-6)
        acc = ((sr > 0).mean(1) + (sn < 0).mean(1)) / 2
        return (np.log(pr) + np.log(1 - pn)).mean(1), acc

    def champion(self, idx):
        return self.w[idx].copy()


def train_unary(G, pos_ids, neg_pool, name="unary", gens=1500, pop=200,
                minibatch=2048, seed=7, log=print):
    """Train a unary word classifier: words that DO occur in the target position
    (pos_ids, e.g. sentence-initial) vs words drawn from the marginal (neg_pool)."""
    G = np.ascontiguousarray(G, np.float32)
    nf = G.shape[1]
    n = len(pos_ids); n_train = int(n * 0.9)
    rng = np.random.default_rng(seed)

    def batch(lo, hi, mb, rr):
        pv = pos_ids[rr.integers(lo, hi, size=mb)]
        neg = neg_pool[rr.integers(0, len(neg_pool), size=mb)]
        return G[pv], G[neg]

    vr = np.random.default_rng(seed + 3)
    vFr, vFn = batch(n_train, n, min(8000, n - n_train), vr)
    popn = UnaryPop(pop, nf, seed)
    best_acc, champ = 0.0, None
    for gen in range(1, gens + 1):
        Fr, Fn = batch(0, n_train, minibatch, rng)
        fit, _ = popn.fitness(Fr, Fn)
        pd = {"w": popn.w, "sigma": popn.sigma}
        wp.ga_step(pd, fit, rng)
        popn.w, popn.sigma = pd["w"], pd["sigma"]
        if gen % 200 == 0 or gen == 1:
            _, acc = popn.fitness(vFr, vFn)
            if float(acc[0]) > best_acc:
                best_acc = float(acc[0]); champ = popn.champion(0)
            log(f"  [{name}] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4)}


def train_windowed(n_classes, corrupt_pos, name="window", K=3, E=8, Hd=24,
                   gens=2500, pop=200, minibatch=1024, seed=7, log=print):
    """Windowed class discriminator. `corrupt_pos` = which slot the negative corrupts
    (-1 = last/completion, 0 = first/left-dependency such as a missing subject)."""
    w2c, cids, nc, vocab = wp.induce_word_classes(n_classes)
    cids = np.asarray(cids, np.int64)
    n = len(cids); n_train = int(n * 0.9)
    offs = np.arange(K)
    rng = np.random.default_rng(seed)
    cnt = np.bincount(cids[:n_train], minlength=nc).astype(np.float64)
    cprob = cnt / cnt.sum()

    def batch(lo, hi, mb, rr):
        st = rr.integers(lo, hi - K, size=mb)
        win = cids[st[:, None] + offs]
        neg = win.copy()
        neg[:, corrupt_pos] = rr.choice(nc, size=mb, p=cprob)
        return win, neg

    vr = np.random.default_rng(seed + 3)
    vwr, vwn = batch(n_train, n, 8000, vr)
    popn = WindowPop(pop, nc, K, E, Hd, seed)
    best_acc, champ = 0.0, None
    for gen in range(1, gens + 1):
        wr, wn = batch(0, n_train, minibatch, rng)
        fit, _ = popn.fitness(wr, wn)
        pd = {"emb": popn.emb, "W1": popn.W1, "b1": popn.b1, "w2": popn.w2,
              "b2": popn.b2, "sigma": popn.sigma}
        wp.ga_step(pd, fit, rng)
        (popn.emb, popn.W1, popn.b1, popn.w2, popn.b2, popn.sigma) = (
            pd["emb"], pd["W1"], pd["b1"], pd["w2"], pd["b2"], pd["sigma"])
        if gen % 200 == 0 or gen == 1:
            _, acc = popn.fitness(vwr, vwn)
            if float(acc[0]) > best_acc:
                best_acc = float(acc[0]); champ = popn.champion(0)
            log(f"  [{name}] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4), "n_classes": nc, "K": K, "E": E}
