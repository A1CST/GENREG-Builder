"""Clause-template genome — evolve a scorer for "is this local class window a real
clause fragment?" The order genome does next-CLASS prediction (one job: spread
probability over what usually comes next). This genome does a DIFFERENT job: a
contrastive discriminator that tells a real K-class window (determiner-noun,
preposition-noun, subject-verb, verb-object, det-adj-noun, subject-verb-object …)
from a corrupted one. It reranks the ORDER skeleton the same way alternation and
agreement do — biasing the next-class logits toward candidates that COMPLETE a real
clause pattern given the last few classes.

Feedstock = the real class sequence of the corpus (kept out of evolution); the tiny
genome (a class embedding + a one-hidden-layer validity head, ~900 params) is what
evolves, gradient-free. No class n-gram TABLE is stored — the genome generalizes the
template shapes into a small net; at generation it expands to a bias tensor.
"""
import numpy as np

from genreg_train import wordpipe as wp

K = 3          # clause-window length (context = K-1 classes, complete the K-th)
E = 8          # class-embedding dim
H = 24         # validity-head hidden units


class ClausePop:
    """Genome = class embedding emb[nc,E] + MLP head over a K-class window ->
    a single 'is this a real clause fragment' logit."""

    def __init__(self, pop, nc, seed):
        rng = np.random.default_rng(seed)
        self.nc = nc
        self.emb = (rng.standard_normal((pop, nc, E)) * 0.3).astype(np.float32)
        self.W1 = (rng.standard_normal((pop, K * E, H)) * (1.0 / np.sqrt(K * E))).astype(np.float32)
        self.b1 = np.zeros((pop, H), np.float32)
        self.w2 = (rng.standard_normal((pop, H)) * (1.0 / np.sqrt(H))).astype(np.float32)
        self.b2 = np.zeros(pop, np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def score(self, win):                          # win (N,K) int -> (P,N)
        P = self.emb.shape[0]; N = win.shape[0]
        x = self.emb[:, win, :].reshape(P, N, K * E)          # (P,N,K*E)
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


def _score_one(champ, win):
    """Score windows (N,K) with a single champion -> (N,)."""
    emb, W1, b1, w2, b2 = champ
    x = emb[win].reshape(win.shape[0], K * E)
    h = np.tanh(x @ W1 + b1)
    return h @ w2 + b2


def bias_tensor(champ, nc):
    """Expand the evolved genome into a dense bias tensor B[c0,…,c_{K-1}] = validity
    logit of that window, for O(1) lookup during generation."""
    idx = np.indices((nc,) * K).reshape(K, -1).T          # (nc^K, K)
    s = _score_one(champ, idx.astype(np.int64))
    return s.reshape((nc,) * K).astype(np.float32)


def train_clause(n_classes=32, gens=2500, pop=200, minibatch=1024, seed=7, log=print):
    """Contrastive clause-fragment discriminator. Negatives = same (K-1) context, but
    the completing class is resampled from the class-unigram distribution (hard: the
    genome must learn which class COMPLETES the fragment, not just which is common)."""
    w2c, cids, nc, vocab = wp.induce_word_classes(n_classes)
    cids = np.asarray(cids, np.int64)
    n = len(cids)
    n_train = int(n * 0.9)
    offs = np.arange(K)
    rng = np.random.default_rng(seed)
    cnt = np.bincount(cids[:n_train], minlength=nc).astype(np.float64)
    cprob = cnt / cnt.sum()

    def batch(lo, hi, mb, rr):
        st = rr.integers(lo, hi - K, size=mb)
        win = cids[st[:, None] + offs]                     # (mb,K) real windows
        neg = win.copy()
        neg[:, -1] = rr.choice(nc, size=mb, p=cprob)       # corrupt the completing class
        return win, neg

    vr = np.random.default_rng(seed + 3)
    vwr, vwn = batch(n_train, n, 8000, vr)
    popn = ClausePop(pop, nc, seed)
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
            log(f"  [clause] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4), "n_classes": nc}
