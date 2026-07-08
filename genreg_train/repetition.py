"""Repetition-penalty genome — ONE job: don't repeat a (content) word that just appeared
in the last N positions. Function words are allowed to recur ("of the … of the"); content
words are not. Insurance against the collapse/repetition failure mode.

Not a pairwise discriminator — a tiny STATEFUL genome: an evolved recency-penalty curve
w[1..N] (+ a content gate) applied to a candidate given the recently emitted words. Trained
as a discriminator: the real next word (which rarely repeats recent content) vs a negative
that forces a repeat of a recent content word. ~N+2 params. Applied at word selection using
the generation history buffer.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al

N = 8            # look-back window


def content_mask(vocab):
    return np.array([w.isalpha() and w not in al.FUNCTION and len(w) > 2
                     for w in vocab], dtype=bool)


class RepetitionPop:
    """Genome = recency weights w[N] + content gate + bias. penalty(cand|history) =
    (content(cand)*cg + b) * sum_k w[k]*[history[-k]==cand]. Higher score = keep."""

    def __init__(self, pop, seed):
        rng = np.random.default_rng(seed)
        self.w = (rng.standard_normal((pop, N)) * 0.3).astype(np.float32)
        self.cg = np.full(pop, 0.5, np.float32)
        self.b = np.zeros(pop, np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def score(self, rep, isc):                       # rep (M,N) 0/1, isc (M,) -> (P,M)
        gate = (self.cg[:, None] * isc[None, :] + self.b[:, None])       # (P,M)
        return gate * np.einsum("pk,mk->pm", self.w, rep)                # signed penalty

    def fitness(self, rep_r, isc_r, rep_n, isc_n):
        sr = self.score(rep_r, isc_r); sn = self.score(rep_n, isc_n)
        # want real >= 0 (no penalty) and negative (forced repeat) << 0
        acc = ((sr > sn).mean(1))
        margin = np.clip(sr - sn, -20, 20)
        return np.log(np.clip(1 / (1 + np.exp(-margin)), 1e-6, 1)).mean(1), acc

    def champion(self, idx):
        return {"w": self.w[idx].copy(), "cg": float(self.cg[idx]), "b": float(self.b[idx])}


def train_rep(vocab_n=4000, gens=1500, pop=200, minibatch=2048, seed=7, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    isc = content_mask(vocab)
    n = len(ids); n_train = int(n * 0.9)
    rng = np.random.default_rng(seed)
    offs = np.arange(1, N + 1)

    def batch(lo, hi, mb, rr):
        pos = rr.integers(lo + N, hi, size=mb)
        hist = ids[pos[:, None] - offs]              # (mb,N) recent words
        cand_r = ids[pos]                            # real next word
        rep_r = (hist == cand_r[:, None]).astype(np.float32)
        # negative: force a repeat of a recent content word (fallback: real)
        recent_c = np.where(isc[hist], hist, -1)
        pick = recent_c.copy()
        cand_n = cand_r.copy()
        for row in range(mb):
            cs = recent_c[row][recent_c[row] >= 0]
            if len(cs):
                cand_n[row] = cs[rr.integers(0, len(cs))]
        rep_n = (hist == cand_n[:, None]).astype(np.float32)
        return rep_r, isc[cand_r], rep_n, isc[cand_n]

    vr = np.random.default_rng(seed + 3)
    vb = batch(n_train, n, 4096, vr)
    popn = RepetitionPop(pop, seed)
    best_acc, champ = 0.0, None
    for gen in range(1, gens + 1):
        rr_, ir, rn, ins = batch(0, n_train, minibatch, rng)
        fit, _ = popn.fitness(rr_, ir, rn, ins)
        pd = {"w": popn.w, "cg": popn.cg, "b": popn.b, "sigma": popn.sigma}
        wp.ga_step(pd, fit, rng)
        popn.w, popn.cg, popn.b, popn.sigma = pd["w"], pd["cg"], pd["b"], pd["sigma"]
        if gen % 200 == 0 or gen == 1:
            _, acc = popn.fitness(*vb)
            if float(acc[0]) > best_acc:
                best_acc = float(acc[0]); champ = popn.champion(0)
            log(f"  [rep] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4), "vocab": vocab}


def penalty(champ, recent_ids, cand_ids, is_content):
    """Per-candidate repetition penalty (<=0) given the recent word buffer (most-recent
    last). `recent_ids`: list; `cand_ids`: (M,) array; `is_content`: (Vw,) bool."""
    w, cg, b = champ["w"], champ["cg"], champ["b"]
    rec = list(recent_ids[-N:])
    rep = np.zeros((len(cand_ids), N), np.float32)
    for k, wid in enumerate(reversed(rec)):          # k=0 -> offset 1
        rep[:, k] = (cand_ids == wid)
    gate = cg * is_content[cand_ids].astype(np.float32) + b
    return gate * (rep @ w)
