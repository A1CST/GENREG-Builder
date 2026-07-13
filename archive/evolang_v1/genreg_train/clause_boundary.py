"""Clause-boundary genome — ONE job: predict where one clause ends and the next begins
INSIDE a sentence (before a subordinator/relative — "…the man, WHO was tall, walked").
Enables nested structure beyond the flat sentence the Boundary genome cuts. Same shape as
the Boundary/Comma specialists (class + clause-position -> P(clause break)); the target is
positions immediately before a subordinating conjunction or relative wh-word.

Overlaps somewhat with the Comma genome (internal punctuation) — the battery will show
whether it adds structure or is redundant; cut and log if so. ~580 params.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al

RELATIVE = al.SUBORD | set("who which that whom whose where when".split())


def build_clause_corpus(n_classes=32, vocab_n=4000):
    _, vocab, stoi = wp.build_word_corpus(vocab_n)
    w2c, _, nc, _ = wp.induce_word_classes(n_classes)
    toks = wp.decode(wp.corpus_ids()).split()
    ids = np.fromiter((stoi.get(t, 0) for t in toks), np.int32, len(toks))
    strip = [t.strip(",.!?;:'\"") for t in toks]
    # target: a clause break FOLLOWS position i if the next token opens a subordinate/
    # relative clause, or this token carries clause punctuation.
    y = np.zeros(len(toks), np.float32)
    for i in range(len(toks) - 1):
        if strip[i + 1] in RELATIVE or (toks[i] and toks[i][-1] in ",;:"):
            y[i] = 1.0
    reset = np.fromiter((1 if t and t[-1] in ",.!?;:" else 0 for t in toks), np.int8, len(toks))
    cls = w2c[ids].astype(np.int64)
    posn = np.empty(len(toks), np.float32); c = 0
    for i in range(len(toks)):
        posn[i] = c; c = 0 if reset[i] else c + 1
    return ids, cls, y, posn, nc


def train_clausebound(n_classes=32, gens=1200, pop=200, E=8, H=32, minibatch=1024,
                      seed=7, log=print):
    ids, cls, y, posn, nc = build_clause_corpus(n_classes)
    n_train = int(len(ids) * 0.9)
    rng = np.random.default_rng(seed)
    popn = wp.BoundaryPop(pop, nc, E, H, seed)
    base = float(y[:n_train].mean())
    vsel = rng.integers(0, len(ids) - n_train, size=8192) + n_train
    vcls, vpos, vy = cls[vsel], posn[vsel], y[vsel]
    base_lp = float((vy * np.log(base) + (1 - vy) * np.log(1 - base)).mean())
    best_val, champ = -1e9, None
    for gen in range(1, gens + 1):
        s = rng.integers(0, n_train, size=minibatch)
        fit = popn.fitness(cls[s], posn[s], y[s])
        pd = {"emb": popn.emb, "W1": popn.W1, "b1": popn.b1, "W2": popn.W2,
              "b2": popn.b2, "sigma": popn.sigma}
        wp.ga_step(pd, fit, rng)
        (popn.emb, popn.W1, popn.b1, popn.W2, popn.b2, popn.sigma) = (
            pd["emb"], pd["W1"], pd["b1"], pd["W2"], pd["b2"], pd["sigma"])
        if gen % 100 == 0 or gen == 1:
            v = float(popn.fitness(vcls, vpos, vy)[0])
            if v > best_val:
                best_val = v; champ = popn.champion(0)
            log(f"  [clausebound] gen {gen}: val_logprob={v:.4f} (base {base_lp:.4f})")
    return {"val_logprob": round(best_val, 4), "base_logprob": round(base_lp, 4),
            "beats_baseline": best_val > base_lp, "champ": champ, "rate": round(base, 4)}
