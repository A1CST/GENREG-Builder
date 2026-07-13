"""Punctuation-sequence genome — the discourse skeleton (user's idea: the
punctuation mark IS the intent, chosen before any word exists; the sequence
of marks across a passage declares its discourse shape before a single word
commits to it). ONE job: given the last few marks that occurred, what mark
comes next? Autoregressive model over a tiny 6-symbol alphabet (period,
comma, semicolon, colon, exclamation, question) instead of the 32 word-
classes Order uses — same architecture (OrderPop), same training loop,
just a much smaller vocabulary. Mined directly from the corpus's real
punctuation sequence, zero external labeling.
"""
import numpy as np

from genreg_train import wordpipe as wp

MARKS = [".", ",", ";", ":", "!", "?"]
MARK_ID = {m: i for i, m in enumerate(MARKS)}
NM = len(MARKS)


def mine_mark_sequence(vocab_n=4000):
    """Walk the raw corpus text and extract the ORDERED sequence of
    punctuation marks (ignoring words) as small integer ids."""
    text = wp.decode(wp.corpus_ids())
    toks = text.split()
    seq = []
    for t in toks:
        for ch in t:
            if ch in MARK_ID:
                seq.append(MARK_ID[ch])
    return np.asarray(seq, np.int64)


def train_intent_punct(gens=1500, pop=200, C=4, E=8, H=32, minibatch=1024, seed=7, log=print):
    mseq = mine_mark_sequence()
    log(f"mined {len(mseq)} punctuation marks in sequence")
    n = len(mseq); n_train = int(n * 0.9)
    rng = np.random.default_rng(seed)
    offs = np.arange(C + 1)

    def batch(lo, hi, mb, rr):
        st = rr.integers(lo, hi - C - 1, size=mb)
        win = mseq[st[:, None] + offs]
        return win[:, :C], win[:, C]

    vr = np.random.default_rng(seed + 3)
    vctx, vtgt = batch(n_train, n, min(8000, n - n_train), vr)
    cnt = np.bincount(mseq[:n_train], minlength=NM).astype(np.float64)
    pu = np.clip(cnt / cnt.sum(), 1e-9, 1.0)
    uni_ppl = float(np.exp(-np.log(pu[vtgt]).mean()))
    log(f"mark marginal: {dict(zip(MARKS, np.round(pu, 4)))}")
    log(f"unigram baseline ppl: {uni_ppl:.3f}")

    popn = wp.OrderPop(pop, NM, C, E, H, seed)
    best_val = -1e9; champ = None
    for gen in range(1, gens + 1):
        ctx, tgt = batch(0, n_train, minibatch, rng)
        fit = popn.fitness_all(ctx, tgt)
        pd = {"emb": popn.emb, "pos": popn.pos, "W1": popn.W1, "b1": popn.b1,
             "W2": popn.W2, "b2": popn.b2, "sigma": popn.sigma}
        wp.ga_step(pd, fit, rng)
        (popn.emb, popn.pos, popn.W1, popn.b1, popn.W2, popn.b2, popn.sigma) = (
            pd["emb"], pd["pos"], pd["W1"], pd["b1"], pd["W2"], pd["b2"], pd["sigma"])
        if gen % 100 == 0 or gen == 1:
            v = float(popn.fitness_all(vctx, vtgt)[0])
            best_val = max(best_val, v)
            if v == best_val:
                champ = popn.champion(0)
            log(f"  [intent-punct] gen {gen}: val_ppl={np.exp(-v):.3f} (unigram {uni_ppl:.3f})")
    return {"champ": champ, "val_ppl": round(float(np.exp(-best_val)), 3),
           "unigram_ppl": round(uni_ppl, 3), "C": C, "E": E, "H": H, "marks": MARKS}


def gen_mark_seq(champ, length, seed_ctx, rng, C=4, temp=0.9):
    """Autoregressively emit `length` mark ids from the trained champion."""
    return wp.gen_class_seq(champ, C, length, seed_ctx, rng, temp)
