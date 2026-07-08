"""Agreement, decomposed. The shipped genome (agreement.py) trains ONE
bilinear head on ALL adjacent pairs over a 22-feature space that mixes two
genuinely distinct grammatical rules (named explicitly in genomes.txt:
"modal->bare, subject-number->copula"):

  Agree-modal:  a modal auxiliary (could/would/should/...) must be followed
                by a BARE verb form ("could be", not "could is"). Mines
                only pairs where prev is a MODAL or cand is BARE/PARTICIPLE.
  Agree-number: subject number/person (he/she/it vs they/we/you vs I) must
                match the following copula/aux's finite form ("he is",
                "they are", "I am"). Mines only pairs touching the
                pronoun/finiteness features.

Same full 22-feature space for both (simpler than re-deriving a trimmed
space per phenomenon), restricted TRAINING PAIRS per phenomenon instead --
each sub-genome only ever sees examples of its own rule, so it can only
ever learn that rule.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import agreement as ag


def _train(G, feat_a, feat_b, name, gens, pop, seed, log):
    n = len(feat_a); n_train = int(n * 0.9)
    rng = np.random.default_rng(seed)
    nf = G.shape[1]

    def batch(lo, hi, mb, rr):
        idx = rr.integers(lo, hi, size=mb)
        pv, nx = feat_a[idx], feat_b[idx]
        neg = feat_b[rr.integers(0, n, size=mb)]
        return G[pv], G[nx], G[pv], G[neg]

    vr = np.random.default_rng(seed + 3)
    vpr, vcr, vpn, vcn = batch(n_train, n, min(8000, n - n_train), vr)
    popn = ag.AgreePop(pop, seed)
    best_acc, champ = 0.0, None
    for gen in range(1, gens + 1):
        pr, cr, pn, cn = batch(0, n_train, 1024, rng)
        fit, _ = popn.fitness(pr, cr, pn, cn)
        pd = {"M": popn.M, "sigma": popn.sigma}
        wp.ga_step(pd, fit, rng)
        popn.M, popn.sigma = pd["M"], pd["sigma"]
        if gen % 200 == 0 or gen == 1:
            _, acc = popn.fitness(vpr, vcr, vpn, vcn)
            if float(acc[0]) > best_acc:
                best_acc = float(acc[0]); champ = popn.champion(0)
            log(f"  [{name}] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4), "nf": nf}


def train_agree_modal(vocab_n=4000, gens=2500, pop=200, seed=7, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    is_modal = np.array([w in ag.MODALS for w in vocab])
    is_bare_or_part = np.array([w in ag.BARE or w in ag.PARTICIPLE for w in vocab])
    nz = ids[:-1] != 0
    a, b = ids[:-1][nz], ids[1:][nz]
    keep = is_modal[a] | is_bare_or_part[b]
    a, b = a[keep], b[keep]
    G = ag.gram_feats(vocab)
    log(f"agree-modal: {len(a)} modal-relevant pairs")
    res = _train(G, a, b, "agree-modal", gens, pop, seed, log)
    res["vocab"] = vocab
    return res


def train_agree_number(vocab_n=4000, gens=2500, pop=200, seed=7, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    is_pron = np.array([w in ag.SING_PRON or w in ag.PLUR_PRON or w in ag.FIRST_PRON
                        for w in vocab])
    is_finite = np.array([w in ag.FIN_3SG or w in ag.FIN_NON3 for w in vocab])
    nz = ids[:-1] != 0
    a, b = ids[:-1][nz], ids[1:][nz]
    keep = is_pron[a] | is_finite[b]
    a, b = a[keep], b[keep]
    G = ag.gram_feats(vocab)
    log(f"agree-number: {len(a)} number/finiteness-relevant pairs")
    res = _train(G, a, b, "agree-number", gens, pop, seed, log)
    res["vocab"] = vocab
    return res
