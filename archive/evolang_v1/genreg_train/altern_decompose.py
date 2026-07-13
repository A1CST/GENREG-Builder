"""Alternation, decomposed. The shipped genome (altern.py) trains ONE
bilinear head on ALL adjacent pairs, forcing it to jointly learn two
distinct phenomena named in its own docstring: (1) the COARSE rhythm --
a function word needs a content word beside it -- and (2) the FINE-GRAINED
legality of specific function->function transitions ("of the" yes,
"the of" no, "and and" never). Splitting these gives observability: is a
generation artifact a rhythm violation or a specific illegal function
chain?

  Altern-rhythm:     mines ALL real adjacent pairs (same as the original),
                     but only exposes each word's CONTENT-vs-FUNCTION bit,
                     not its function subtype -- can only ever learn the
                     coarse alternation pattern.
  Altern-func-chain: mines ONLY pairs where BOTH words are function words,
                     with the full 14-feature subtype space -- can only
                     ever learn which specific function->function chains
                     are legal, since content words are never in its data.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al

NF_RHYTHM = 2   # [is_content, bias]


class BinaryFeatPop:
    """Same math as al.AlternPop but feature-dim-agnostic."""
    def __init__(self, pop, nf, seed):
        rng = np.random.default_rng(seed)
        self.M = (rng.standard_normal((pop, nf, nf)) * (1.0 / nf)).astype(np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def score(self, pf, cf):
        return np.einsum("pne,ne->pn", np.einsum("nd,pde->pne", pf, self.M), cf)

    def fitness(self, pf_r, cf_r, pf_n, cf_n):
        sr = self.score(pf_r, cf_r); sn = self.score(pf_n, cf_n)
        pr = np.clip(1 / (1 + np.exp(-sr)), 1e-6, 1 - 1e-6)
        pn = np.clip(1 / (1 + np.exp(-sn)), 1e-6, 1 - 1e-6)
        acc = ((sr > 0).mean(1) + (sn < 0).mean(1)) / 2
        return (np.log(pr) + np.log(1 - pn)).mean(1), acc

    def champion(self, idx):
        return self.M[idx].copy()


def rhythm_feats(vocab):
    """[is_content, bias] -- deliberately blind to function SUBTYPE."""
    F = np.zeros((len(vocab), NF_RHYTHM), np.float32)
    for i, w in enumerate(vocab):
        F[i, 0] = 1.0 if (w.isalpha() and w not in al.FUNCTION) else 0.0
        F[i, 1] = 1.0
    return F


def _train(pf_all, cf_all, feat_a, feat_b, name, gens, pop, seed, log):
    n = len(feat_a); n_train = int(n * 0.9)
    rng = np.random.default_rng(seed)
    nf = pf_all.shape[1]

    def batch(lo, hi, mb, rr):
        idx = rr.integers(lo, hi, size=mb)
        pv, nx = feat_a[idx], feat_b[idx]
        neg = feat_b[rr.integers(0, n, size=mb)]
        return pf_all[pv], cf_all[nx], pf_all[pv], cf_all[neg]

    vr = np.random.default_rng(seed + 3)
    vpr, vcr, vpn, vcn = batch(n_train, n, min(8000, n - n_train), vr)
    popn = BinaryFeatPop(pop, nf, seed)
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
    return {"champ": champ, "val_acc": round(best_acc, 4)}


def train_altern_rhythm(vocab_n=4000, gens=2000, pop=200, seed=7, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    nz = ids[:-1] != 0
    a, b = ids[:-1][nz], ids[1:][nz]
    G = rhythm_feats(vocab)
    log(f"altern-rhythm: {len(a)} adjacent pairs, feature dim {NF_RHYTHM} (content-bit only)")
    res = _train(G, G, a, b, "altern-rhythm", gens, pop, seed, log)
    res["vocab"] = vocab
    return res


def train_altern_funcchain(vocab_n=4000, gens=2000, pop=200, seed=7, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    is_func = np.array([w in al.FUNCTION for w in vocab])
    nz = ids[:-1] != 0
    a, b = ids[:-1][nz], ids[1:][nz]
    both_func = is_func[a] & is_func[b]
    a, b = a[both_func], b[both_func]
    G = al.func_feats(vocab)
    log(f"altern-func-chain: {len(a)} function-function pairs, feature dim {al.NF} (full subtype space)")
    res = _train(G, G, a, b, "altern-func-chain", gens, pop, seed, log)
    res["vocab"] = vocab
    return res
