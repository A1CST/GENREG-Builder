"""Content-function alternation genome — evolve a scorer for the alternation rhythm
of English: a function word (the, of, and, to, is, …) must sit next to a CONTENT
word, and only a few function->function transitions are legal (of->the yes, the->of
no; and->and never). Kills the "the of", "and and", "of the of" salad the order
skeleton leaks.

Distinct from agreement (finiteness/number) and selection (distribution): this genome
sees each word's FUNCTION TYPE (article, prep, conj, aux, modal, copula, pronoun,
determiner, wh, "to", or CONTENT) and evolves a bilinear head that scores whether an
adjacent pair alternates legally. Fitness = tell real corpus pairs from hard negatives
(same prev, candidate resampled from the real next-word distribution — so only the
content/function rhythm separates them). Tiny (a 14x14 matrix), gradient-free.
"""
import numpy as np

from genreg_train import wordpipe as wp

ARTICLES = set("a an the".split())
DET = set("this that these those my your his her its our their no every each some any".split())
PREPS = set("of to in on at by for with from as into over under about after before "
            "between through during without within against toward upon among".split())
COORD = set("and or but nor".split())
SUBORD = set("if when while because that as than though since unless although whether".split())
AUX = set("have has had do does did having".split())
MODALS = set("could would should can will may might must shall".split())
COPULA = set("be is are was were been being am".split())
PRON = set("he she it they we you i him her them us me who".split())
WH = set("what where who how why which when whom whose".split())
NEG = set("not no never none nothing".split())
TO = set("to".split())

FUNCTION = (ARTICLES | DET | PREPS | COORD | SUBORD | AUX | MODALS | COPULA
            | PRON | WH | NEG | TO)

NF = 14


def func_feats(vocab):
    """Fixed function-type feature vector per word (closed-class membership; no parser,
    no labels, no transition table). Row 13 is bias. Row 0 = CONTENT (open class)."""
    F = np.zeros((len(vocab), NF), np.float32)
    for i, w in enumerate(vocab):
        is_fn = w in FUNCTION
        F[i, 0] = 1.0 if (w.isalpha() and not is_fn) else 0.0   # CONTENT (open class)
        F[i, 1] = w in ARTICLES
        F[i, 2] = w in PREPS
        F[i, 3] = w in COORD
        F[i, 4] = w in SUBORD
        F[i, 5] = w in AUX
        F[i, 6] = w in MODALS
        F[i, 7] = w in COPULA
        F[i, 8] = w in PRON
        F[i, 9] = w in DET
        F[i, 10] = w in WH
        F[i, 11] = w in TO
        F[i, 12] = w in NEG
        F[i, 13] = 1.0                                           # bias
    return F


class AlternPop:
    """Genome = a bilinear head M (NF x NF). score(prev, cand) = t(prev)^T M t(cand),
    high when the pair alternates legally, low for function->function salad."""

    def __init__(self, pop, seed):
        rng = np.random.default_rng(seed)
        self.M = (rng.standard_normal((pop, NF, NF)) * (1.0 / NF)).astype(np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def score(self, pf, cf):                       # (N,NF),(N,NF) -> (P,N)
        return np.einsum("pne,ne->pn", np.einsum("nd,pde->pne", pf, self.M), cf)

    def fitness(self, pf_r, cf_r, pf_n, cf_n):
        """-BCE: real pairs -> 1, corrupted pairs -> 0."""
        sr = self.score(pf_r, cf_r); sn = self.score(pf_n, cf_n)
        pr = np.clip(1 / (1 + np.exp(-sr)), 1e-6, 1 - 1e-6)
        pn = np.clip(1 / (1 + np.exp(-sn)), 1e-6, 1 - 1e-6)
        acc = ((sr > 0).mean(1) + (sn < 0).mean(1)) / 2
        return (np.log(pr) + np.log(1 - pn)).mean(1), acc

    def champion(self, idx):
        return self.M[idx].copy()


def train_altern(vocab_n=4000, gens=2500, pop=200, minibatch=1024, seed=7, log=print):
    """Corrupted-pair discriminator with HARD negatives (same prev, candidate drawn
    from the real next-word distribution). GATE: held-out accuracy beats chance AND
    it ranks alternation minimal pairs correctly."""
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    G = func_feats(vocab)
    nz = ids[:-1] != 0
    a, b = ids[:-1][nz], ids[1:][nz]                # real adjacent pairs (prev, next), non-unk
    n_train = int(len(a) * 0.9)
    rng = np.random.default_rng(seed)

    def batch(lo, hi, mb, rr):
        idx = rr.integers(lo, hi, size=mb)
        pv, nx = a[idx], b[idx]
        neg = b[rr.integers(0, len(a), size=mb)]    # hard: real next-word, wrong for pv
        return G[pv], G[nx], G[pv], G[neg]

    vr = np.random.default_rng(seed + 3)
    vpr, vcr, vpn, vcn = batch(n_train, len(a), 8000, vr)
    popn = AlternPop(pop, seed)
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
            log(f"  [altern] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4), "vocab": vocab}


def score_pair(M, vocab, w1, w2):
    """Alternation score for a bigram (higher = legal alternation)."""
    stoi = {w: i for i, w in enumerate(vocab)}
    if w1 not in stoi or w2 not in stoi:
        g1 = func_feats([w1])[0]; g2 = func_feats([w2])[0]
        return float(g1 @ M @ g2)
    G = func_feats(vocab)
    return float(G[stoi[w1]] @ M @ G[stoi[w2]])
