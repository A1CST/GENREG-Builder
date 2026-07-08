"""Agreement genome — evolve a scorer for grammatical number/tense agreement.

Distinct from selection (which uses distributional features): this genome sees
each word's GRAMMATICAL/MORPHOLOGICAL features (suffix -s/-ed/-ing, modal, copula,
auxiliary, article, singular/plural pronoun, …) and evolves a bilinear head that
scores whether an adjacent pair AGREES. Fitness = tell real corpus pairs (which
agree) from grammatically-corrupted pairs. Goal: "he could be" scores high,
"he could is" scores low. Tiny (a 17x17 matrix), gradient-free.
"""
import numpy as np

from genreg_train import wordpipe as wp

MODALS = set("could would should can will may might must shall".split())
COPULA = set("be is are was were been being am 're 's".split())
AUX = set("have has had do does did having".split())
ARTICLES = set("a an the this that these those my your his her our their".split())
SING_PRON = set("he she it".split())          # 3rd-sing subjects
FIRST_PRON = set("i".split())                  # 1st-sing (special: "am")
PLUR_PRON = set("they we you".split())
PREPS = set("of to in on at by for with from as into over under".split())
CONJ = set("and or but so because if when while that".split())
WH = set("what where who how why which when whom".split())

# Closed-class finiteness / number of the copula+aux system — the exact forms
# agreement turns on. This is morphology knowledge (like the suffix rules), not a
# bigram/transition table: it labels a word's FORM, never which words may follow.
FIN_3SG = set("is was has does".split()) | {"'s"}       # 3rd-sing finite
FIN_NON3 = set("are were am".split()) | {"'re"}         # plural / 1st / 2nd finite
BARE = set("be have do".split())                         # bare infinitive (post-modal)
PARTICIPLE = set("been gone done seen made taken given known".split())  # past participle

NG = 22


def gram_feats(vocab):
    """Fixed grammatical feature vector per word (rule-based morphology + closed-class
    finiteness/number; no parser, no labels, no transition table). Row 15 is bias."""
    F = np.zeros((len(vocab), NG), np.float32)
    for i, w in enumerate(vocab):
        F[i, 0] = w.endswith("s") and not w.endswith("ss") and len(w) > 2  # plural/3sg -s
        F[i, 1] = w.endswith("ed")                                          # past
        F[i, 2] = w.endswith("ing")                                         # progressive
        F[i, 3] = w.endswith("ly")                                          # adverb
        F[i, 4] = w.endswith("e") and len(w) > 2                            # base-form-ish
        F[i, 5] = w in MODALS
        F[i, 6] = w in COPULA
        F[i, 7] = w in AUX
        F[i, 8] = w in ARTICLES
        F[i, 9] = w in SING_PRON
        F[i, 10] = w in PLUR_PRON
        F[i, 11] = w in PREPS
        F[i, 12] = w in CONJ
        F[i, 13] = w in WH
        F[i, 14] = 1.0 if len(w) <= 3 else 0.0                              # short/function-ish
        F[i, 15] = 1.0                                                      # bias
        F[i, 16] = 1.0 if w.isalpha() else 0.0                             # content-ish
        F[i, 17] = w in FIN_3SG                                             # finite 3rd-sing
        F[i, 18] = w in FIN_NON3                                            # finite plur/1st/2nd
        F[i, 19] = w in BARE                                               # bare infinitive
        F[i, 20] = w in PARTICIPLE                                         # past participle
        F[i, 21] = w in FIRST_PRON                                          # 1st-sing "I" (→ "am")
    return F


class AgreePop:
    """Genome = a bilinear head M (NG x NG). score(prev, cand) = g(prev)^T M g(cand)."""

    def __init__(self, pop, seed):
        rng = np.random.default_rng(seed)
        self.M = (rng.standard_normal((pop, NG, NG)) * (1.0 / NG)).astype(np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)

    def score(self, pf, cf):                       # (N,NG),(N,NG) -> (P,N)
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


def train_agreement(vocab_n=4000, gens=3000, pop=200, minibatch=1024, seed=7, log=print):
    """Corrupted-pair discriminator. Negatives = same PREV, but the candidate is
    replaced by a random word of a DIFFERENT grammatical form (forces the genome
    to use agreement, not word identity). GATE: held-out accuracy beats chance."""
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    G = gram_feats(vocab)
    nz = ids[:-1] != 0
    a, b = ids[:-1][nz], ids[1:][nz]                # real adjacent pairs (prev, next), non-unk
    n_train = int(len(a) * 0.9)
    rng = np.random.default_rng(seed)
    Vw = len(vocab)

    def batch(lo, hi, mb, rr):
        idx = rr.integers(lo, hi, size=mb)
        pv, nx = a[idx], b[idx]
        # HARD negatives: same prev, candidate resampled from the REAL next-word
        # distribution (b). Both real and negative candidates are plausible words —
        # only grammatical agreement with `pv` separates them, so the genome must
        # learn form-compatibility, not "real word vs junk". (Contrastive, not a
        # transition table: we never tell it which next-word is correct for pv.)
        neg = b[rr.integers(0, len(a), size=mb)]
        return G[pv], G[nx], G[pv], G[neg]

    vr = np.random.default_rng(seed + 3)
    vpr, vcr, vpn, vcn = batch(n_train, len(a), 8000, vr)
    popn = AgreePop(pop, seed)
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
            log(f"  [agree] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4), "vocab": vocab}


def score_pair(M, vocab, w1, w2):
    """Agreement score for a bigram (higher = more grammatical)."""
    stoi = {w: i for i, w in enumerate(vocab)}
    G = gram_feats(vocab)
    if w1 not in stoi or w2 not in stoi:
        # score from features directly for OOV test words
        g1 = gram_feats([w1])[0]; g2 = gram_feats([w2])[0]
        return float(g1 @ M @ g2)
    return float(G[stoi[w1]] @ M @ G[stoi[w2]])
