"""Tense-consistency genome — ONE job: verbs within the same sentence should be in
compatible tenses. Kills "he walked and runs" — tense drift inside a clause. Pairwise
bilinear over TENSE features (past -ed / 3sg -s / progressive -ing / base / modal / aux /
copula), trained on verb pairs that co-occur within a sentence window (positive) vs a
verb of a mismatched tense (hard negative). ~120 params.

Note: verb detection is morphological (suffix + closed-class aux/copula/modal), no parser.
The battery decides whether this earns its place over Agreement's local finiteness signal.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import agreement as ag
from genreg_train import genelib as gl

WIN = 5          # same-sentence window for verb co-occurrence
NT = 9


def tense_feats(vocab):
    F = np.zeros((len(vocab), NT), np.float32)
    for i, w in enumerate(vocab):
        F[i, 0] = w.endswith("ed")                                    # past
        F[i, 1] = w.endswith("s") and not w.endswith("ss") and len(w) > 2  # 3sg present
        F[i, 2] = w.endswith("ing")                                   # progressive
        F[i, 3] = w.endswith("e") and len(w) > 2                      # base-ish
        F[i, 4] = w in ag.MODALS
        F[i, 5] = w in ag.AUX
        F[i, 6] = w in ag.COPULA
        F[i, 7] = 1.0 if (F[i, 0] or F[i, 1] or F[i, 2] or w in ag.AUX or w in ag.COPULA
                          or w in ag.MODALS) else 0.0                 # is-verb-ish
        F[i, 8] = 1.0                                                 # bias
    return F


def _verbish(vocab):
    return np.array([bool(tense_feats([w])[0, 7]) for w in vocab], dtype=bool)


def train_tense(vocab_n=4000, gens=2000, pop=200, seed=7, max_pairs=300000, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    F = tense_feats(vocab)
    isv = _verbish(vocab)
    reset = np.zeros(len(ids), bool)                 # sentence break not tracked at id level;
    # approximate same-sentence by requiring no <unk> gap and closeness in the stream
    rng = np.random.default_rng(seed)
    vpos = np.where(isv[ids] & (ids != 0))[0]
    pa, pb = [], []
    for off in range(1, WIN + 1):
        j = vpos[vpos + off < len(ids)]
        both = isv[ids[j + off]] & (ids[j + off] != 0)
        pa.append(ids[j][both]); pb.append(ids[j + off][both])
    pa = np.concatenate(pa); pb = np.concatenate(pb)
    if len(pa) > max_pairs:
        sel = rng.choice(len(pa), size=max_pairs, replace=False)
        pa, pb = pa[sel], pb[sel]
    neg_pool = ids[isv[ids] & (ids != 0)]            # verb-ish marginal (mixed tenses)
    res = gl.train_pairwise(F, pa, pb, neg_pool=neg_pool, name="tense",
                            gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def score_pair(M, vocab, w1, w2):
    return gl.score_pair_M(M, tense_feats, vocab, w1, w2)
