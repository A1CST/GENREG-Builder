"""Determiner-binding genome — ONE job: a determiner (the, a, my, his, this…) must be
bound to a noun phrase — followed by an adjective or noun (content), or another modifier,
never left orphaned before a preposition/conjunction/verb. Kills "the the", "the and",
"the of" — orphaned determiners.

Surgical specialization of alternation: same function-type features, but trained ONLY on
pairs whose previous word is an article/determiner, so the evolved head concentrates on
what legally follows a determiner (the->great yes, the->house yes, the->of no, the->the no).
Pairwise bilinear, gradient-free, ~200 params.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

func_feats = al.func_feats
NF = al.NF
DETS = al.ARTICLES | al.DET


def train_det(vocab_n=4000, gens=2500, pop=200, seed=7, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    G = func_feats(vocab)
    is_det = np.array([w in DETS for w in vocab], dtype=bool)
    nz = ids[:-1] != 0
    a, b = ids[:-1][nz], ids[1:][nz]
    mask = is_det[a]                                 # only determiner contexts
    a, b = a[mask], b[mask]
    res = gl.train_pairwise(G, a, b, neg_pool=ids[ids != 0], name="det",
                            gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def score_pair(M, vocab, w1, w2):
    return gl.score_pair_M(M, func_feats, vocab, w1, w2)
