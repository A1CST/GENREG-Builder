"""Preposition-completion genome — ONE job: a preposition must be completed by an
object (a content word, or a determiner/possessive that heads one), never another
bare preposition/conjunction. Kills "of and", "to to", "for of" — dangling prepositions.

Surgical specialization of alternation: same function-type features (altern.func_feats),
but trained ONLY on pairs whose previous word is a preposition, so the evolved head
concentrates entirely on what legally follows a preposition (of->the yes, of->house yes,
of->and no, to->to no). Pairwise bilinear, gradient-free, ~200 params.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

func_feats = al.func_feats
NF = al.NF


def train_prep(vocab_n=4000, gens=2500, pop=200, seed=7, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    G = func_feats(vocab)
    prep_ids = np.array([i for i, w in enumerate(vocab) if w in al.PREPS], dtype=np.int64)
    is_prep = np.zeros(len(vocab), bool); is_prep[prep_ids] = True
    nz = ids[:-1] != 0
    a, b = ids[:-1][nz], ids[1:][nz]
    mask = is_prep[a]                                # only preposition contexts
    a, b = a[mask], b[mask]
    res = gl.train_pairwise(G, a, b, neg_pool=ids[ids != 0], name="prep",
                            gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def score_pair(M, vocab, w1, w2):
    return gl.score_pair_M(M, func_feats, vocab, w1, w2)
