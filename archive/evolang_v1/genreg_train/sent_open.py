"""Sentence-opener genome — ONE job: the first word of a sentence should be a plausible
sentence-starter. Kills the current "Of … / To … / And …" openings. Unary classifier over
function-type features: words that actually begin sentences in the corpus (pronouns,
articles, wh, subordinators, some conjunctions) vs the marginal. Applied at word selection
when a new sentence starts (position 0). ~14 params.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

func_feats = al.func_feats
NF = al.NF


def sentence_initial_ids(vocab_n=4000):
    _, vocab, stoi = wp.build_word_corpus(vocab_n)
    toks = wp.decode(wp.corpus_ids()).split()
    init = []
    prev_end = True
    for t in toks:
        s = t.strip(".,!?;:'\"")
        if prev_end and s in stoi and stoi[s] != 0:
            init.append(stoi[s])
        prev_end = bool(t and t[-1] in ".!?")
    return np.asarray(init, np.int64), vocab, stoi


def train_open(vocab_n=4000, gens=1500, pop=200, seed=7, log=print):
    init, vocab, _ = sentence_initial_ids(vocab_n)
    ids, _, _ = wp.build_word_corpus(vocab_n)
    G = func_feats(vocab)
    res = gl.train_unary(G, init, neg_pool=ids[ids != 0], name="open",
                         gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def word_scores(w_vec, vocab):
    """Per-word opener score vector (higher = better sentence starter)."""
    return func_feats(vocab) @ w_vec
