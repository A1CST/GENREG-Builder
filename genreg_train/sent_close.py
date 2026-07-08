"""Sentence-closer genome — ONE job: the last word before a period should be a plausible
sentence-ender (a noun, verb, adjective — content), never a preposition/determiner/
conjunction dangling at the end. Unary classifier over function-type features: words that
actually end sentences in the corpus vs the marginal. Applied at word selection when a
sentence is about to end. ~14 params.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

func_feats = al.func_feats
NF = al.NF


def sentence_final_ids(vocab_n=4000):
    _, vocab, stoi = wp.build_word_corpus(vocab_n)
    toks = wp.decode(wp.corpus_ids()).split()
    final = []
    for t in toks:
        if t and t[-1] in ".!?":
            s = t.strip(".,!?;:'\"")
            if s in stoi and stoi[s] != 0:
                final.append(stoi[s])
    return np.asarray(final, np.int64), vocab, stoi


def train_close(vocab_n=4000, gens=1500, pop=200, seed=7, log=print):
    final, vocab, _ = sentence_final_ids(vocab_n)
    ids, _, _ = wp.build_word_corpus(vocab_n)
    G = func_feats(vocab)
    res = gl.train_unary(G, final, neg_pool=ids[ids != 0], name="close",
                         gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def word_scores(w_vec, vocab):
    """Per-word closer score vector (higher = better sentence ender)."""
    return func_feats(vocab) @ w_vec
