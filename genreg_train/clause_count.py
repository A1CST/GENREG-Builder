"""Clause-count genome — ONE job: does this word plausibly OPEN A COMPOUND
sentence ("X, and Y" / "X, but Y" / "X, so Y"...), as opposed to a SIMPLE
one? Skeleton-stage, same shape as sent_type.py: positive = words that start
sentences containing a mid-sentence ", and"/", but"/", or"/", so"/", yet"
coordination before the sentence-final punctuation, hard negative = words
that start sentences with no such pattern.

A coarser, upstream cousin of the already-cut Clause template genome (that
one failed at the class-trigram level, trying to decide clause structure
DURING generation; this one would act before any classes are chosen at all
— just biasing whether the sentence PLANS to coordinate).
"""
import re

import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

func_feats = al.func_feats
NF = al.NF

COORD_RE = re.compile(r",\s+(and|but|or|so|yet)\b")


def sentence_initial_ids_by_clause(vocab_n=4000):
    _, vocab, stoi = wp.build_word_corpus(vocab_n)
    text = wp.decode(wp.corpus_ids())
    toks = text.split()
    compound_init, simple_init = [], []
    cur_first = None
    cur_words = []
    for t in toks:
        s = t.strip(".,!?;:'\"")
        if cur_first is None and s in stoi and stoi[s] != 0:
            cur_first = stoi[s]
        cur_words.append(t)
        if t and t[-1] in ".!?":
            if cur_first is not None:
                sent_text = " ".join(cur_words)
                (compound_init if COORD_RE.search(sent_text) else simple_init).append(cur_first)
            cur_first = None
            cur_words = []
    return (np.asarray(compound_init, np.int64), np.asarray(simple_init, np.int64),
            vocab, stoi)


def train_clause_count(vocab_n=4000, gens=1500, pop=200, seed=7, log=print):
    compound_init, simple_init, vocab, _ = sentence_initial_ids_by_clause(vocab_n)
    log(f"compound-initial samples: {len(compound_init)}, simple-initial: {len(simple_init)}")
    G = func_feats(vocab)
    res = gl.train_unary(G, compound_init, neg_pool=simple_init, name="clause_count",
                         gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    res["compound_rate"] = round(len(compound_init) / (len(compound_init) + len(simple_init)), 4)
    return res


def word_scores(w_vec, vocab):
    return func_feats(vocab) @ w_vec
