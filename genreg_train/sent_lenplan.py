"""Sentence-length-plan genome — ONE job: does this word plausibly OPEN A
LONG sentence, as opposed to a SHORT one? Skeleton-stage. Unary classifier
over function-type features, same shape as sent_type.py: positive = words
that start sentences whose word count is above the corpus median, hard
negative = words that start sentences at/below the median (not the general
marginal) — so it learns the length-shaped distinction specifically, not
just "sentence starter in general" (sent_open already covers that).

Boundary currently decides length AFTER generation, one word at a time, with
no upstream signal. This gives the opener a bias toward "this sentence is
going to run long" vs "this one should wrap up quick" before a single word
after the opener has been chosen.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

func_feats = al.func_feats
NF = al.NF


def sentence_initial_ids_by_length(vocab_n=4000):
    """Walk the corpus; bucket each sentence-initial word by whether THIS
    sentence's word count is above (long) or at/below (short) the corpus
    median sentence length."""
    _, vocab, stoi = wp.build_word_corpus(vocab_n)
    toks = wp.decode(wp.corpus_ids()).split()
    sentences = []           # (first_id, length) per sentence
    cur_first = None
    cur_len = 0
    for t in toks:
        s = t.strip(".,!?;:'\"")
        if cur_first is None and s in stoi and stoi[s] != 0:
            cur_first = stoi[s]
        if s:
            cur_len += 1
        if t and t[-1] in ".!?":
            if cur_first is not None:
                sentences.append((cur_first, cur_len))
            cur_first = None
            cur_len = 0
    lengths = np.array([n for _, n in sentences])
    median = float(np.median(lengths))
    long_init = np.asarray([f for f, n in sentences if n > median], np.int64)
    short_init = np.asarray([f for f, n in sentences if n <= median], np.int64)
    return long_init, short_init, median, vocab, stoi


def train_sent_lenplan(vocab_n=4000, gens=1500, pop=200, seed=7, log=print):
    long_init, short_init, median, vocab, _ = sentence_initial_ids_by_length(vocab_n)
    log(f"median sentence length: {median} words")
    log(f"long-initial samples: {len(long_init)}, short-initial: {len(short_init)}")
    G = func_feats(vocab)
    res = gl.train_unary(G, long_init, neg_pool=short_init, name="sent_lenplan",
                         gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    res["median_len"] = median
    return res


def word_scores(w_vec, vocab):
    """Per-word long-sentence-opener score (higher = more long-sentence-shaped)."""
    return func_feats(vocab) @ w_vec
