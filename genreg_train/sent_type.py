"""Sentence-type genome — ONE job: does this word plausibly OPEN A QUESTION,
as opposed to a statement? Skeleton-stage (currently thin: only Order/
Alternation/Agreement). Unary classifier over function-type features, same
shape as sent_open.py's opener genome, but a harder discrimination: positive
= words that start QUESTION-ending sentences in the corpus ("will", "do",
"what", "is", ...), hard negative = words that start STATEMENT-ending
sentences (not the general marginal) — so it learns the question/statement
distinction specifically, not just "sentence starter in general" (which
sent_open already covers).
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

func_feats = al.func_feats
NF = al.NF


def sentence_initial_ids_by_type(vocab_n=4000):
    """Walk the corpus; for each sentence-initial word, bucket it by whether
    THIS sentence ends in '?' (question) or '.'/'!' (statement/exclaim)."""
    _, vocab, stoi = wp.build_word_corpus(vocab_n)
    toks = wp.decode(wp.corpus_ids()).split()
    # first pass: find sentence boundaries and each sentence's end punctuation
    q_init, s_init = [], []
    prev_end = True
    pending_first = None
    for t in toks:
        s = t.strip(".,!?;:'\"")
        if prev_end and s in stoi and stoi[s] != 0 and pending_first is None:
            pending_first = stoi[s]
        if t and t[-1] in ".!?":
            if pending_first is not None:
                (q_init if t[-1] == "?" else s_init).append(pending_first)
            pending_first = None
        prev_end = bool(t and t[-1] in ".!?")
    return np.asarray(q_init, np.int64), np.asarray(s_init, np.int64), vocab, stoi


def train_sent_type(vocab_n=4000, gens=1500, pop=200, seed=7, log=print):
    q_init, s_init, vocab, _ = sentence_initial_ids_by_type(vocab_n)
    log(f"question-initial samples: {len(q_init)}, statement-initial: {len(s_init)}")
    G = func_feats(vocab)
    res = gl.train_unary(G, q_init, neg_pool=s_init, name="sent_type",
                         gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    res["question_rate"] = round(len(q_init) / (len(q_init) + len(s_init)), 4)
    return res


def word_scores(w_vec, vocab):
    """Per-word question-opener score (higher = more question-shaped start)."""
    return func_feats(vocab) @ w_vec
