"""Sentence-type genome, exclamation half — ONE job: does this word plausibly
OPEN AN EXCLAMATION, as opposed to a plain statement? Generalizes sent_type.py
(question vs statement) to a third class via a SECOND binary genome instead
of new ternary machinery — same decompose-into-binary-questions pattern used
for Alternation/Agreement this session. Positive = words that start
'!'-ending sentences, hard negative = words that start '.'-ending sentences
(question-ending sentences excluded from this contrast entirely — that's
sent_type's job, not this one's). Feeds the punctuation-as-intent-anchor
architecture: composed with sent_type, a word's [question-affinity,
exclaim-affinity] pair is its full intent profile.

Corpus caution: exclamation marks are rare in this (Wikipedia) corpus —
6,832 occurrences vs 2.7M periods (~0.25%) — a much thinner sample than
sent_type's question-mark mining. Expect a noisier signal; the probe is the
verdict, not val_acc, same discipline as every other genome here.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

func_feats = al.func_feats
NF = al.NF


def sentence_initial_ids_by_exclaim(vocab_n=4000):
    """Walk the corpus; for each sentence-initial word, bucket it by whether
    THIS sentence ends in '!' (exclaim) or '.' (statement) — '?' sentences
    excluded from this contrast."""
    _, vocab, stoi = wp.build_word_corpus(vocab_n)
    toks = wp.decode(wp.corpus_ids()).split()
    e_init, s_init = [], []
    prev_end = True
    pending_first = None
    for t in toks:
        s = t.strip(".,!?;:'\"")
        if prev_end and s in stoi and stoi[s] != 0 and pending_first is None:
            pending_first = stoi[s]
        if t and t[-1] in ".!?":
            if pending_first is not None and t[-1] != "?":
                (e_init if t[-1] == "!" else s_init).append(pending_first)
            pending_first = None
        prev_end = bool(t and t[-1] in ".!?")
    return np.asarray(e_init, np.int64), np.asarray(s_init, np.int64), vocab, stoi


def train_sent_type_exclaim(vocab_n=4000, gens=1500, pop=200, seed=7, log=print):
    e_init, s_init, vocab, _ = sentence_initial_ids_by_exclaim(vocab_n)
    log(f"exclaim-initial samples: {len(e_init)}, statement-initial: {len(s_init)}")
    G = func_feats(vocab)
    res = gl.train_unary(G, e_init, neg_pool=s_init, name="sent_type_exclaim",
                         gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    res["exclaim_rate"] = round(len(e_init) / (len(e_init) + len(s_init)), 4)
    return res


def word_scores(w_vec, vocab):
    """Per-word exclaim-opener score (higher = more exclamation-shaped start)."""
    return func_feats(vocab) @ w_vec
