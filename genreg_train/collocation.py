"""Collocation-strength genome — ONE job: is this SPECIFIC (content word, next
word) pair a genuine fixed collocation ("depend ON", "look AT", "consist OF"),
tighter than Semantic's loose ±4 window co-occurrence? Absorbs what a separate
"Verb-preposition" genome would have measured — the pipeline has no POS tags,
so cleanly isolating "verb+its governed preposition" from "any tight content-
function bigram" isn't possible; this genome covers the real phenomenon
without a redundant near-duplicate.

Positives: adjacent (content_word, next_word) bigrams whose PMI is genuinely
elevated (observed bigram rate well above what the two words' marginal
frequencies would predict) — not just "co-occurred somewhere in a window"
(that's Semantic's job), but "this SPECIFIC pair sits together far more than
chance." Hard negatives: the same content word paired with a DIFFERENT,
frequency-matched next word — so the genome must learn the specific pairing,
not just "this content word often has SOME function word after it."
"""
import numpy as np
from collections import Counter

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

D = 24


def mine_collocations(vocab_n=4000, min_count=5, pmi_thresh=2.0):
    ids, vocab, stoi = wp.build_word_corpus(vocab_n)
    is_content = np.array([w.isalpha() and w not in al.FUNCTION and len(w) > 2
                           for w in vocab])
    n = len(ids)
    uni = np.bincount(ids, minlength=len(vocab)).astype(np.float64)
    uni_p = uni / uni.sum()
    bigram = Counter()
    for i in range(n - 1):
        a, b = int(ids[i]), int(ids[i + 1])
        if a == 0 or b == 0 or not is_content[a] or is_content[b]:
            continue                                    # content word -> FUNCTION word only
        bigram[(a, b)] += 1
    pairs = []
    for (a, b), c in bigram.items():
        if c < min_count:
            continue
        pmi = np.log((c / n) / (uni_p[a] * uni_p[b] + 1e-12) + 1e-12)
        if pmi > pmi_thresh:
            pairs.append((a, b))
    return pairs, vocab, stoi, ids


def train_collocation(vocab_n=4000, gens=1500, pop=200, seed=7, log=print):
    pairs, vocab, stoi, ids = mine_collocations(vocab_n)
    log(f"mined {len(pairs)} collocation pairs (PMI-thresholded)")
    feat, _ = wp.word_features(vocab_n, D)
    A = np.array([a for a, b in pairs]); B = np.array([b for a, b in pairs])
    is_content = np.array([w.isalpha() and w not in al.FUNCTION and len(w) > 2
                           for w in vocab])
    func_pool = np.array([i for i in range(1, len(vocab)) if not is_content[i]])
    res = gl.train_pairwise(feat, A, B, neg_pool=func_pool, name="colloc",
                            gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def score_pair(champ, vocab, w1, w2, vocab_n=4000):
    feat, _ = wp.word_features(vocab_n, D)
    stoi = {w: i for i, w in enumerate(vocab)}
    if w1 not in stoi or w2 not in stoi:
        return None
    return float(feat[stoi[w1]] @ champ @ feat[stoi[w2]])
