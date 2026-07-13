"""List-parallelism genome — ONE job: when two content words are directly
coordinated ("X and Y" / "X or Y"), are they distributionally the SAME KIND
of thing (dog and cat; ran and jumped) as opposed to two unrelated content
words that just happen to sit on either side of a conjunction?

Positives: adjacent (content_word, content_word) pairs with exactly one
"and"/"or" between them in the corpus. Hard negatives: the same first item
paired with a content word teleported in from elsewhere (frequency-matched
by drawing from the real content-word marginal) — so the genome must learn
"these two are the same TYPE of thing", not just "content word near content
word", which Semantic (adjacency) already covers via a completely different
±4 window mechanism.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

D = 24


def _content_mask(vocab):
    return np.array([w.isalpha() and w not in al.FUNCTION and len(w) > 2 for w in vocab])


def mine_coordinated_pairs(vocab_n=4000, min_count=3):
    ids, vocab, stoi = wp.build_word_corpus(vocab_n)
    is_content = _content_mask(vocab)
    and_id = stoi.get("and"); or_id = stoi.get("or")
    conj_ids = {i for i in (and_id, or_id) if i is not None}
    n = len(ids)
    A, B = [], []
    for i in range(n - 2):
        a, conj, b = int(ids[i]), int(ids[i + 1]), int(ids[i + 2])
        if a == 0 or b == 0 or conj not in conj_ids:
            continue
        if not (is_content[a] and is_content[b]):
            continue
        A.append(a); B.append(b)
    return np.asarray(A, np.int64), np.asarray(B, np.int64), vocab, stoi


def train_parallelism(vocab_n=4000, gens=1500, pop=200, seed=7, log=print):
    A, B, vocab, stoi = mine_coordinated_pairs(vocab_n)
    log(f"mined {len(A)} coordinated content-content pairs")
    feat, _ = wp.word_features(vocab_n, D)
    is_content = _content_mask(vocab)
    content_pool = np.array([i for i in range(1, len(vocab)) if is_content[i]])
    res = gl.train_pairwise(feat, A, B, neg_pool=content_pool, name="parallel",
                            gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def score_pair(champ, vocab, w1, w2, vocab_n=4000):
    feat, _ = wp.word_features(vocab_n, D)
    stoi = {w: i for i, w in enumerate(vocab)}
    if w1 not in stoi or w2 not in stoi:
        return None
    return float(feat[stoi[w1]] @ champ @ feat[stoi[w2]])
