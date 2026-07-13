"""Semantic-compatibility genome — ONE job: adjacent content words should be ones that
actually co-occur in the corpus within a short window. Turns "the horse democracy" into
"the horse stable" — a MEANING-level pair constraint, the first genome that isn't purely
grammatical.

Uses the fixed distributional word features (the 24-dim SVD of co-occurrence, kept OUT of
evolution) and an evolved bilinear head M. Positives = content-word pairs that occur within
+/-WIN of each other in the corpus; hard negatives = random content words (grammatically
fine, semantically unrelated). Distinct from Selection (strict prev-word adjacency,
grammatical fit): this scores topical/semantic co-occurrence over a window. ~580 params.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

WIN = 4          # co-occurrence window (+/- tokens)
D = 24           # distributional feature dim


def content_mask(vocab):
    return np.array([w.isalpha() and w not in al.FUNCTION and len(w) > 2
                     for w in vocab], dtype=bool)


def train_sem(vocab_n=4000, gens=2500, pop=200, seed=7, max_pairs=400000, log=print):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    feat, _ = wp.word_features(vocab_n, D)
    isc = content_mask(vocab)
    rng = np.random.default_rng(seed)
    # positive pairs: content words co-occurring within WIN
    pa, pb = [], []
    cont_pos = np.where(isc[ids])[0]                 # corpus positions holding a content word
    for off in range(1, WIN + 1):
        j = cont_pos[cont_pos + off < len(ids)]
        both = isc[ids[j + off]]
        pa.append(ids[j][both]); pb.append(ids[j + off][both])
    pa = np.concatenate(pa); pb = np.concatenate(pb)
    if len(pa) > max_pairs:
        sel = rng.choice(len(pa), size=max_pairs, replace=False)
        pa, pb = pa[sel], pb[sel]
    neg_pool = ids[isc[ids]]                          # content-token marginal
    res = gl.train_pairwise(feat, pa, pb, neg_pool=neg_pool, name="sem",
                            gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def score_pair(M, vocab, w1, w2, vocab_n=4000):
    feat, _ = wp.word_features(vocab_n, D)
    stoi = {w: i for i, w in enumerate(vocab)}
    if w1 in stoi and w2 in stoi:
        return float(feat[stoi[w1]] @ M @ feat[stoi[w2]])
    return 0.0
