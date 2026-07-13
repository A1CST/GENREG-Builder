"""Semantic (adjacency), decomposed. The shipped genome (sem_compat.py)
mines content-word pairs across offsets 1-4 together, undifferentiated by
distance, forcing one bilinear head to jointly learn tight collocation-like
fit AND loose topical-window fit. Splitting by distance:

  Sem-adjacent: offset=1 only (immediate neighbor) -- tight, collocation-
                like ("horse stable", not "horse democracy").
  Sem-window:   offsets 2-4 only -- loose topical co-occurrence over a
                short span, excluding the immediate-neighbor signal.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import sem_compat as sc
from genreg_train import genelib as gl

D = 24


def _mine(off_lo, off_hi, vocab_n, max_pairs, seed):
    ids, vocab, _ = wp.build_word_corpus(vocab_n)
    isc = sc.content_mask(vocab)
    rng = np.random.default_rng(seed)
    pa, pb = [], []
    cont_pos = np.where(isc[ids])[0]
    for off in range(off_lo, off_hi + 1):
        j = cont_pos[cont_pos + off < len(ids)]
        both = isc[ids[j + off]]
        pa.append(ids[j][both]); pb.append(ids[j + off][both])
    pa = np.concatenate(pa); pb = np.concatenate(pb)
    if len(pa) > max_pairs:
        sel = rng.choice(len(pa), size=max_pairs, replace=False)
        pa, pb = pa[sel], pb[sel]
    neg_pool = ids[isc[ids]]
    return pa, pb, neg_pool, vocab


def train_sem_adjacent(vocab_n=4000, gens=2000, pop=200, seed=7, max_pairs=400000, log=print):
    pa, pb, neg_pool, vocab = _mine(1, 1, vocab_n, max_pairs, seed)
    feat, _ = wp.word_features(vocab_n, D)
    log(f"sem-adjacent: {len(pa)} distance-1 pairs")
    res = gl.train_pairwise(feat, pa, pb, neg_pool=neg_pool, name="sem-adjacent",
                            gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res


def train_sem_window(vocab_n=4000, gens=2000, pop=200, seed=7, max_pairs=400000, log=print):
    pa, pb, neg_pool, vocab = _mine(2, 4, vocab_n, max_pairs, seed)
    feat, _ = wp.word_features(vocab_n, D)
    log(f"sem-window: {len(pa)} distance-2..4 pairs")
    res = gl.train_pairwise(feat, pa, pb, neg_pool=neg_pool, name="sem-window",
                            gens=gens, pop=pop, seed=seed, log=log)
    res["vocab"] = vocab
    return res
