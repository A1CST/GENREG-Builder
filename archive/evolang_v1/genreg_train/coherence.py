"""Coherence genomes — split out of the monolithic "topical drift" idea, which
conflated two different timescales into one compound question:

  * Sentence coherence (LOCAL, fast window): does the next content word fit a
    TIGHT centroid of the last few content words (same clause/sentence)?
  * Theme consistency (THEME, slow window): does it fit a SLOWER centroid over
    a much longer recent span (last ~50 content words, i.e. several sentences
    — a stand-in for "the passage" since the corpus has no document markers)?
    Also absorbs what a separate "Domain purity" genome would have measured.

Both are a bilinear discriminator: positive = (recent-centroid, actual next
content word), hard negative = (SAME centroid, a content word teleported in
from a random distant point in the corpus) — so the genome must learn
"fits this specific local context" not just "is a common content word."

Unlike the other pairwise genomes in this codebase (Semantic, Collocation),
"prev" here is a CONTINUOUS centroid vector, not a single word's feature row
— genelib.train_pairwise is id-indexed, so this reimplements the same
BilinearPop/ga_step training loop directly over precomputed centroid arrays.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import genelib as gl

D = 24
LOCAL_WIN = 8      # fast window: last N content words (~1 clause)
THEME_WIN = 50      # slow window: last N content words (~several sentences)
MIN_GAP = 500       # negatives are teleported at least this many content words away


def _content_mask(vocab):
    return np.array([w.isalpha() and w not in al.FUNCTION and len(w) > 2 for w in vocab])


def mine_centroids(vocab_n=4000, D=D):
    """One pass over the corpus: for every content-word occurrence with enough
    history, compute the LOCAL and THEME centroid feature vectors (mean of the
    preceding content words' features in each window) and the actual next
    content word's id. Returns arrays aligned by occurrence index."""
    ids, vocab, stoi = wp.build_word_corpus(vocab_n)
    feat, _ = wp.word_features(vocab_n, D)
    is_content = _content_mask(vocab)
    content_pos = np.where(is_content[ids])[0]           # positions in `ids` that are content words
    content_ids = ids[content_pos]
    n = len(content_ids)
    cfeat = feat[content_ids]                              # (n, D) feature per content occurrence, in order

    # running sums for O(1) windowed means
    csum = np.concatenate([[np.zeros(D, np.float32)], np.cumsum(cfeat, axis=0)])

    def window_centroid(i, w):
        lo = max(0, i - w)
        cnt = i - lo
        if cnt == 0:
            return None
        return (csum[i] - csum[lo]) / cnt

    local_c, theme_c, targets, tgt_idx = [], [], [], []
    for i in range(THEME_WIN, n):                          # require full theme history
        lc = window_centroid(i, LOCAL_WIN)
        tc = window_centroid(i, THEME_WIN)
        if lc is None:
            continue
        local_c.append(lc); theme_c.append(tc)
        targets.append(content_ids[i]); tgt_idx.append(i)

    return {"local_c": np.asarray(local_c, np.float32),
            "theme_c": np.asarray(theme_c, np.float32),
            "targets": np.asarray(targets, np.int64),
            "tgt_idx": np.asarray(tgt_idx, np.int64),
            "content_ids": content_ids, "cfeat": cfeat, "vocab": vocab, "n": n}


class _CentroidBilinearPop(gl.BilinearPop):
    """Same math as BilinearPop, just trained over continuous centroid rows
    instead of G[word_id] rows — see module docstring."""
    pass


def _train_centroid(centroid, cfeat_all, targets, tgt_idx, n, name, gens=1500, pop=200, seed=7, log=print):
    nf = centroid.shape[1]
    m = len(targets)
    n_train = int(m * 0.9)
    rng = np.random.default_rng(seed)

    def batch(lo, hi, mb, rr):
        idx = rr.integers(lo, hi, size=mb)
        pc = centroid[idx]
        pos_word_idx = tgt_idx[idx]
        pos_cf = cfeat_all[pos_word_idx]
        # negative: teleport to a random content occurrence far from this one
        neg_idx = (pos_word_idx + rr.integers(MIN_GAP, n - MIN_GAP, size=mb)) % n
        neg_cf = cfeat_all[neg_idx]
        return pc, pos_cf, pc, neg_cf

    vr = np.random.default_rng(seed + 3)
    vpr, vcr, vpn, vcn = batch(n_train, m, min(8000, m - n_train), vr)
    popn = _CentroidBilinearPop(pop, nf, seed)
    best_acc, champ = 0.0, None
    for gen in range(1, gens + 1):
        pr, cr, pn, cn = batch(0, n_train, 1024, rng)
        fit, _ = popn.fitness(pr, cr, pn, cn)
        pd = {"M": popn.M, "sigma": popn.sigma}
        wp.ga_step(pd, fit, rng)
        popn.M, popn.sigma = pd["M"], pd["sigma"]
        if gen % 200 == 0 or gen == 1:
            _, acc = popn.fitness(vpr, vcr, vpn, vcn)
            if float(acc[0]) > best_acc:
                best_acc = float(acc[0]); champ = popn.champion(0)
            log(f"  [{name}] gen {gen}: val_acc={acc[0]:.3f}")
    return {"champ": champ, "val_acc": round(best_acc, 4)}


def train_sem_coh_local(vocab_n=4000, gens=1500, pop=200, seed=7, log=print, mined=None):
    m = mined or mine_centroids(vocab_n)
    log(f"coherence-mining occurrences: {m['n']} content words, {len(m['targets'])} training pairs")
    res = _train_centroid(m["local_c"], m["cfeat"], m["targets"], m["tgt_idx"], m["n"],
                          "sem_coh_local", gens, pop, seed, log)
    res["vocab"] = m["vocab"]
    return res, m


def train_sem_coh_theme(vocab_n=4000, gens=1500, pop=200, seed=7, log=print, mined=None):
    m = mined or mine_centroids(vocab_n)
    res = _train_centroid(m["theme_c"], m["cfeat"], m["targets"], m["tgt_idx"], m["n"],
                          "sem_coh_theme", gens, pop, seed, log)
    res["vocab"] = m["vocab"]
    return res, m


def score(champ, centroid_vec, feat, word_id):
    return float(centroid_vec @ champ @ feat[word_id])
