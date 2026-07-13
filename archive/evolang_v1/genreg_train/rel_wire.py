"""Crosswalk for the standalone Wikipedia relation genomes (hypernym, meronym,
unified synonym/antonym — see genomes.txt) into the /evolang pipeline's word
space. Those genomes were trained on Wikipedia's 30K-word vocab + 128d SVD
features; the pipeline generates over its own 4000-word novel-corpus vocab.
Rather than retrain on the pipeline's smaller vocab (losing the Wikipedia
signal these genomes need), build a per-pipeline-word lookup into the
Wikipedia feature space: most common English words are common in both
corpora. Words absent from the Wikipedia vocab get a zero feature row, which
zeroes their re-rank contribution (graceful degradation, same pattern used
elsewhere for missing genomes).
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_FEATS = os.path.join(ROOT, "corpora", "wikipedia", "wiki_feats.npz")
CHAMP_DIR = os.path.join(ROOT, "corpora", "wikipedia", "build")


def load_relation_genomes(pipeline_vocab):
    """Returns a dict with whatever of {hyper, mero, synant} is available on
    disk: {'M': ..., 'feat': (len(pipeline_vocab), D), 'coverage': float}.
    Missing artifacts (not yet exported) are simply absent from the dict —
    callers check membership, same as self.champs elsewhere in the service."""
    out = {}
    if not os.path.exists(WIKI_FEATS):
        return out
    d = np.load(WIKI_FEATS, allow_pickle=True)
    wiki_vocab = list(d["vocab"]); wiki_feat = d["feat"]
    wiki_stoi = {w: i for i, w in enumerate(wiki_vocab)}

    hits = np.array([wiki_stoi.get(w, -1) for w in pipeline_vocab])
    covered = hits >= 0
    coverage = float(covered.mean())
    base_feat = np.zeros((len(pipeline_vocab), wiki_feat.shape[1]), np.float32)
    base_feat[covered] = wiki_feat[hits[covered]]

    hyper_p = os.path.join(CHAMP_DIR, "hypernym_champ.npz")
    if os.path.exists(hyper_p):
        out["hyper"] = {"M": np.load(hyper_p)["M"], "feat": base_feat, "coverage": coverage}

    mero_p = os.path.join(CHAMP_DIR, "meronym_champ.npz")
    if os.path.exists(mero_p):
        out["mero"] = {"M": np.load(mero_p)["M"], "feat": base_feat, "coverage": coverage}

    synant_p = os.path.join(CHAMP_DIR, "synant_champ.npz")
    if os.path.exists(synant_p):
        z = np.load(synant_p)
        cfeat = z["cfeat"]
        base_cfeat = np.zeros((len(pipeline_vocab), cfeat.shape[1]), np.float32)
        base_cfeat[covered] = cfeat[hits[covered]]
        synant_feat = np.concatenate([base_feat, base_cfeat], axis=1)
        out["synant"] = {"M": z["M"], "feat": synant_feat, "coverage": coverage}

    return out
