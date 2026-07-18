"""build_topic_stream.py - a TOPIC-labeled word stream for the persistence
operator on language.

The persistence operator (validated on corrupted letters: 0.3835 -> 0.93)
accumulates a per-frame detector's response over a stream: recurring signal
reinforces, transient washes out. The language analog of "the same letter
under 8 corruptions" is "the same TOPIC under W different words": every word
of a window is one noisy view of the topic - topical words fire a
topic-detector wherever they land, function words are the corruption.

Data: 8 topics x 8 Wikipedia articles each, fetched locally via zetifile.
TRAIN and TEST windows come from DISJOINT articles (6 train / 2 test per
topic), so the question is the TOPIC, not article memorization. Each window
is W consecutive in-vocab words, each word its evolved embed_rs vector
(30k vocab, 128-d). Saved in the wf_*-stream cache format
(tr/te = per-timestep lists) plus labels.

  python build_topic_stream.py [--smoke]
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401
import json
import os
import sys
import time

import numpy as np

import zetifile
from radial_lm import _clean

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
OUT = "wf_topic_stream.pt"
W = 12
CAP_PER_ARTICLE = 350                     # window cap: no article dominates

# 8 topics x (6 train + 2 test) articles. Test articles are LAST TWO.
TOPICS = {
    "chemistry": ["Chemistry", "Oxygen", "Hydrogen", "Acid",
                  "Periodic table", "Chemical reaction", "Carbon", "Molecule"],
    "astronomy": ["Astronomy", "Galaxy", "Solar System", "Black hole",
                  "Telescope", "Supernova", "Star", "Planet"],
    "football":  ["Association football", "FIFA World Cup", "Premier League",
                  "Cristiano Ronaldo", "Lionel Messi",
                  "Goalkeeper (association football)",
                  "UEFA Champions League", "Diego Maradona"],
    "music":     ["Music", "Piano", "Guitar", "Orchestra", "Jazz", "Opera",
                  "Violin", "Wolfgang Amadeus Mozart"],
    "medicine":  ["Medicine", "Surgery", "Cancer", "Vaccine", "Antibiotic",
                  "Anatomy", "Infection", "Physician"],
    "law":       ["Law", "Contract", "Jury", "Criminal law", "Constitution",
                  "Judge", "Lawyer", "Common law"],
    "food":      ["Cooking", "Bread", "Cheese", "Wine", "Pasta", "Chocolate",
                  "Rice", "Soup"],
    "geography": ["Mountain", "River", "Desert", "Volcano", "Glacier",
                  "Ocean", "Island", "Earthquake"],
}


def article_windows(title, vocab, cap):
    """Fetch one article, tokenize, keep in-vocab words, return
    non-overlapping W-word windows (as vocab ids)."""
    _, text = zetifile.page_text(title)
    if not text:
        return []
    toks = _clean(text).split()
    ids = [vocab[w] for w in toks if w in vocab]
    wins = [ids[p:p + W] for p in range(0, len(ids) - W + 1, W)]
    return wins[:cap]


def build(smoke=False):
    import torch
    t0 = time.time()
    ze = np.load(os.path.join(RD, "embed_rs.npz"), allow_pickle=True)
    vocab = {str(w): i for i, w in enumerate(ze["vocab"])}
    E = ze["feat"].astype(np.float32)                 # (V, D)
    D = E.shape[1]
    cap = 60 if smoke else CAP_PER_ARTICLE
    names = sorted(TOPICS)
    tr_w, tr_y, te_w, te_y = [], [], [], []
    counts = {}
    for ci, topic in enumerate(names):
        arts = TOPICS[topic]
        for ai, title in enumerate(arts):
            wins = article_windows(title, vocab, cap)
            counts[title] = len(wins)
            dst_w, dst_y = (te_w, te_y) if ai >= 6 else (tr_w, tr_y)
            dst_w.extend(wins)
            dst_y.extend([ci] * len(wins))
            print(f"[topic] {topic:<10} {'TEST ' if ai >= 6 else 'train'} "
                  f"{title}: {len(wins)} windows", flush=True)
    ntr, nte = len(tr_w), len(te_w)
    if min(counts.values()) == 0:
        empty = [t for t, n in counts.items() if n == 0]
        print(f"[topic] WARNING empty articles: {empty}", flush=True)

    rng = np.random.default_rng(0)
    p_tr = rng.permutation(ntr)               # windows arrive grouped by topic:
    p_te = rng.permutation(nte)               # shuffle so val split is honest
    Xtr = np.stack([E[np.array(tr_w[i])] for i in p_tr]).astype(np.float32)
    Xte = np.stack([E[np.array(te_w[i])] for i in p_te]).astype(np.float32)
    ytr = np.array(tr_y, np.int64)[p_tr]
    yte = np.array(te_y, np.int64)[p_te]

    Xt, Xe = torch.tensor(Xtr), torch.tensor(Xte)
    torch.save({"tr": [Xt[:, t].contiguous() for t in range(W)],
                "te": [Xe[:, t].contiguous() for t in range(W)],
                "ytr": torch.tensor(ytr), "yte": torch.tensor(yte),
                "topics": names, "counts": counts, "W": W, "D": D},
               os.path.join(RD, OUT))
    per = {n: int((ytr == i).sum()) for i, n in enumerate(names)}
    print(f"[topic] saved {OUT}: {len(names)} topics, W={W}, D={D}, "
          f"{ntr} train / {nte} test windows (article-disjoint) "
          f"({round(time.time() - t0)}s)", flush=True)
    print(f"[topic] train windows per topic: {json.dumps(per)}", flush=True)
    print("TOPIC STREAM DONE", flush=True)


if __name__ == "__main__":
    build(smoke="--smoke" in sys.argv)
