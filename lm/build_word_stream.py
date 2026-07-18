"""build_word_stream.py - a WORD-level stream for the grammar discriminator.

The two-run conclusion: temporal composition earns its keep on ORDER/STRUCTURE
tasks, not next-word. The cleanest structural task for language is GRAMMAR:
real word sequences vs locally-shuffled copies (same words, order broken). This
builds long word sequences from the corpus, each word represented by its
evolved embed_rs vector, saved in the cache format radial_temporal_shuffle
consumes. Grammar lives in the ORDER (relative-time transitions), which the
per-word embedding table does NOT encode - so temporal composition, not the
table, is what must detect it. The local-shuffle control keeps absolute
position from cheating.

  python build_word_stream.py [--smoke]
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os
import sys
import time

import numpy as np

from radial_lm import _clean

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
OUT = "wf_word_stream.pt"


def build(smoke=False, W=10, n_train=40000, n_test=8000):
    import torch
    t0 = time.time()
    ze = np.load(os.path.join(RD, "embed_rs.npz"), allow_pickle=True)
    vocab = {str(w): i for i, w in enumerate(ze["vocab"])}
    E = ze["feat"].astype(np.float32)                 # (V, D)
    D = E.shape[1]
    print(f"[word] embed_rs vocab={len(vocab)} dim={D}", flush=True)
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        f.seek(20_000_000)
        toks = _clean(f.read(28_000_000)).split()
    if smoke:
        n_train, n_test = 1500, 500
    n = n_train + n_test
    X = np.zeros((n, W, D), np.float32)
    got = 0
    rng = np.random.default_rng(0)
    order = rng.permutation(len(toks) - W)
    for p in order:
        seq = toks[p:p + W]
        ids = [vocab.get(w) for w in seq]
        if all(i is not None for i in ids):          # every word has a vector
            for t, i in enumerate(ids):
                X[got, t] = E[i]
            got += 1
            if got == n:
                break
    if got < n:
        print(f"[word] only {got}/{n} all-in-vocab {W}-grams - using {got}",
              flush=True)
        X = X[:got]
        n_train = int(got * (n_train / n))
    Xt = torch.tensor(X)
    tr = [Xt[:n_train, t].contiguous() for t in range(W)]
    te = [Xt[n_train:, t].contiguous() for t in range(W)]
    torch.save({"tr": tr, "te": te}, os.path.join(RD, OUT))
    print(f"[word] saved {OUT}: W={W} words, D={D}, "
          f"{n_train} train / {len(Xt) - n_train} test ({round(time.time()-t0)}s)",
          flush=True)
    print("WORD STREAM DONE", flush=True)


if __name__ == "__main__":
    build(smoke="--smoke" in sys.argv)
