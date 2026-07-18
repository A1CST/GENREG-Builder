"""build_char_stream.py - the CHAR-level stream for the temporal experiment.

The word-level run gave the temporal space only a 4-step window - too short a
time axis to decide anything. The character stream is 8x longer and is where
sequential regularity (spelling, morphology) actually lives. The 32 char tiles
per example (4 words x 8 padded letters) are ALREADY in kid_next as
(N, P_C, L_MAX, 32, 32); this reads each tile through the frozen stage-A LETTER
eye (kid_modelA, no B composition) and window-pools each A-genome to one scalar
per char -> a (N, T=32, C=nA) stream, saved in the same cache format
radial_temporal.py already consumes.

  python build_char_stream.py [--smoke]
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
import os
import sys
import time

import numpy as np

from radial_evo import _tprims, _STOP
from radial_evo2 import Env
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
OUT = "wf_char_stream.pt"


def build(smoke=False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    rk.GRID = 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    with open(os.path.join(RD, "kid_modelA.json")) as f:
        A = json.load(f)["spaces"][0]                 # the letter eye, space 0
    z = np.load(os.path.join(RD, "kid_next.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    Ntr = Xtr.shape[0]; Nte = Xte.shape[0]
    P_C, L_MAX = Xtr.shape[1], Xtr.shape[2]
    T = P_C * L_MAX                                    # 32 char steps
    if smoke:
        Xtr, Xte = Xtr[:1500], Xte[:500]
        Ntr, Nte = 1500, 500
        A = A[:40]
        print(f"[char] SMOKE ({len(A)} eye genomes)", flush=True)

    rows_tr = np.repeat(Xtr.reshape(Ntr * T, 32, 32)[..., None], 3, axis=3)
    rows_te = np.repeat(Xte.reshape(Nte * T, 32, 32)[..., None], 3, axis=3)
    env = Env(torch, dev, rows_tr, rows_te, max_cached=1)
    A_sorted = sorted(range(len(A)), key=lambda k: A[k].get("ps", 8))
    print(f"[char] {len(A)} letter-eye genomes over {Ntr}x{T} + {Nte}x{T} "
          f"tiles -> {T}-step char stream", flush=True)

    def sweep(test):
        N = Nte if test else Ntr
        cols = []
        for i, k in enumerate(A_sorted):              # scale-sorted: one build/ps
            c = rk.feature_r0(torch, tp, env, A[k], test=test)   # (N*T,) pooled
            cols.append(torch.nan_to_num(c).half().cpu())
            if (i + 1) % 100 == 0:
                print(f"  [char] {'te' if test else 'tr'} genome {i+1}/{len(A)} "
                      f"({round(time.time()-t0)}s)", flush=True)
        M = torch.stack(cols, 1).view(N, T, len(A))   # (N, T, nA)
        return [M[:, t].contiguous() for t in range(T)]

    tr = sweep(False)
    te = sweep(True)
    torch.save({"tr": tr, "te": te}, os.path.join(RD, OUT))
    print(f"[char] saved {OUT}: T={T} steps, C={len(A)} channels "
          f"({round(time.time()-t0)}s)", flush=True)
    print("CHAR STREAM DONE", flush=True)


if __name__ == "__main__":
    build(smoke="--smoke" in sys.argv)
