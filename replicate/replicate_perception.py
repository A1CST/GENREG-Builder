"""replicate_perception.py — module 44: the perception primitive. Does
preserving SPATIAL LAYOUT break 0.77?

Every readout/memory/reconstruction lever confirmed the 0.77 wall is the
substrate's clean-image information. But the substrate collapses each genome
to ONE scalar (global soft-window pool) — position is thrown away. Three
converging clues say that's the loss: reconstruction is coarse-but-spatial,
per-view helped by not destroying structure, and the word signatures are
LOCALIZED detectors whose position was being averaged out. Coates-Ng's ~80%
specifically kept 2x2 spatial pooling.

Cheap decisive test: take the SAME 3,645 union genomes and read their SPATIAL
GRID (GxG cells) instead of their scalar. If grid >> scalar, spatial layout
is the missing information and the perception lever is real — no re-evolution
needed to find out.

    python3 replicate/replicate_perception.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
for _p in ("replicate", "radial", "ocr"):
    sys.path.insert(0, os.path.join(_HERE, _p))
import genreg_paths                               # noqa: F401

from radial_evo import _tprims, _ridge_soft
from radial_evo2 import Env
import radial_stack as rk
from replicate_cifar import load_cifar
from replicate_r1build import union_genomes

RD = os.path.join(_HERE, "radial_data")
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    tp = _tprims(torch)
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"].astype(np.int64), z["yte"].astype(np.int64)
    N = len(ytr)
    env = Env(torch, dev, Xtr, Xte, max_cached=6)
    gs = union_genomes()
    n_fit = int(N * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((N, 10), device=dev)
    Yfull[torch.arange(N), torch.tensor(ytr, device=dev)] = 1.0

    def test_of(Ftr, Fte):
        bl, bv = 3.0, -1.0
        for lam in (1.0, 3.0, 10.0, 30.0, 100.0):
            _, a = _ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf, yv, lam=lam)
            if a > bv:
                bl, bv = lam, a
        n, d = Ftr.shape
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        A = torch.hstack([(Ftr - mu) / sd, torch.ones(n, 1, device=dev)])
        G = (A.T @ A).double() + bl * torch.eye(d + 1, device=dev,
                                                dtype=torch.float64)
        W = torch.linalg.solve(G, (A.T @ Yfull).double()).float()
        B = torch.hstack([(Fte - mu) / sd, torch.ones(len(Fte), 1, device=dev)])
        acc = float(((B @ W).argmax(1) == yte_t).float().mean())
        return acc, bv, d

    # scalar baseline (global pool — the current substrate)
    S_tr = torch.stack([rk.feature_r0(torch, tp, env, g) for g in gs], 1)
    S_te = torch.stack([rk.feature_r0(torch, tp, env, g, test=True)
                        for g in gs], 1)
    a0, v0, d0 = test_of(S_tr, S_te)
    log(f"[perc] SCALAR (global pool): TEST {a0:.4f} ({d0} feats, "
        f"{round(time.time()-t0)}s)")

    # spatial grids at increasing resolution
    for G in (2, 3, 4):
        rk.GRID = G
        Gtr = torch.stack([rk.feature_r0(torch, tp, env, g, want_grid=True)
                           for g in gs], 1).reshape(N, -1)
        Gte = torch.stack([rk.feature_r0(torch, tp, env, g, test=True,
                                         want_grid=True)
                           for g in gs], 1).reshape(len(yte), -1)
        a, v, d = test_of(Gtr, Gte)
        log(f"[perc] GRID {G}x{G} (spatial layout kept): TEST {a:.4f} "
            f"({d} feats, delta over scalar {a - a0:+.4f}, "
            f"{round(time.time()-t0)}s)")
        del Gtr, Gte
        torch.cuda.empty_cache()

    json.dump({"module": "perception", "scalar": round(a0, 4)},
              open(os.path.join(RD, "replicate_perception.json"), "w"))
    log("[perc] done")


if __name__ == "__main__":
    main()
