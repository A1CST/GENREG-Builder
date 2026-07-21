"""replicate_r1probe.py — module 50: R1's CEILING over R0 features.

R1 reads R0's outputs and composes them. Before evolving R1 yet again, bound
what R1 can POSSIBLY reach reading R0 features, with closed-form readouts that
upper-bound the evolved grammar:

  1. linear ridge (R0's own 0.77 baseline)
  2. NONLINEAR readout: RBF random-Fourier features over R0 (sweep bandwidth)
     -> the best nonlinear function of R0's outputs. This is the CEILING of
     any R1 grammar that reads R0 features. If it beats 0.77 by a point+, the
     nonlinear headroom is real and R1 just has to find it. If it caps at
     0.77, NO R1 over R0 outputs can break through -> R1 must read new info.
  3. one-vs-one decomposition: 45 pairwise binary ridges voting -> does asking
     the DECOMPOSED question (which of these two) unlock a point the single
     global head compromises away?

All gradient-free (RFF + ridge are closed-form). Diagnostic only — tells us
what R1's information and fitness must be to earn a real signal.

    python3 replicate/replicate_r1probe.py
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

from radial_evo import _tprims
from radial_evo2 import Env
import radial_stack as rk
from replicate_r1build import union_genomes

RD = os.path.join(_HERE, "radial_data")
LOG = []
CLS = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
       "ship", "truck"]


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
    ytr = torch.tensor(z["ytr"].astype(np.int64), device=dev)
    yte = torch.tensor(z["yte"].astype(np.int64), device=dev)
    N = len(z["ytr"])
    env = Env(torch, dev, Xtr, Xte, max_cached=6)
    gs = union_genomes()
    R0tr = torch.stack([rk.feature_r0(torch, tp, env, g) for g in gs], 1)
    R0te = torch.stack([rk.feature_r0(torch, tp, env, g, test=True)
                        for g in gs], 1)
    mu, sd = R0tr.mean(0), R0tr.std(0) + 1e-6
    Ztr = (R0tr - mu) / sd
    Zte = (R0te - mu) / sd
    n_fit = int(N * 0.8)
    log(f"[probe] R0 features {tuple(Ztr.shape)} ({round(time.time()-t0)}s)")

    Y = -torch.ones((N, 10), device=dev)
    Y[torch.arange(N), ytr] = 1.0

    def ridge_fit(F, Ft, Ytar, lams):
        n, d = F.shape
        A = torch.hstack([F[:n_fit], torch.ones(n_fit, 1, device=dev)])
        Gm = (A.T @ A).double()
        best = (-1, None, None)
        Av = torch.hstack([F[n_fit:], torch.ones(n - n_fit, 1, device=dev)])
        for lam in lams:
            W = torch.linalg.solve(Gm + lam * torch.eye(d + 1, device=dev,
                                   dtype=torch.float64),
                                   (A.T @ Ytar[:n_fit]).double()).float()
            va = float(((Av @ W).argmax(1) == ytr[n_fit:]).float().mean())
            if va > best[0]:
                best = (va, lam, None)
        lam = best[1]
        Af = torch.hstack([F, torch.ones(n, 1, device=dev)])
        W = torch.linalg.solve((Af.T @ Af).double() + lam * torch.eye(
            d + 1, device=dev, dtype=torch.float64),
            (Af.T @ Ytar).double()).float()
        At = torch.hstack([Ft, torch.ones(len(Ft), 1, device=dev)])
        return (At @ W), best[0], lam

    # 1. linear baseline
    pred, va, lam = ridge_fit(Ztr, Zte, Y, (1.0, 3.0, 10.0, 30.0, 100.0))
    lin_acc = float((pred.argmax(1) == yte).float().mean())
    log(f"[probe] 1. LINEAR ridge: val {va:.4f} TEST {lin_acc:.4f}")

    # 2. nonlinear RBF random-Fourier features over R0 (sweep bandwidth)
    D = 8000
    gen = torch.Generator(device=dev).manual_seed(0)
    best_nl = (0.0, None)
    # RBF bandwidth for d-dim standardized data: ||x-y||^2 ~ 2d, so
    # gamma ~ O(1/d) ~ 1/3645 ~ 3e-4; sweep around it
    for gamma in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3):
        Wr = torch.randn(Ztr.shape[1], D, generator=gen, device=dev
                         ) * (2 * gamma) ** 0.5
        b = torch.rand(D, generator=gen, device=dev) * 2 * np.pi
        Ptr = torch.cos(Ztr @ Wr + b) * (2.0 / D) ** 0.5
        Pte = torch.cos(Zte @ Wr + b) * (2.0 / D) ** 0.5
        pred, va, lam = ridge_fit(Ptr, Pte, Y, (0.1, 1.0, 10.0))
        ta = float((pred.argmax(1) == yte).float().mean())
        log(f"[probe]    RFF gamma={gamma}: val {va:.4f} TEST {ta:.4f}")
        if va > best_nl[0]:
            best_nl = (va, ta)
        del Wr, Ptr, Pte
        torch.cuda.empty_cache()
    log(f"[probe] 2. NONLINEAR RFF(R0) best: TEST {best_nl[1]:.4f} "
        f"(vs linear {lin_acc:.4f}, delta {best_nl[1]-lin_acc:+.4f})")

    # 3. one-vs-one decomposition: 45 pairwise binary ridges -> margin vote
    votes = torch.zeros((len(yte), 10), device=dev)
    Af = torch.hstack([Ztr, torch.ones(N, 1, device=dev)])
    At = torch.hstack([Zte, torch.ones(len(yte), 1, device=dev)])
    for i in range(10):
        for j in range(i + 1, 10):
            m = (ytr == i) | (ytr == j)
            mf = m[:n_fit]
            yb = torch.where(ytr[:n_fit][mf] == i, 1.0, -1.0)
            Ai = Af[:n_fit][mf]
            d = Ai.shape[1]
            W = torch.linalg.solve((Ai.T @ Ai).double() + 10.0 * torch.eye(
                d, device=dev, dtype=torch.float64),
                (Ai.T @ yb).double()).float()
            s = (At @ W)                                 # >0 favors i
            votes[:, i] += torch.sigmoid(s * 2)
            votes[:, j] += torch.sigmoid(-s * 2)
    ovo_acc = float((votes.argmax(1) == yte).float().mean())
    log(f"[probe] 3. ONE-VS-ONE (45 pairwise): TEST {ovo_acc:.4f} "
        f"(vs linear {lin_acc:.4f}, delta {ovo_acc-lin_acc:+.4f})")

    # confusion structure: where do R0's errors concentrate?
    pl = pred.argmax(1) if False else None
    predL, _, _ = ridge_fit(Ztr, Zte, Y, (10.0,))
    pl = predL.argmax(1)
    conf = torch.zeros((10, 10), device=dev)
    for a, b in zip(yte.tolist(), pl.tolist()):
        if a != b:
            conf[a, b] += 1
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            pairs.append((float(conf[i, j] + conf[j, i]), CLS[i], CLS[j]))
    pairs.sort(reverse=True)
    log("[probe] top confused pairs (R0 linear): " +
        ", ".join(f"{a}-{b}:{int(n)}" for n, a, b in pairs[:6]))
    log(f"[probe] done ({round(time.time()-t0)}s)")
    json.dump({"linear": round(lin_acc, 4), "rff": round(best_nl[1], 4),
               "ovo": round(ovo_acc, 4)},
              open(os.path.join(RD, "replicate_r1probe.json"), "w"))


if __name__ == "__main__":
    main()
