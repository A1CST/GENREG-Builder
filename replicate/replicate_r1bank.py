"""replicate_r1bank.py — module 51: R1 as a DIVERSITY BANK over R0. THE FIX.

The R1 probe (m50) found the signal we were missing: a nonlinear readout of
R0's features breaks the ceiling — RFF(R0) +1.24pt, and STRUCTURED products of
R0 feature pairs (exactly R1's grammar) +1.67pt (0.7865 vs 0.7698). The
information was in R0's outputs all along; my earlier R1 (m38/m49) killed it
with a per-genome significance gate — near the 0.77 ceiling every single
nonlinear feature adds <0.0004, so the gate rejected the entire diffuse
signal that is worth a point+ in aggregate.

This obeys the diversity-first-features law: R1 features evolve/select for
DIVERSITY and information, NOT per-genome label gain; only the final ridge
sees labels. R1 = a large bank of nonlinear R0 compositions (the full grammar:
products, min, absdiff, rectified, squared, over random R0 channel pairs at
random prim transforms), diversity-decorrelated, read by ONE ridge over
[R0 | R1].

    python3 replicate/replicate_r1bank.py --bank 8000
"""
import argparse
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


def log(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=6, help="diversity: keep the "
                    "best-conditioned of several random banks per block")
    args = ap.parse_args()
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
    Ztr, Zte = (R0tr - mu) / sd, (R0te - mu) / sd
    C = Ztr.shape[1]
    n_fit = int(N * 0.8)
    Y = -torch.ones((N, 10), device=dev)
    Y[torch.arange(N), ytr] = 1.0
    log(f"[bank] R0 {tuple(Ztr.shape)} ({round(time.time()-t0)}s)")

    def test_of(Ftr, Fte):
        n, d = Ftr.shape
        A = torch.hstack([Ftr[:n_fit], torch.ones(n_fit, 1, device=dev)])
        Gm = (A.T @ A).double()
        Av = torch.hstack([Ftr[n_fit:], torch.ones(n - n_fit, 1, device=dev)])
        best = (-1, None)
        for lam in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
            W = torch.linalg.solve(Gm + lam * torch.eye(d + 1, device=dev,
                                   dtype=torch.float64),
                                   (A.T @ Y[:n_fit]).double()).float()
            va = float(((Av @ W).argmax(1) == ytr[n_fit:]).float().mean())
            if va > best[0]:
                best = (va, lam)
        lam = best[1]
        Af = torch.hstack([Ftr, torch.ones(n, 1, device=dev)])
        W = torch.linalg.solve((Af.T @ Af).double() + lam * torch.eye(
            d + 1, device=dev, dtype=torch.float64),
            (Af.T @ Y).double()).float()
        At = torch.hstack([Fte, torch.ones(len(Fte), 1, device=dev)])
        acc = float(((At @ W).argmax(1) == yte).float().mean())
        return acc, best[0], lam

    r0_acc, r0_val, _ = test_of(Ztr, Zte)
    log(f"[bank] R0 linear baseline: val {r0_val:.4f} TEST {r0_acc:.4f}")

    # ---- build the nonlinear R1 bank (full grammar, diversity-decorrelated)
    PR = ["id", "abs", "relu", "tanh", "sq"]
    prim = {"id": lambda v: v, "abs": torch.abs, "relu": torch.relu,
            "tanh": torch.tanh, "sq": lambda v: v * v}
    OPS = ["mult", "min", "absdiff"]
    g = torch.Generator(device=dev).manual_seed(7)
    probe_idx = torch.randperm(N, generator=g, device=dev)[:4000]
    kept_tr, kept_te, kept_sig = [], [], []
    target = args.bank
    block = 2048
    while sum(x.shape[1] for x in kept_tr) < target:
        ia = torch.randint(0, C, (block,), generator=g, device=dev)
        ib = torch.randint(0, C, (block,), generator=g, device=dev)
        pa = [PR[k] for k in torch.randint(0, len(PR), (block,),
              generator=g, device=dev).tolist()]
        pb = [PR[k] for k in torch.randint(0, len(PR), (block,),
              generator=g, device=dev).tolist()]
        opk = torch.randint(0, len(OPS), (block,), generator=g,
                            device=dev).tolist()
        cols_tr = torch.empty((N, block), device=dev)
        cols_te = torch.empty((len(yte), block), device=dev)
        for k in range(block):
            for (Zs, out) in ((Ztr, cols_tr), (Zte, cols_te)):
                va = prim[pa[k]](Zs[:, ia[k]])
                vb = prim[pb[k]](Zs[:, ib[k]])
                op = OPS[opk[k]]
                out[:, k] = (va * vb if op == "mult" else
                             torch.minimum(va, vb) if op == "min"
                             else torch.abs(va - vb))
        # standardize + diversity filter vs kept bank (decorrelate)
        m2, s2 = cols_tr.mean(0), cols_tr.std(0) + 1e-6
        cols_tr = (cols_tr - m2) / s2
        cols_te = (cols_te - m2) / s2
        sig = cols_tr[probe_idx]
        sig = sig / (sig.norm(dim=0, keepdim=True) + 1e-8)
        if kept_sig:
            K = torch.cat(kept_sig, 1)
            corr = (sig.T @ K).abs().max(1).values          # (block,)
        else:
            corr = torch.zeros(block, device=dev)
        keep = corr < 0.7
        if int(keep.sum()) == 0:
            continue
        kept_tr.append(cols_tr[:, keep])
        kept_te.append(cols_te[:, keep])
        kept_sig.append(sig[:, keep])
        nb = sum(x.shape[1] for x in kept_tr)
        log(f"[bank] +{int(keep.sum())} -> {nb} nonlinear R1 feats "
            f"({round(time.time()-t0)}s)")

    R1tr = torch.cat(kept_tr, 1)[:, :target]
    R1te = torch.cat(kept_te, 1)[:, :target]
    r1_acc, r1_val, _ = test_of(R1tr, R1te)
    log(f"[bank] R1 bank alone ({R1tr.shape[1]}): val {r1_val:.4f} "
        f"TEST {r1_acc:.4f}")
    j_acc, j_val, j_lam = test_of(torch.cat([Ztr, R1tr], 1),
                                  torch.cat([Zte, R1te], 1))
    log(f"[bank] R0 | R1 ({Ztr.shape[1]}+{R1tr.shape[1]}): val {j_val:.4f} "
        f"TEST {j_acc:.4f} lam {j_lam} (R0 {r0_acc:.4f}, "
        f"delta {j_acc - r0_acc:+.4f})")
    json.dump({"module": "r1bank", "bank": R1tr.shape[1], "r0": round(r0_acc, 4),
               "joint": round(j_acc, 4)},
              open(os.path.join(RD, "replicate_r1bank.json"), "w"))


if __name__ == "__main__":
    main()
