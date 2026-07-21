"""replicate_coatesng.py — module 47: the NONLINEAR CEILING PROBE.

m46 fed a k-means code to the genome grammar and lost (-5pt) — but that's a
sparse-code-into-low-order-grammar mismatch, not a verdict on nonlinear patch
codes. The genuine question: can a FULL-STRENGTH nonlinear single-layer patch
code beat our 0.77 substrate on this data AT ALL? That is the Coates-Ng
pipeline itself (the gradient-free method that reached ~80% on CIFAR):

  contrast-normalize 6x6 patches -> ZCA whiten -> k-means K atoms
  -> triangle activation f_k = relu(mean_k(dist) - dist_k)   (nonlinear)
  -> dense stride-1 extraction -> sum-pool into a 2x2 grid (K*4 features)
  -> ridge readout.

No evolution, no genome grammar — a clean ceiling probe. All pieces are
gradient-free + label-free feature building (k-means/ZCA are data statistics)
with a closed-form ridge readout, so it stays inside the rules. If this beats
0.77, a nonlinear patch code IS the perception lever and the follow-up is to
let genomes exploit it; if it caps at 0.77 too, the single-layer ceiling is
deeper than any encoding choice.

    python3 replicate/replicate_coatesng.py --K 400 --ps 6
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import genreg_paths                               # noqa: F401

RD = os.path.join(_HERE, "radial_data")
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--ps", type=int, default=6)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pool", type=int, default=2)      # 2x2 quadrant pooling
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr = torch.tensor(z["Xtr"].astype(np.float32) / 255.0, device=dev
                       ).permute(0, 3, 1, 2).contiguous()
    Xte = torch.tensor(z["Xte"].astype(np.float32) / 255.0, device=dev
                       ).permute(0, 3, 1, 2).contiguous()
    ytr = torch.tensor(z["ytr"].astype(np.int64), device=dev)
    yte = torch.tensor(z["yte"].astype(np.int64), device=dev)
    ps, st, K = args.ps, args.stride, args.K
    d = ps * ps * 3
    log(f"[cng] Coates-Ng probe: ps={ps} stride={st} K={K} pool={args.pool}x"
        f"{args.pool}, {torch.cuda.get_device_name(0)}")

    def norm_patches(P):                                # (M,d) contrast norm
        x = P - P.mean(1, keepdim=True)
        return x / (x.var(1, keepdim=True).sqrt() + 1e-2)

    # sample patches for dict learning
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randperm(len(Xtr), generator=g)[:8000].to(dev)
    P = Fn.unfold(Xtr[idx], ps, stride=st).permute(0, 2, 1).reshape(-1, d)
    g2 = torch.Generator(device="cpu").manual_seed(1)
    P = P[torch.randperm(len(P), generator=g2)[:400000].to(dev)]
    Xn = norm_patches(P)
    # ZCA whitening
    mu = Xn.mean(0)
    Xc = Xn - mu
    cov = (Xc.T @ Xc) / len(Xc)
    ev, V = torch.linalg.eigh(cov)
    zca = (V @ torch.diag(1.0 / torch.sqrt(ev + 0.01)) @ V.T)
    Xw = Xc @ zca
    log(f"[cng] patches {tuple(P.shape)} normalized+ZCA ({round(time.time()-t0)}s)")

    # k-means (spherical) on whitened patches
    g3 = torch.Generator(device="cpu").manual_seed(2)
    C = Xw[torch.randperm(len(Xw), generator=g3)[:K].to(dev)].clone()
    C = C / (C.norm(dim=1, keepdim=True) + 1e-8)
    for it in range(args.iters):
        sim = Xw @ C.T
        a = sim.argmax(1)
        newC = torch.zeros_like(C)
        cnt = torch.zeros(K, device=dev)
        newC.index_add_(0, a, Xw)
        cnt.index_add_(0, a, torch.ones(len(a), device=dev))
        m = cnt > 0
        newC[m] = newC[m] / cnt[m].unsqueeze(1)
        C = torch.where(m.unsqueeze(1), newC, C)
        C = C / (C.norm(dim=1, keepdim=True) + 1e-8)
    log(f"[cng] k-means done, {int((cnt>0).sum())}/{K} live atoms "
        f"({round(time.time()-t0)}s)")

    S = Xtr.shape[2]
    H = W = (S - ps) // st + 1
    half = H // args.pool
    Cn = C.contiguous()

    def encode(X, bs=250):
        out = torch.zeros((len(X), K * args.pool * args.pool), device=dev)
        for b in range(0, len(X), bs):
            U = Fn.unfold(X[b:b + bs], ps, stride=st)          # (B,d,L)
            B_ = U.shape[0]
            xw = ((norm_patches(U.permute(0, 2, 1).reshape(-1, d)) - mu) @ zca)
            sim = xw @ Cn.T                                    # (B*L, K)
            # triangle activation: relu(mean_k dist - dist_k); with unit atoms
            # and whitened x, dist_k^2 = |x|^2 + 1 - 2 sim  => rank order by sim
            dist = (xw.pow(2).sum(1, keepdim=True) + 1.0 - 2 * sim).clamp_min(0
                    ).sqrt()                                   # (B*L, K)
            act = torch.relu(dist.mean(1, keepdim=True) - dist)  # triangle
            act = act.reshape(B_, H, W, K)
            # sum-pool into pool x pool grid
            feats = []
            for i in range(args.pool):
                for j in range(args.pool):
                    blk = act[:, i * half:(i + 1) * half,
                              j * half:(j + 1) * half, :]
                    feats.append(blk.sum((1, 2)))
            out[b:b + B_] = torch.cat(feats, 1)
        return out

    Ftr, Fte = encode(Xtr), encode(Xte)
    log(f"[cng] encoded {tuple(Ftr.shape)} ({round(time.time()-t0)}s)")

    # standardize + ridge
    mu2, sd2 = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu2) / sd2
    Fte = (Fte - mu2) / sd2
    n_fit = int(len(Ftr) * 0.8)
    Y = -torch.ones((len(Ftr), 10), device=dev)
    Y[torch.arange(len(Ftr)), ytr] = 1.0
    dd = Ftr.shape[1]
    A = torch.hstack([Ftr[:n_fit], torch.ones(n_fit, 1, device=dev)])
    Gm = (A.T @ A).double()
    best = (-1, None)
    for lam in (1.0, 10.0, 50.0, 200.0, 1000.0):
        W = torch.linalg.solve(Gm + lam * torch.eye(dd + 1, device=dev,
                               dtype=torch.float64),
                               (A.T @ Y[:n_fit]).double()).float()
        Bv = torch.hstack([Ftr[n_fit:], torch.ones(len(Ftr) - n_fit, 1,
                                                    device=dev)])
        va = float(((Bv @ W).argmax(1) == ytr[n_fit:]).float().mean())
        if va > best[0]:
            best = (va, lam)
    lam = best[1]
    Af = torch.hstack([Ftr, torch.ones(len(Ftr), 1, device=dev)])
    W = torch.linalg.solve((Af.T @ Af).double() + lam * torch.eye(
        dd + 1, device=dev, dtype=torch.float64),
        (Af.T @ Y).double()).float()
    Bte = torch.hstack([Fte, torch.ones(len(Fte), 1, device=dev)])
    acc = float(((Bte @ W).argmax(1) == yte).float().mean())
    log(f"[cng] FINAL: K={K} {dd} feats, val {best[0]:.4f} lam {lam} "
        f"TEST {acc:.4f} (evolved substrate 0.7698) ({round(time.time()-t0)}s)")
    json.dump({"module": "coatesng", "K": K, "test": round(acc, 4)},
              open(os.path.join(RD, "replicate_coatesng.json"), "w"))


if __name__ == "__main__":
    main()
