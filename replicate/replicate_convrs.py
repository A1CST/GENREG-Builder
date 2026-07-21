"""replicate_convrs.py — module 62: a CNN built from radial spaces.

The bet: if RS can build an embedding space, it can build the components of a
CNN. A conv layer = one RS (a bank of evolved filter-genomes); a filter = one
genome applied at EVERY spatial position (weight-shared); a feature map = its
response across positions; depth = stack RS on the MAPS, not on pooled scalars
(pooling before hand-off is what killed every earlier stack and pinned us to
the single-layer ~0.77 ceiling). Pool + head only at the very end.

Every layer is label-free (contrastive: a filter's pooled response is stable
across two augmented views) under the energy economy. The decisive test: does
LAYER 2, reading layer 1's full-resolution maps through a local 3x3 window,
ADD over layer 1 -- i.e. does spatial hierarchy finally earn.

    python3 replicate/replicate_convrs.py --l1_rounds 30 --l2_rounds 30
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

from radial_evo import _tprims, _ridge_soft
import radial_evo2 as re2
from radial_evo2 import Env, SCALES
import radial_stack as rk
from replicate_langspace import _fp
from replicate_langstack import quality
from replicate_wordspace import energy_evolve

RD = os.path.join(_HERE, "radial_data")


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--l1_rounds", type=int, default=30)
    ap.add_argument("--l2_rounds", type=int, default=30)
    ap.add_argument("--l1_cap", type=int, default=400)
    ap.add_argument("--l2_cap", type=int, default=400)
    ap.add_argument("--grid", type=int, default=8)      # keep maps SPATIAL
    ap.add_argument("--pool", type=int, default=4)      # final pool grid
    ap.add_argument("--n", type=int, default=15000,
                    help="train images (subset keeps full-res maps in memory)")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    rk.GRID = args.grid
    G = rk.GRID
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    sub = np.random.default_rng(0).permutation(len(z["ytr"]))[:args.n]
    sub.sort()                                           # balanced random subset
    Xtr = z["Xtr"].astype(np.float32)[sub] / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr = torch.tensor(z["ytr"].astype(np.int64)[sub], device=dev)
    yte = torch.tensor(z["yte"].astype(np.int64), device=dev)
    N = len(ytr)
    env0 = Env(torch, dev, Xtr, Xte, max_cached=len(SCALES))
    for ps in SCALES:
        env0.maps(ps)
    basis = {ps: re2._SVD_CACHE[(_fp(Xtr), ps)] for ps in SCALES}
    n_fit = int(N * 0.8); yv = ytr[n_fit:]
    Y = -torch.ones((N, 10), device=dev); Y[torch.arange(N), ytr] = 1.
    dev_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"

    def augment(Xnp, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        t = torch.tensor(Xnp, device=dev).permute(0, 3, 1, 2); n = len(t)
        b = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=dev) - .5) * .5
        col = 1 + (torch.rand(n, 3, 1, 1, generator=g, device=dev) - .5) * .3
        m = t.mean((2, 3), keepdim=True)
        con = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=dev) - .5) * .5
        t = (((t - m) * con + m) * b * col).clamp(0, 1)
        return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def env_for(Xa):
        e = Env(torch, dev, Xa, Xa[:100], max_cached=len(SCALES))
        for ps in SCALES:
            re2._SVD_CACHE[(_fp(Xa), ps)] = basis[ps]
        return e

    def l0_maps(env, test=False):
        """layer-0 feature maps: multiscale patch-PCA, resampled to G x G."""
        bank = []
        for ps in SCALES:
            Mtr, Mte, H, W = env.maps(ps)
            M = (Mte if test else Mtr).float().view(-1, re2.C_PER_SCALE, H, W)
            bank.append(Fn.adaptive_avg_pool2d(M, (G, G)))
        return torch.cat(bank, 1).half()                 # (N, C0, G, G) fp16

    # conv filter = grid genome applied across the map (want_grid=True -> a map)
    def conv_maps(gridT, genomes):
        return torch.stack([rk.feature_grid_g(torch, tp, gridT, g, want_grid=True)
                            for g in genomes], 1).half()  # (N, K, G, G) fp16

    def pooled(mapsT):
        return Fn.adaptive_avg_pool2d(mapsT, (args.pool, args.pool)
                                      ).flatten(1)         # (N, K*pool*pool)

    def head(F, Fte, tag):
        mu = F[:n_fit].mean(0); sd = F[:n_fit].std(0) + 1e-6   # fit-only stats
        Z, Zt = (F - mu) / sd, (Fte - mu) / sd
        A = torch.hstack([Z[:n_fit], torch.ones(n_fit, 1, device=dev)])
        Gm = (A.T @ A).double()
        Av = torch.hstack([Z[n_fit:], torch.ones(N - n_fit, 1, device=dev)])
        bl, bv = 30., -1
        for lam in (1., 3., 10., 30., 100., 300.):
            W = torch.linalg.solve(Gm + lam * torch.eye(A.shape[1], device=dev,
                                   dtype=torch.float64),
                                   (A.T @ Y[:n_fit]).double()).float()
            a = float(((Av @ W).argmax(1) == yv).float().mean())   # same-solve val
            if a > bv:
                bl, bv = lam, a
        A2 = torch.hstack([Z, torch.ones(N, 1, device=dev)])
        W = torch.linalg.solve((A2.T @ A2).double() + bl * torch.eye(A2.shape[1],
                               device=dev, dtype=torch.float64),
                               (A2.T @ Y).double()).float()
        B = torch.hstack([Zt, torch.ones(len(yte), 1, device=dev)])
        acc = float(((B @ W).argmax(1) == yte).float().mean())
        log(f"[conv] HEAD [{tag}]: val {bv:.4f} TEST {acc:.4f}")
        return acc

    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:4000], device=dev)
    rng = np.random.default_rng(62)

    # layer-0 maps on clean + two augmented views (for the label-free question)
    L0 = l0_maps(env0); L0te = l0_maps(env0, test=True)
    eA, eB = env_for(augment(Xtr, 1)), env_for(augment(Xtr, 2))
    L0a, L0b = l0_maps(eA), l0_maps(eB)
    C0 = L0.shape[1]
    log(f"[conv] layer-0 maps {tuple(L0.shape)} ({round(time.time()-t0)}s, {dev_name})")

    def make_layer(prev, prevA, prevB, cap, rounds, tag):
        Cin = prev.shape[1]

        def fit(gs):                                     # contrastive: pooled map stable across views
            CA = pooled(conv_maps(prevA, gs))
            CB = pooled(conv_maps(prevB, gs))
            # per-genome invariance x info over the pooled multi-cell response
            a = (CA - CA.mean(0)) / (CA.std(0) + 1e-6)
            b = (CB - CB.mean(0)) / (CB.std(0) + 1e-6)
            K = len(gs); pc = args.pool * args.pool
            inv = (a * b).view(len(a), K, pc).mean((0, 2)).clamp(min=0)
            info = CA.view(len(CA), K, pc).std(0).mean(1)
            info = info / (info.mean() + 1e-6)
            q = (inv * info).cpu().numpy()
            q[~np.isfinite(q)] = -1
            return q

        def sig(gs):
            return pooled(conv_maps(prev, gs))[probe][:, ::args.pool * args.pool]
        genomes = energy_evolve(rng, args.pop, args.gens, rounds, cap, 0.02,
                                lambda r: rk.new_grid_genome(r, Cin),
                                lambda r, g, s: rk.mutate_grid_g(r, g, s, Cin),
                                fit, sig, tag)
        return genomes

    # ---- LAYER 1 : conv filters over layer-0 maps -------------------------
    L1_g = make_layer(L0, L0a, L0b, args.l1_cap, args.l1_rounds, "L1")
    L1 = conv_maps(L0, L1_g); L1te = conv_maps(L0te, L1_g)
    L1a, L1b = conv_maps(L0a, L1_g), conv_maps(L0b, L1_g)
    log(f"[conv] LAYER 1: {len(L1_g)} filters -> maps {tuple(L1.shape)} "
        f"({round(time.time()-t0)}s)")

    # ---- LAYER 2 : conv filters over layer-1 MAPS (the hierarchy test) -----
    L2_g = make_layer(L1, L1a, L1b, args.l2_cap, args.l2_rounds, "L2")
    L2 = conv_maps(L1, L2_g); L2te = conv_maps(L1te, L2_g)
    log(f"[conv] LAYER 2: {len(L2_g)} filters -> maps {tuple(L2.shape)} "
        f"({round(time.time()-t0)}s)")

    # ---- pool + head : does layer 2 add over layer 1? ---------------------
    P1, P1te = pooled(L1), pooled(L1te)
    P2, P2te = pooled(L2), pooled(L2te)
    a1 = head(P1, P1te, f"LAYER 1 only ({len(L1_g)} filters)")
    a2 = head(torch.cat([P1, P2], 1), torch.cat([P1te, P2te], 1),
              f"LAYER 1 | LAYER 2 ({len(L1_g)}+{len(L2_g)})")
    log(f"[conv] HIERARCHY: L1 {a1:.4f} -> L1|L2 {a2:.4f}  (layer-2 spatial "
        f"gain {a2 - a1:+.4f}; pooled-handoff stacks earned 0)")
    log(f"[conv] COMPUTE: {round(time.time()-t0)}s on {dev_name}, peak "
        f"{torch.cuda.max_memory_allocated()/1e9 if dev=='cuda' else 0:.1f}GB")
    json.dump({"module": "convrs", "l1": len(L1_g), "l2": len(L2_g),
               "l1_test": round(a1, 4), "l12_test": round(a2, 4),
               "l2_gain": round(a2 - a1, 4)},
              open(os.path.join(RD, "replicate_convrs.json"), "w"))


if __name__ == "__main__":
    main()
