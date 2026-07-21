"""replicate_r1space.py — module 56: evolve R1 as a PROPER radial space.

Everything before this (m38, m49, m52-54) ran a hand-rolled greedy elite GA
with the energy economy switched OFF and top-N selection, then called it R1.
That is not a radial space. This runs the REAL machinery: radial_stack's
_evolve_space, which is what run_stacked uses for space 1 --

  - R0 goes IN the scoring base, so R1 is scored on gain OVER R0 (the residual)
  - energy economy: starved genomes die and are replaced (homeostasis on)
  - tournament selection among the living, adaptive per-genome mutation scale
  - freeze every decorrelated contributor, emergent cap (space fills until
    rounds stop earning)

over the strong frozen 0.77 union R0 (its pre-pool 4x4 field + raw skip), the
proper way. Then test the R0 | R1 stack.

    python3 replicate/replicate_r1space.py --pop 64 --gens 12 --rounds 120
"""
import argparse
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
from radial_evo2 import Env, SCALES
import radial_stack as rk
from replicate_r1build import union_genomes

RD = os.path.join(_HERE, "radial_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--cap", type=float, default=0.0002)
    ap.add_argument("--patience", type=int, default=3)
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
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"].astype(np.int64), z["yte"].astype(np.int64)
    N = len(ytr)
    env = Env(torch, dev, Xtr, Xte, max_cached=6)
    gs = union_genomes()
    C_prev = len(gs)
    print(f"[r1space] union R0 = {C_prev} genomes; evolving R1 as a REAL radial "
          f"space (energy on) over its {G}x{G} field ({dev})", flush=True)

    # R0 scalar outputs = the frozen base R1 is scored ON TOP OF
    R0tr = torch.stack([rk.feature_r0(torch, tp, env, g) for g in gs], 1)
    R0te = torch.stack([rk.feature_r0(torch, tp, env, g, test=True)
                        for g in gs], 1)
    # R0 pre-pool field (what R1 genomes read) + raw skip bank
    gtr = torch.empty((N, C_prev, G, G), dtype=torch.float16, device=dev)
    for j, g in enumerate(gs):
        gtr[:, j] = rk.feature_r0(torch, tp, env, g, want_grid=True).half()
    bt = []
    for ps in SCALES:
        Mtr, _, H, W = env.maps(ps)
        bt.append(Fn.adaptive_avg_pool2d(Mtr.float().view(len(Mtr), -1, H, W),
                                         (G, G)))
    raw_tr = torch.cat(bt, 1).half()
    del bt
    torch.cuda.empty_cache()
    print(f"[r1space] R0 field {tuple(gtr.shape)} + raw skip {tuple(raw_tr.shape)} "
          f"({round(time.time()-t0)}s)", flush=True)

    n_fit = int(N * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((N, 10), device=dev)
    Yfull[torch.arange(N), torch.tensor(ytr, device=dev)] = 1.0

    def test_of(Ftr, Fte):
        bl, bv = 3.0, -1.0
        for lam in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
            _, a = _ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf, yv, lam=lam)
            if a > bv:
                bl, bv = lam, a
        n, d = Ftr.shape
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        A = torch.hstack([(Ftr - mu) / sd, torch.ones(n, 1, device=dev)])
        Gm = (A.T @ A).double() + bl * torch.eye(d + 1, device=dev,
                                                 dtype=torch.float64)
        W = torch.linalg.solve(Gm, (A.T @ Yfull).double()).float()
        B = torch.hstack([(Fte - mu) / sd, torch.ones(len(Fte), 1, device=dev)])
        return float(((B @ W).argmax(1) == yte_t).float().mean()), bv

    r0_test, r0_val = test_of(R0tr, R0te)
    print(f"[r1space] R0 alone: val {r0_val:.4f} TEST {r0_test:.4f}", flush=True)

    # ---- THE REAL SPACE EVOLUTION (energy economy, tournament, emergent cap)
    rng = np.random.default_rng(56)
    new_fn = lambda r: rk.new_grid_genome(r, C_prev)
    mut_fn = lambda r, g, sc: rk.mutate_grid_g(r, g, sc, C_prev)
    feat_tr = lambda g: rk.feature_grid_g(torch, tp, gtr, g, rawT=raw_tr)

    def log(msg, verbose=True):
        print(msg, flush=True)

    frozen, fcols = rk._evolve_space(
        torch, rng, args.pop, args.gens, args.rounds, n_fit, Yf, yv,
        base_prev=R0tr, new_fn=new_fn, mut_fn=mut_fn, feat_tr=feat_tr,
        log=log, verbose=True, cap_override=args.cap, patience=args.patience)

    # ---- test the R0 | R1 stack ------------------------------------------
    if frozen:
        R1tr = torch.stack(fcols, 1)
        R1te = torch.stack([rk.feature_grid_g(torch, tp,
                            torch.empty((len(yte), C_prev, G, G), dtype=torch.float16,
                                        device=dev), g) for g in frozen], 1) \
            if False else None
        # build test field once, then R1 test cols
        gte = torch.empty((len(yte), C_prev, G, G), dtype=torch.float16, device=dev)
        for j, g in enumerate(gs):
            gte[:, j] = rk.feature_r0(torch, tp, env, g, test=True,
                                      want_grid=True).half()
        bt = []
        for ps in SCALES:
            _, Mte, H, W = env.maps(ps)
            bt.append(Fn.adaptive_avg_pool2d(Mte.float().view(len(Mte), -1, H, W),
                                             (G, G)))
        raw_te = torch.cat(bt, 1).half()
        R1te = torch.stack([rk.feature_grid_g(torch, tp, gte, g, rawT=raw_te)
                            for g in frozen], 1)
        j_test, j_val = test_of(torch.cat([R0tr, R1tr], 1),
                                torch.cat([R0te, R1te], 1))
        print(f"[r1space] R0 | R1 STACK ({len(frozen)} R1 genomes): "
              f"val {j_val:.4f} TEST {j_test:.4f}  (R0 {r0_test:.4f}, "
              f"delta test {j_test - r0_test:+.4f}, delta val {j_val - r0_val:+.4f})",
              flush=True)
    print(f"[r1space] done ({round(time.time()-t0)}s)", flush=True)


if __name__ == "__main__":
    main()
