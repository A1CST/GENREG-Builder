"""replicate_r1fieldevo.py — module 53: EVOLVE R1 over R0's pre-pool field.

m52 confirmed the direction's prediction: R0's pre-pool spatial field carries
information its pooled outputs don't (random field compositions give a real,
consistent +0.25pt val over the pooled baseline, where the pooled-output bank
was flat-to-negative). But random compositions are individually weak and cap
at +0.25. This evolves the field compositions — the actual RS stack.

Requirements from the project direction, kept intact:
  - freeze R0; R1 reads the pre-pool field (G=8), src="prev" only (no raw skip)
  - R1 composes across channels, positions, LOCAL NEIGHBORHOODS (shift genes),
    keeps a field, pools at its own stage
  - SELECT for diversity + STABILITY, not ensemble label gain: a genome's
    fitness is the cross-split reproducibility of its per-class correlation
    (stability) times its correlation magnitude — i.e. find field compositions
    that RELIABLY (not fit-set-luckily) carry class structure. Admission is
    diversity + a stability floor. No significance-gate over a base.
  - judge only the full R0 | R1 stack on VALIDATION; +1pt bar.

    python3 replicate/replicate_r1fieldevo.py --rounds 60 --grid 8
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
from radial_evo2 import Env
import radial_stack as rk
from replicate_r1build import union_genomes

RD = os.path.join(_HERE, "radial_data")
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def force_prev(g):
    for t in g.get("terms", []):
        t["src"] = "prev"
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--pop", type=int, default=160)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--cap", type=int, default=1200)
    args = ap.parse_args()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    tp = _tprims(torch)
    rk.GRID = args.grid
    G = rk.GRID
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr = torch.tensor(z["ytr"].astype(np.int64), device=dev)
    yte = torch.tensor(z["yte"].astype(np.int64), device=dev)
    N = len(z["ytr"])
    env = Env(torch, dev, Xtr, Xte, max_cached=6)
    gs = union_genomes()
    C_prev = len(gs)
    n_fit = int(N * 0.8)
    yv = ytr[n_fit:]
    Y = -torch.ones((N, 10), device=dev)
    Y[torch.arange(N), ytr] = 1.0
    log(f"[fevo] R0 {C_prev} genomes, field G={G} "
        f"({torch.cuda.get_device_name(0)})")

    R0tr = torch.stack([rk.feature_r0(torch, tp, env, g) for g in gs], 1)
    R0te = torch.stack([rk.feature_r0(torch, tp, env, g, test=True)
                        for g in gs], 1)

    def test_of(Ftr, Fte, lams=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0)):
        n, d = Ftr.shape
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        A = torch.hstack([((Ftr - mu) / sd)[:n_fit],
                          torch.ones(n_fit, 1, device=dev)])
        Gm = (A.T @ A).double()
        Av = torch.hstack([((Ftr - mu) / sd)[n_fit:],
                           torch.ones(n - n_fit, 1, device=dev)])
        best = (-1, None)
        for lam in lams:
            W = torch.linalg.solve(Gm + lam * torch.eye(d + 1, device=dev,
                                   dtype=torch.float64),
                                   (A.T @ Y[:n_fit]).double()).float()
            va = float(((Av @ W).argmax(1) == yv).float().mean())
            if va > best[0]:
                best = (va, lam)
        lam = best[1]
        Afull = torch.hstack([(Ftr - mu) / sd, torch.ones(n, 1, device=dev)])
        W = torch.linalg.solve((Afull.T @ Afull).double() + lam * torch.eye(
            d + 1, device=dev, dtype=torch.float64),
            (Afull.T @ Y).double()).float()
        At = torch.hstack([(Fte - mu) / sd, torch.ones(len(Fte), 1, device=dev)])
        ta = float(((At @ W).argmax(1) == yte).float().mean())
        return ta, best[0], lam

    r0_test, r0_val, _ = test_of(R0tr, R0te)
    log(f"[fevo] R0 POOLED baseline: val {r0_val:.4f} TEST {r0_test:.4f} "
        f"(+1pt bar: val {r0_val + 0.01:.4f})")

    gtr = torch.empty((N, C_prev, G, G), dtype=torch.float16, device=dev)
    gte = torch.empty((len(yte), C_prev, G, G), dtype=torch.float16, device=dev)
    for j, g in enumerate(gs):
        gtr[:, j] = rk.feature_r0(torch, tp, env, g, want_grid=True).half()
        gte[:, j] = rk.feature_r0(torch, tp, env, g, test=True,
                                  want_grid=True).half()
    log(f"[fevo] field {tuple(gtr.shape)} "
        f"({gtr.element_size()*gtr.nelement()/1e9:.1f}GB, {round(time.time()-t0)}s)")

    # stability fitness: cross-split reproducibility of per-class correlation
    hA = torch.arange(0, n_fit // 2, device=dev)
    hB = torch.arange(n_fit // 2, n_fit, device=dev)
    Ya = Y[hA] - Y[hA].mean(0)
    Yb = Y[hB] - Y[hB].mean(0)

    def fitness(cc):                                     # cc (N, K) raw
        c = (cc - cc.mean(0)) / (cc.std(0) + 1e-6)
        csA = c[hA].T @ Ya
        csB = c[hB].T @ Yb
        nA = csA.norm(dim=1) + 1e-8
        nB = csB.norm(dim=1) + 1e-8
        stab = (csA * csB).sum(1) / (nA * nB)            # cross-split cosine
        mag = 0.5 * (nA + nB) / (len(hA) ** 0.5)         # corr magnitude
        f = stab.clamp_min(0) * mag                      # stable AND informative
        f[~torch.isfinite(f)] = -1
        return f, stab

    rng = np.random.default_rng(53)
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:4000],
                         device=dev)
    bank_tr, bank_te, bank_sig = [], [], []
    best_stack = (r0_val, r0_test, 0)
    for rnd in range(args.rounds):
        if sum(c.shape[1] for c in bank_tr) >= args.cap:
            break
        pop = [force_prev(rk.new_grid_genome(rng, C_prev))
               for _ in range(args.pop)]
        for gen in range(args.gens):
            cc = torch.stack([rk.feature_grid_g(torch, tp, gtr, g)
                              for g in pop], 1)
            f, _ = fitness(cc)
            order = torch.argsort(-f).cpu().numpy()
            elite = order[:max(2, args.pop // 4)]
            npop = [pop[i] for i in elite]
            while len(npop) < args.pop:
                src = elite[rng.integers(len(elite))]
                npop.append(force_prev(rk.mutate_grid_g(
                    rng, pop[src], float(rng.uniform(0.25, 0.5)), C_prev)))
            pop = npop
        cc = torch.stack([rk.feature_grid_g(torch, tp, gtr, g) for g in pop], 1)
        f, stab = fitness(cc)
        order = torch.argsort(-f).cpu().numpy()
        m2, s2 = cc.mean(0), cc.std(0) + 1e-6
        ccz = (cc - m2) / s2
        sig = ccz[probe]
        sign = sig / (sig.norm(dim=0, keepdim=True) + 1e-8)
        keptS = torch.cat(bank_sig, 1) if bank_sig else None
        adm = []
        for i in order:
            if float(f[i]) <= 0 or float(stab[i]) < 0.35 or len(adm) >= 12:
                break
            s = sign[:, i:i + 1]
            if keptS is not None and float((s.T @ keptS).abs().max()) > 0.8:
                continue
            if adm and float((s.T @ sign[:, adm]).abs().max()) > 0.8:
                continue
            adm.append(int(i))
        if adm:
            sel = torch.tensor(adm, device=dev)
            bank_tr.append(ccz[:, sel])
            cte = torch.stack([rk.feature_grid_g(torch, tp, gte, pop[i])
                               for i in adm], 1)
            bank_te.append((cte - m2[sel]) / s2[sel])
            bank_sig.append(sign[:, sel])
        nb = sum(c.shape[1] for c in bank_tr)
        if rnd % 5 == 0 or nb >= args.cap:
            Btr = torch.cat(bank_tr, 1)
            Bte = torch.cat(bank_te, 1)
            jt, jv, jl = test_of(torch.cat([R0tr, Btr], 1),
                                 torch.cat([R0te, Bte], 1))
            if jv > best_stack[0]:
                best_stack = (jv, jt, nb)
            dv = jv - r0_val
            mark = "  <== +1pt!" if dv >= 0.01 else ""
            log(f"[fevo] round {rnd}: +{len(adm)} -> {nb} feats; stack val "
                f"{jv:.4f} TEST {jt:.4f} (delta val {dv:+.4f}){mark} "
                f"({round(time.time()-t0)}s)")
        else:
            log(f"[fevo] round {rnd}: +{len(adm)} -> {nb} "
                f"({round(time.time()-t0)}s)")

    dv = best_stack[0] - r0_val
    log(f"[fevo] BEST stack: val {best_stack[0]:.4f} TEST {best_stack[1]:.4f} "
        f"@ {best_stack[2]} feats (R0 val {r0_val:.4f}, delta {dv:+.4f}) -> "
        f"{'SIGNAL +1pt' if dv >= 0.01 else 'below +1pt bar'}")
    json.dump({"module": "r1fieldevo", "best_val": round(best_stack[0], 4),
               "best_test": round(best_stack[1], 4), "r0_val": round(r0_val, 4),
               "delta_val": round(dv, 4)},
              open(os.path.join(RD, "replicate_r1fieldevo.json"), "w"))


if __name__ == "__main__":
    main()
