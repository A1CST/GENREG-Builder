"""replicate_r1field.py — module 52: R1 over R0's PRE-POOL FIELD. THE stack test.

Per the project direction: R1 must read R0's spatial activation FIELD before
collapse (the natural output of one radial space = input to the next), NOT
R0's pooled outputs (m50/m51 proved those are nonlinearly exhausted ~0.78),
and NOT bypass R0 (that just rebuilds R0). The failed bank shows the COLLAPSED
R0 representation is exhausted; it does NOT show R0's computation is.

The clean experiment (all requirements from the direction note):
  - freeze R0; expose its pre-pool maps (higher-res field, G=8, vs m38's 4x4)
  - R1 composes across channels, positions, and LOCAL NEIGHBORHOODS
    (grid grammar with shift genes), src="prev" only (pure R0->R1, no raw skip)
  - R1 preserves its output as a spatial field until ITS OWN pooling stage
  - select R1 features for DIVERSITY/STABILITY, not individual label gain
    (the m38/m49 label-significance gate strangled the diffuse signal)
  - judge only the complete R0 | R1 | head stack on VALIDATION
  - require >= +1 validation point over the R0 pooled baseline to count

Decisive question: does a strong readout on R0's pre-pool field beat the
pooled-output baseline? If yes, the RS stack earns and R0's field (not its
collapse) is the substrate for depth.

    python3 replicate/replicate_r1field.py --grid 8 --bank 2000
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--bank", type=int, default=2000)
    ap.add_argument("--corr", type=float, default=0.8, help="diversity: reject "
                    "a feature if |corr| with the kept bank exceeds this")
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
    log(f"[field] R0 {C_prev} genomes, pre-pool field G={G} "
        f"({torch.cuda.get_device_name(0)})")

    # R0 pooled baseline (the number to beat by +1pt on VAL)
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
    log(f"[field] R0 POOLED baseline: val {r0_val:.4f} TEST {r0_test:.4f} "
        f"(bar to beat: val {r0_val + 0.01:.4f})")

    # pre-pool field bank (N, C_prev, G, G) fp16 — filled incrementally so we
    # never materialize the full fp32 stack (that OOMs at ~46GB)
    gtr = torch.empty((N, C_prev, G, G), dtype=torch.float16, device=dev)
    gte = torch.empty((len(yte), C_prev, G, G), dtype=torch.float16, device=dev)
    for j, g in enumerate(gs):
        gtr[:, j] = rk.feature_r0(torch, tp, env, g, want_grid=True).half()
        gte[:, j] = rk.feature_r0(torch, tp, env, g, test=True,
                                  want_grid=True).half()
    log(f"[field] pre-pool field tr {tuple(gtr.shape)} "
        f"({gtr.element_size()*gtr.nelement()/1e9:.1f}GB, "
        f"{round(time.time()-t0)}s)")

    # ---- build a DIVERSE bank of R1 field-composition features -------------
    # each R1 genome composes R0 channels across positions/neighborhoods
    # (shift genes), keeps a field, then pools at ITS OWN stage. src forced
    # "prev": pure R0 -> R1, no raw skip. Admission = diversity only (no labels).
    rng = np.random.default_rng(52)
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:4000],
                         device=dev)
    # STABILITY: a feature's discriminative direction must reproduce across two
    # fit-set halves (cosine of its per-class correlation vectors). This keeps
    # generalizable features and rejects fit-set noise WITHOUT gating on
    # individual label gain. Ycen halves precomputed once.
    hA = torch.arange(0, n_fit // 2, device=dev)
    hB = torch.arange(n_fit // 2, n_fit, device=dev)
    Ya = Y[hA] - Y[hA].mean(0)
    Yb = Y[hB] - Y[hB].mean(0)
    STAB = 0.35
    kept_tr, kept_te, kept_sig = [], [], []
    tried = 0
    while sum(c.shape[1] for c in kept_tr) < args.bank and tried < args.bank * 40:
        blk = 256
        genomes = []
        for _ in range(blk):
            g = rk.new_grid_genome(rng, C_prev)
            for t in g.get("terms", []):                 # res-arch has no terms
                t["src"] = "prev"                        # pure R0 field
            genomes.append(g)                            # rawT unset -> no skip
        cc = torch.stack([rk.feature_grid_g(torch, tp, gtr, g)
                          for g in genomes], 1)           # (N, blk)
        tried += blk
        m2, s2 = cc.mean(0), cc.std(0) + 1e-6
        cc = (cc - m2) / s2
        sig = cc[probe]
        # stability: cosine between per-class correlation on the two fit halves
        csA = cc[hA].T @ Ya                              # (blk,10)
        csB = cc[hB].T @ Yb
        csA = csA / (csA.norm(dim=1, keepdim=True) + 1e-8)
        csB = csB / (csB.norm(dim=1, keepdim=True) + 1e-8)
        stab = (csA * csB).sum(1)                        # (blk,)
        good = (torch.isfinite(sig).all(0) & (s2 > 1e-4) & (stab > STAB))
        sign = sig / (sig.norm(dim=0, keepdim=True) + 1e-8)
        idxs = torch.where(good)[0].tolist()
        # greedy diversity admission within block + vs kept
        keptS = torch.cat(kept_sig, 1) if kept_sig else None
        admitted = []
        for i in idxs:
            s = sign[:, i:i + 1]
            if keptS is not None and float((s.T @ keptS).abs().max()) > args.corr:
                continue
            if admitted:
                AS = sign[:, admitted]
                if float((s.T @ AS).abs().max()) > args.corr:
                    continue
            admitted.append(i)
        if admitted:
            sel = torch.tensor(admitted, device=dev)
            kept_tr.append(cc[:, sel])
            cte = torch.stack([rk.feature_grid_g(torch, tp, gte, genomes[i])
                               for i in admitted], 1)
            cte = (cte - m2[sel]) / s2[sel]
            kept_te.append(cte)
            kept_sig.append(sign[:, sel])
        nb = sum(c.shape[1] for c in kept_tr)
        if tried % 1024 == 0 or nb >= args.bank:
            log(f"[field] bank {nb} (tried {tried}) ({round(time.time()-t0)}s)")

    R1tr_all = torch.cat(kept_tr, 1)
    R1te_all = torch.cat(kept_te, 1)
    log(f"[field] R1 field pool {R1tr_all.shape[1]} stable+diverse features; "
        f"sweeping bank size for the val peak")

    # sweep bank size: find where the field's signal peaks before ridge overfit
    best = (-1, 0, 0)
    for k in (100, 200, 300, 500, 800, 1200, 1800, 2600, R1tr_all.shape[1]):
        k = min(k, R1tr_all.shape[1])
        jt, jv, jl = test_of(torch.cat([R0tr, R1tr_all[:, :k]], 1),
                             torch.cat([R0te, R1te_all[:, :k]], 1))
        dv = jv - r0_val
        mark = "  <== +1pt!" if dv >= 0.01 else ""
        log(f"[field]   bank {k:5d}: stack val {jv:.4f} TEST {jt:.4f} "
            f"(delta val {dv:+.4f}){mark}")
        if jv > best[0]:
            best = (jv, k, jt)
    # also try PCA-compressing the whole pool (extract signal, dodge overfit)
    Zp = (R1tr_all - R1tr_all.mean(0)) / (R1tr_all.std(0) + 1e-6)
    U, S, V = torch.linalg.svd(Zp[:n_fit] - Zp[:n_fit].mean(0),
                               full_matrices=False)
    for kp in (100, 200, 400):
        Vp = V[:kp].T
        Ptr = (R1tr_all - R1tr_all.mean(0)) @ Vp
        Pte = (R1te_all - R1tr_all.mean(0)) @ Vp
        jt, jv, jl = test_of(torch.cat([R0tr, Ptr], 1),
                             torch.cat([R0te, Pte], 1))
        dv = jv - r0_val
        mark = "  <== +1pt!" if dv >= 0.01 else ""
        log(f"[field]   PCA {kp:4d}: stack val {jv:.4f} TEST {jt:.4f} "
            f"(delta val {dv:+.4f}){mark}")
        if jv > best[0]:
            best = (jv, -kp, jt)
    dvb = best[0] - r0_val
    log(f"[field] BEST stack: val {best[0]:.4f} TEST {best[2]:.4f} at "
        f"bank/pca {best[1]} (R0 val {r0_val:.4f}, delta val {dvb:+.4f}) -> "
        f"{'SIGNAL +1pt' if dvb >= 0.01 else 'below +1pt bar'}")
    json.dump({"module": "r1field", "grid": G, "pool": R1tr_all.shape[1],
               "r0_val": round(r0_val, 4), "best_val": round(best[0], 4),
               "best_test": round(best[2], 4), "delta_val": round(dvb, 4)},
              open(os.path.join(RD, "replicate_r1field.json"), "w"))


if __name__ == "__main__":
    main()
