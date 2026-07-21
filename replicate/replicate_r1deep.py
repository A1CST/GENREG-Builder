"""replicate_r1deep.py — module 54: RICHER/DEEPER R1 grammar over R0's field.

m52/53 validated the direction: R0's pre-pool field earns where its collapse
doesn't (+0.49pt val evolved). But finer resolution was refuted (G=10/12 <
G=8) — the field's advantage is NOT fine detail, it is spatial STRUCTURE /
CO-OCCURRENCE. Yet my R1 terms only read a SHIFTED POINT of each R0 map, which
cannot express local structure. This gives R1 genuine NEIGHBORHOOD OPERATORS:
each term applies a local spatial operator over an R0 activation map before
composing — gradient (x/y/mag), Laplacian, center-surround, local-max,
local-std — reading the SHAPE of an R0 feature's activation, not just its
value at a point. Composition depth raised (order up to 4). This is "compose
across local neighborhoods," taken further.

Everything else the direction fixed is kept: freeze R0, pre-pool field G=8,
pure R0->R1, select for diversity + cross-split STABILITY (not ensemble label
gain), judge the full R0|R1 stack on VALIDATION, +1pt bar.

    python3 replicate/replicate_r1deep.py --rounds 80 --grid 8
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
NBR = ["id", "gradx", "grady", "gradmag", "lap", "csurr", "lmax", "lstd"]
PRIMS = ["id", "abs", "relu", "tanh", "gauss", "sq", "soft", "sin"]
OPS = ["mult", "min", "absdiff"]
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def _nbr(torch, v, kind):
    """local spatial operator over a field map v (N,G,G). roll = circular
    neighborhood (fine at G=8)."""
    if kind == "id":
        return v
    sx = torch.roll(v, 1, 2); sX = torch.roll(v, -1, 2)
    sy = torch.roll(v, 1, 1); sY = torch.roll(v, -1, 1)
    if kind == "gradx":
        return v - sx
    if kind == "grady":
        return v - sy
    if kind == "gradmag":
        return torch.sqrt((v - sx) ** 2 + (v - sy) ** 2 + 1e-6)
    if kind == "lap":
        return 4 * v - sx - sX - sy - sY
    if kind == "csurr":
        return v - 0.25 * (sx + sX + sy + sY)
    if kind == "lmax":
        import torch.nn.functional as Fn
        return Fn.max_pool2d(v.unsqueeze(1), 3, 1, 1).squeeze(1)
    if kind == "lstd":
        import torch.nn.functional as Fn
        m = Fn.avg_pool2d(v.unsqueeze(1), 3, 1, 1)
        m2 = Fn.avg_pool2d((v * v).unsqueeze(1), 3, 1, 1)
        return torch.sqrt((m2 - m * m).clamp_min(0) + 1e-6).squeeze(1)
    return v


def new_term(rng, C):
    return {"c": int(rng.integers(C)),
            "nbr": int(rng.integers(len(NBR))),
            "sh": [int(rng.integers(-2, 3)), int(rng.integers(-2, 3))],
            "prog": [(int(rng.integers(len(PRIMS))),
                      float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1, 1)))
                     for _ in range(1 if rng.random() < 0.6 else 2)]}


def new_genome(rng, C):
    order = int(rng.choice([2, 3, 4], p=[0.35, 0.4, 0.25]))
    return {"terms": [new_term(rng, C) for _ in range(order)],
            "op": int(rng.integers(len(OPS))),
            "cx": float(rng.uniform(0.1, 0.9)), "cy": float(rng.uniform(0.1, 0.9)),
            "lsig": float(rng.uniform(np.log(0.15), np.log(1.5))),
            "stat": int(rng.integers(5)), "beta": 0.0}


def mutate(rng, g, sc, C):
    c = json.loads(json.dumps(g))
    for t in c["terms"]:
        if rng.random() < 0.12:
            t["c"] = int(rng.integers(C))
        if rng.random() < 0.15:
            t["nbr"] = int(rng.integers(len(NBR)))
        if rng.random() < 0.2:
            t["sh"] = [int(np.clip(t["sh"][0] + rng.integers(-1, 2), -3, 3)),
                       int(np.clip(t["sh"][1] + rng.integers(-1, 2), -3, 3))]
        prog = [list(s) for s in t["prog"]]
        for s in prog:
            if rng.random() < 0.1:
                s[0] = int(rng.integers(len(PRIMS)))
            s[1] = float(np.clip(s[1] + rng.normal(0, sc), 0.1, 4.0))
            s[2] = float(np.clip(s[2] + rng.normal(0, sc), -2.0, 2.0))
        if rng.random() < 0.1:
            if len(prog) == 1:
                prog.append([int(rng.integers(len(PRIMS))),
                             float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1, 1))])
            else:
                prog.pop(int(rng.integers(len(prog))))
        t["prog"] = [tuple(s) for s in prog]
    if rng.random() < 0.1:                            # depth evolves (2..4)
        if len(c["terms"]) < 4 and rng.random() < 0.5:
            c["terms"].append(new_term(rng, C))
        elif len(c["terms"]) > 2:
            c["terms"].pop(int(rng.integers(len(c["terms"]))))
    if rng.random() < 0.08:
        c["op"] = int(rng.integers(len(OPS)))
    if rng.random() < 0.08:
        c["stat"] = int(rng.integers(5))
    c["cx"] = float(np.clip(g["cx"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
    c["cy"] = float(np.clip(g["cy"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
    c["lsig"] = float(np.clip(g["lsig"] + rng.normal(0, sc * 0.5),
                              np.log(0.05), np.log(3.0)))
    return c


def feature(torch, tp, gridT, g):
    z = None
    for t in g["terms"]:
        v = gridT[:, t["c"] % gridT.shape[1]].float()
        v = _nbr(torch, v, NBR[t["nbr"]])
        sh = t.get("sh") or [0, 0]
        v = rk._shift2d(torch, v, int(sh[0]), int(sh[1]))
        for prim, a, b in t["prog"]:
            v = tp[PRIMS[prim]](a * v + b)
        if z is None:
            z = v
        else:
            op = OPS[g["op"]]
            z = z * v if op == "mult" else (torch.minimum(z, v) if op == "min"
                                            else torch.abs(z - v))
    return rk._window_pool(torch, z, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--pop", type=int, default=160)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--cap", type=int, default=1500)
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
    log(f"[deep] R0 {C_prev}, field G={G}, {len(NBR)} nbr ops, order<=4 "
        f"({torch.cuda.get_device_name(0)})")

    R0tr = torch.stack([rk.feature_r0(torch, tp, env, g) for g in gs], 1)
    R0te = torch.stack([rk.feature_r0(torch, tp, env, g, test=True)
                        for g in gs], 1)

    def test_of(Ftr, Fte, lams=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0)):
        n, d = Ftr.shape
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        A = torch.hstack([((Ftr - mu) / sd)[:n_fit], torch.ones(n_fit, 1, device=dev)])
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
    log(f"[deep] R0 POOLED baseline: val {r0_val:.4f} TEST {r0_test:.4f} "
        f"(+1pt bar: val {r0_val + 0.01:.4f})")

    gtr = torch.empty((N, C_prev, G, G), dtype=torch.float16, device=dev)
    gte = torch.empty((len(yte), C_prev, G, G), dtype=torch.float16, device=dev)
    for j, g in enumerate(gs):
        gtr[:, j] = rk.feature_r0(torch, tp, env, g, want_grid=True).half()
        gte[:, j] = rk.feature_r0(torch, tp, env, g, test=True,
                                  want_grid=True).half()
    log(f"[deep] field {tuple(gtr.shape)} ({round(time.time()-t0)}s)")

    hA = torch.arange(0, n_fit // 2, device=dev)
    hB = torch.arange(n_fit // 2, n_fit, device=dev)
    Ya, Yb = Y[hA] - Y[hA].mean(0), Y[hB] - Y[hB].mean(0)

    def fitness(cc):
        c = (cc - cc.mean(0)) / (cc.std(0) + 1e-6)
        csA, csB = c[hA].T @ Ya, c[hB].T @ Yb
        nA, nB = csA.norm(dim=1) + 1e-8, csB.norm(dim=1) + 1e-8
        stab = (csA * csB).sum(1) / (nA * nB)
        f = stab.clamp_min(0) * (0.5 * (nA + nB) / (len(hA) ** 0.5))
        f[~torch.isfinite(f)] = -1
        return f, stab

    rng = np.random.default_rng(54)
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:4000], device=dev)
    bank_tr, bank_te, bank_sig = [], [], []
    best_stack = (r0_val, r0_test, 0)
    for rnd in range(args.rounds):
        if sum(c.shape[1] for c in bank_tr) >= args.cap:
            break
        pop = [new_genome(rng, C_prev) for _ in range(args.pop)]
        for gen in range(args.gens):
            cc = torch.stack([feature(torch, tp, gtr, g) for g in pop], 1)
            f, _ = fitness(cc)
            order = torch.argsort(-f).cpu().numpy()
            elite = order[:max(2, args.pop // 4)]
            npop = [pop[i] for i in elite]
            while len(npop) < args.pop:
                src = elite[rng.integers(len(elite))]
                npop.append(mutate(rng, pop[src], float(rng.uniform(0.25, 0.5)), C_prev))
            pop = npop
        cc = torch.stack([feature(torch, tp, gtr, g) for g in pop], 1)
        f, stab = fitness(cc)
        order = torch.argsort(-f).cpu().numpy()
        m2, s2 = cc.mean(0), cc.std(0) + 1e-6
        ccz = (cc - m2) / s2
        sign = ccz[probe] / (ccz[probe].norm(dim=0, keepdim=True) + 1e-8)
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
            cte = torch.stack([feature(torch, tp, gte, pop[i]) for i in adm], 1)
            bank_te.append((cte - m2[sel]) / s2[sel])
            bank_sig.append(sign[:, sel])
        nb = sum(c.shape[1] for c in bank_tr)
        if rnd % 5 == 0 or nb >= args.cap:
            Btr, Bte = torch.cat(bank_tr, 1), torch.cat(bank_te, 1)
            jt, jv, jl = test_of(torch.cat([R0tr, Btr], 1),
                                 torch.cat([R0te, Bte], 1))
            if jv > best_stack[0]:
                best_stack = (jv, jt, nb)
            dv = jv - r0_val
            mark = "  <== +1pt!" if dv >= 0.01 else ""
            log(f"[deep] round {rnd}: +{len(adm)} -> {nb}; stack val {jv:.4f} "
                f"TEST {jt:.4f} (delta val {dv:+.4f}){mark} ({round(time.time()-t0)}s)")
        else:
            log(f"[deep] round {rnd}: +{len(adm)} -> {nb} ({round(time.time()-t0)}s)")

    dv = best_stack[0] - r0_val
    log(f"[deep] BEST stack: val {best_stack[0]:.4f} TEST {best_stack[1]:.4f} "
        f"@ {best_stack[2]} feats (R0 val {r0_val:.4f}, delta {dv:+.4f}) -> "
        f"{'SIGNAL +1pt' if dv >= 0.01 else 'below +1pt bar'}")
    json.dump({"module": "r1deep", "best_val": round(best_stack[0], 4),
               "best_test": round(best_stack[1], 4), "delta_val": round(dv, 4)},
              open(os.path.join(RD, "replicate_r1deep.json"), "w"))


if __name__ == "__main__":
    main()
