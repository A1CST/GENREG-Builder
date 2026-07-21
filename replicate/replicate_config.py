"""replicate_config.py — module 48: SPATIAL CONFIGURATION binding (depth).

Perception axis (m44-47) concluded 0.77 is the single-layer info ceiling —
not readout, pooling, linear enrichment, or nonlinear encoding. The one thing
single-layer structurally cannot represent is the RELATIVE SPATIAL
CONFIGURATION of features. The standard grammar's terms all read the map at
the SAME location (combined elementwise, then ONE window pools), so a genome
detects CO-LOCATED features; it cannot express "feature A in the top region
AND feature B in the bottom region" — a part configuration, which is what
distinguishes objects.

The configuration grammar gives each PART its own window (own position + own
scale), pools each part to a scalar, then binds the parts (product/min/
absdiff). One genome can now require co-occurrence of two features at two
DIFFERENT relative positions and scales — a spatial conjunction outside the
single-layer's reach. Same env (linear patch-PCA), same evolution loop, same
seeds -> matched A/B against the standard-grammar raw baseline (0.6694 @ 573,
module 45). Does spatial part-configuration binding add information beyond
co-located single-layer detectors?

    python3 replicate/replicate_config.py --rounds 80 --cap 2500
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
from radial_evo2 import (Env, make_scorer, SCALES, _PRIMS, _OPS, C_PER_SCALE)

RD = os.path.join(_HERE, "radial_data")
CACHE = os.path.join(_HERE, "replicate", "cache")
STATE = os.path.join(CACHE, "state_config.json")
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def new_part(rng):
    return {"ps": int(rng.choice(SCALES)),
            "c": int(rng.integers(C_PER_SCALE)),
            "prog": [(int(rng.integers(len(_PRIMS))),
                      float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1, 1)))
                     for _ in range(1 if rng.random() < 0.7 else 2)],
            "cx": float(rng.uniform(0.1, 0.9)),
            "cy": float(rng.uniform(0.1, 0.9)),
            "lsig": float(rng.uniform(np.log(0.12), np.log(0.9)))}


def new_genome(rng):
    order = 2 if rng.random() < 0.65 else 3
    return {"parts": [new_part(rng) for _ in range(order)],
            "op": int(rng.integers(len(_OPS)))}


def mutate(rng, g, sc):
    c = json.loads(json.dumps(g))
    for p in c["parts"]:
        if rng.random() < 0.12:
            p["c"] = int(rng.integers(C_PER_SCALE))
        if rng.random() < 0.08:                       # own scale walks
            i = SCALES.index(p["ps"])
            p["ps"] = SCALES[int(np.clip(i + rng.choice([-1, 1]), 0,
                                         len(SCALES) - 1))]
        prog = [list(s) for s in p["prog"]]
        for s in prog:
            if rng.random() < 0.10:
                s[0] = int(rng.integers(len(_PRIMS)))
            s[1] = float(np.clip(s[1] + rng.normal(0, sc), 0.1, 4.0))
            s[2] = float(np.clip(s[2] + rng.normal(0, sc), -2.0, 2.0))
        if rng.random() < 0.10:
            if len(prog) == 1:
                prog.append([int(rng.integers(len(_PRIMS))),
                             float(rng.uniform(0.5, 2.5)),
                             float(rng.uniform(-1, 1))])
            else:
                prog.pop(int(rng.integers(len(prog))))
        p["prog"] = [tuple(s) for s in prog]
        p["cx"] = float(np.clip(p["cx"] + rng.normal(0, sc * 0.6), 0.02, 0.98))
        p["cy"] = float(np.clip(p["cy"] + rng.normal(0, sc * 0.6), 0.02, 0.98))
        p["lsig"] = float(np.clip(p["lsig"] + rng.normal(0, sc * 0.5),
                                  np.log(0.05), np.log(1.5)))
    if rng.random() < 0.08:                           # order evolves
        if len(c["parts"]) == 2:
            c["parts"].append(new_part(rng))
        else:
            c["parts"].pop(int(rng.integers(len(c["parts"]))))
    if rng.random() < 0.08:
        c["op"] = int(rng.integers(len(_OPS)))
    return c


def feature(torch, tp, env, g, test=False):
    """Each part self-pools with its OWN window at its OWN scale -> a scalar;
    parts are then bound. Detects spatial configurations of parts."""
    z = None
    for p in g["parts"]:
        Mtr, Mte, H, W = env.maps(p["ps"])
        M = Mte if test else Mtr
        v = M[:, p["c"] % M.shape[1], :].float().view(len(M), H, W)
        for prim, a, b in p["prog"]:
            v = tp[_PRIMS[prim]](a * v + b)
        ys = torch.linspace(0, 1, H, device=v.device).view(H, 1)
        xs = torch.linspace(0, 1, W, device=v.device).view(1, W)
        sig = float(np.exp(p["lsig"]))
        wgt = torch.exp(-(((xs - p["cx"]) ** 2) + ((ys - p["cy"]) ** 2))
                        / (2 * sig * sig))
        s = (v * wgt).sum((1, 2)) / (wgt.sum() + 1e-9)   # part scalar
        if z is None:
            z = s
        else:
            op = _OPS[g["op"]]
            z = z * s if op == "mult" else (torch.minimum(z, s) if op == "min"
                                            else torch.abs(z - s))
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=10)
    args = ap.parse_args()
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
    env = Env(torch, dev, Xtr, Xte, max_cached=8)
    log(f"[cfg] configuration grammar (per-part window+scale), "
        f"{torch.cuda.get_device_name(0)}")

    n_fit = int(N * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yfull = -torch.ones((N, 10), device=dev)
    Yfull[torch.arange(N), torch.tensor(ytr, device=dev)] = 1.0

    state = {"genomes": [], "round": 0}
    if os.path.isfile(STATE):
        state = json.load(open(STATE, encoding="utf-8"))
    rng = np.random.default_rng(4500 + 1000 * state["round"])
    cols_tr = [feature(torch, tp, env, g) for g in state["genomes"]]
    cols_te = [feature(torch, tp, env, g, test=True) for g in state["genomes"]]
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:3000],
                         device=dev)
    log(f"[cfg] {len(state['genomes'])} genomes, round {state['round']}")

    for rnd in range(args.rounds):
        if len(state["genomes"]) >= args.cap:
            break
        pr = torch.tensor(rng.permutation(N), device=dev)
        yy = torch.tensor(ytr, device=dev)[pr]
        Yf = -torch.ones((n_fit, 10), device=dev)
        Yf[torch.arange(n_fit), yy[:n_fit]] = 1.0
        base = (torch.stack(cols_tr, 1)[pr] if cols_tr
                else torch.zeros((N, 0), device=dev))
        scorer, s0, a0 = make_scorer(torch, base, n_fit, Yf, yy[n_fit:])
        pr2 = torch.tensor(rng.permutation(N), device=dev)
        yy2 = torch.tensor(ytr, device=dev)[pr2]
        Yf2 = -torch.ones((n_fit, 10), device=dev)
        Yf2[torch.arange(n_fit), yy2[:n_fit]] = 1.0
        base2 = (torch.stack(cols_tr, 1)[pr2] if cols_tr
                 else torch.zeros((N, 0), device=dev))
        scorer2, _s2, a02 = make_scorer(torch, base2, n_fit, Yf2, yy2[n_fit:])

        pop = [new_genome(rng) for _ in range(args.pop)]
        for gen in range(args.gens):
            cc = torch.stack([feature(torch, tp, env, g) for g in pop], 1)
            _, accs = scorer(cc[pr])
            accs = np.array(accs)
            order = np.argsort(-accs)
            elite = order[:max(2, args.pop // 4)]
            npop = [pop[i] for i in elite]
            while len(npop) < args.pop:
                npop.append(mutate(rng, pop[elite[rng.integers(len(elite))]],
                                   float(rng.uniform(0.25, 0.5))))
            pop = npop
        cc = torch.stack([feature(torch, tp, env, g) for g in pop], 1)
        _, accs = scorer(cc[pr])
        accs = np.array(accs)
        order = np.argsort(-accs)
        adm, sigs, n_o, n_v = [], [], 0, 0
        for c in cols_tr[-64:]:
            s = c[probe] - c[probe].mean()
            sigs.append(s / (s.norm() + 1e-8))
        pcc = cc[probe]
        for i in order:
            if accs[i] - a0 < 0.0004 or len(adm) >= 8:
                break
            s = pcc[:, i] - pcc[:, i].mean()
            s = s / (s.norm() + 1e-8)
            if any(float(torch.abs(s @ t)) > 0.85 for t in sigs):
                n_o += 1
                continue
            _, a2 = scorer2(cc[pr2][:, i:i + 1])
            if a2[0] - a02 < 0.0002:
                n_v += 1
                continue
            adm.append(i)
            sigs.append(s)
        for i in adm:
            state["genomes"].append(pop[i])
            cols_tr.append(cc[:, i])
            cols_te.append(feature(torch, tp, env, pop[i], test=True))
        state["round"] += 1
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(state, f)
        if rnd % 5 == 0 or len(state["genomes"]) >= args.cap:
            Ftr = torch.stack(cols_tr, 1)
            Yf0 = -torch.ones((n_fit, 10), device=dev)
            Yf0[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
            bv = max(_ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf0, yv,
                                 lam=l)[1] for l in (1.0, 3.0, 10.0, 30.0))
            log(f"[cfg] round {state['round']-1}: +{len(adm)} -> "
                f"{len(state['genomes'])} (rej {n_o},{n_v}); val {bv:.4f} "
                f"({round(time.time()-t0)}s)")
        else:
            log(f"[cfg] round {state['round']-1}: +{len(adm)} -> "
                f"{len(state['genomes'])} (rej {n_o},{n_v}) "
                f"({round(time.time()-t0)}s)")

    Ftr = torch.stack(cols_tr, 1)
    Fte = torch.stack(cols_te, 1)
    bl, bv = 3.0, -1.0
    Yf0 = -torch.ones((n_fit, 10), device=dev)
    Yf0[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    for lam in (1.0, 3.0, 10.0, 30.0, 100.0):
        _, a = _ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf0, yv, lam=lam)
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
    log(f"[cfg] FINAL [configuration]: {len(state['genomes'])} genomes, "
        f"val {bv:.4f} TEST {acc:.4f} (std-grammar raw 0.6694 @ 573)")
    json.dump({"module": "config", "n": len(state["genomes"]),
               "test": round(acc, 4)},
              open(os.path.join(RD, "replicate_config.json"), "w"))


if __name__ == "__main__":
    main()
