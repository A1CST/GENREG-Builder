"""replicate_richenv.py — module 45: a richer PERCEPTION ENVIRONMENT.

Spatial-layout was a null (module 44): the genome's learned pool is already
right; the limit is WHAT it detects. Every genome reads patch-PCA of RAW
pixels — color blobs + dominant edges, nothing finer. This gives evolution
richer material to detect: the image is enriched with gradient/edge channels
(Sobel-x, Sobel-y, gradient magnitude) alongside RGB, so the patch-PCA basis
captures oriented edge/texture structure — the fine boundary information
reconstruction showed is missing — and genomes can compose detectors over it.

Then evolution SEARCHES this richer space for R0 detectors (class-informed,
orthogonal admission), and we measure whether the ridge ceiling exceeds the
raw-pixel substrate's 0.77. If a richer environment lifts perception, the
lever is real and worth scaling.

    python3 replicate/replicate_richenv.py --rounds 60 --cap 2500
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
from radial_evo2 import Env, make_scorer, new_genome, mutate, feature
from replicate_cifar import load_cifar

RD = os.path.join(_HERE, "radial_data")
CACHE = os.path.join(_HERE, "replicate", "cache")
STATE = os.path.join(CACHE, "state_richenv.json")
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def enrich(X):
    """(N,32,32,3) -> (N,32,32,7): RGB + gray-Sobel-x/y + grad-mag + luminance.
    Gives the patch-PCA basis oriented-edge and texture structure."""
    g = X.mean(3)                                  # (N,32,32) luminance
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, :, 1:-1] = g[:, :, 2:] - g[:, :, :-2]
    gy[:, 1:-1, :] = g[:, 2:, :] - g[:, :-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.concatenate([X, gx[..., None], gy[..., None],
                           mag[..., None], g[..., None]], 3).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--raw", action="store_true",
                    help="ablation: 3-channel raw pixels, no enrichment "
                         "(matched-loop control for the enrichment effect)")
    args = ap.parse_args()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    tp = _tprims(torch)
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    _e = (lambda A: A) if args.raw else enrich
    Xtr = _e(z["Xtr"].astype(np.float32) / 255.0)
    Xte = _e(z["Xte"].astype(np.float32) / 255.0)
    ytr, yte = z["ytr"].astype(np.int64), z["yte"].astype(np.int64)
    N = len(ytr)
    env = Env(torch, dev, Xtr, Xte, max_cached=6)
    tag = "RAW" if args.raw else "enriched"
    log(f"[rich] {tag} env {Xtr.shape}, {torch.cuda.get_device_name(0)}")

    n_fit = int(N * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yfull = -torch.ones((N, 10), device=dev)
    Yfull[torch.arange(N), torch.tensor(ytr, device=dev)] = 1.0

    state_path = STATE.replace(".json", "_raw.json") if args.raw else STATE
    state = {"genomes": [], "round": 0}
    if os.path.isfile(state_path):
        state = json.load(open(state_path, encoding="utf-8"))
    rng = np.random.default_rng(4500 + 1000 * state["round"])
    cols_tr = [feature(torch, tp, env, g) for g in state["genomes"]]
    cols_te = [feature(torch, tp, env, g, test=True) for g in state["genomes"]]
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:3000],
                         device=dev)
    log(f"[rich] {len(state['genomes'])} genomes, round {state['round']}")

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
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        if rnd % 5 == 0 or len(state["genomes"]) >= args.cap:
            Ftr = torch.stack(cols_tr, 1)
            Yf0 = -torch.ones((n_fit, 10), device=dev)
            Yf0[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
            bv = max(_ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf0, yv,
                                 lam=l)[1] for l in (1.0, 3.0, 10.0, 30.0))
            log(f"[rich] round {state['round']-1}: +{len(adm)} -> "
                f"{len(state['genomes'])} (rej {n_o},{n_v}); val {bv:.4f} "
                f"({round(time.time()-t0)}s)")
        else:
            log(f"[rich] round {state['round']-1}: +{len(adm)} -> "
                f"{len(state['genomes'])} (rej {n_o},{n_v}) "
                f"({round(time.time()-t0)}s)")

    # final ridge test
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
    log(f"[rich] FINAL [{tag}]: {len(state['genomes'])} genomes, "
        f"val {bv:.4f} TEST {acc:.4f}")
    json.dump({"module": "richenv", "tag": tag, "n": len(state["genomes"]),
               "test": round(acc, 4)},
              open(os.path.join(RD, f"replicate_richenv_{tag}.json"), "w"))


if __name__ == "__main__":
    main()
