"""anim_ablate.py — ablation suite for the SHAPE checkpoint.

The shape model trained on shapes riding the animation dataset's ten
motion paths. Is it motion-invariant, or did it only learn shapes-under-
those-ten-paths? Each ablation set keeps the task identical (name the
shape, 10 classes) but swaps the motion regime for one the model NEVER
saw:

  control      — the trained paths, fresh draw (anchors the suite)
  static       — no motion at all (a degenerate "pattern")
  random-walk  — Brownian jitter, no path structure
  bezier       — a smooth random curve, unseen path family
  fast-2x      — trained paths at double speed (stride-2 window)
  wide-offset  — trained paths shifted far outside the trained ±6 range

The checkpoint's genomes are FIXED; the head is fit once on the ORIGINAL
training set and reused unchanged for every ablation — nothing adapts.
Exports radial_data/anim_ablation.json for the Animation page.

    python anim_ablate.py          # ~1-2 min on GPU
"""
import json
import os
import time

import numpy as np

import genreg_paths                               # noqa: F401
from radial_anim import T, PATHS, SHAPES, SHAPE_NAMES
from radial_evo import _tprims
from radial_evo2 import Env
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))
F = 24                                   # frames in a full clip (ad.FRAMES)
N_AB = 1500                              # sequences per ablation set


def _finish(X, rng, noise=0.05):
    X = X * rng.uniform(0.7, 1.0, (len(X), 1, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    return (np.repeat(X[..., None], 3, axis=4) * 255).astype(np.uint8)


def gen(kind, n=N_AB, seed=101):
    """n sequences of a random shape under the given motion regime."""
    rng = np.random.default_rng(seed + sum(ord(c) for c in kind))
    y = rng.integers(0, len(SHAPES), n)
    X = np.zeros((n, T, 32, 32), np.float32)
    for i in range(n):
        sfn = SHAPES[y[i]]
        if kind == "control":            # the trained regime, fresh draw
            pfn = PATHS[rng.integers(0, len(PATHS))][1]
            s = rng.integers(0, F - T + 1)
            oy, ox = rng.uniform(-6, 6), rng.uniform(-6, 6)
            cs = [pfn((s + f) / (F - 1)) for f in range(T)]
            cs = [(cx + ox, cy + oy) for cx, cy in cs]
        elif kind == "static":           # never seen: no motion
            cx, cy = rng.uniform(12, 52), rng.uniform(12, 52)
            cs = [(cx, cy)] * T
        elif kind == "random-walk":      # never seen: Brownian jitter
            cx, cy = rng.uniform(14, 50), rng.uniform(14, 50)
            cs = []
            for _f in range(T):
                cs.append((cx, cy))
                cx = float(np.clip(cx + rng.normal(0, 2.5), 6, 58))
                cy = float(np.clip(cy + rng.normal(0, 2.5), 6, 58))
        elif kind == "bezier":           # never seen: smooth random curve
            P = rng.uniform(10, 54, (4, 2))
            s = rng.integers(0, F - T + 1)
            cs = []
            for f in range(T):
                t = (s + f) / (F - 1)
                u = 1.0 - t
                p = (u**3 * P[0] + 3 * u**2 * t * P[1]
                     + 3 * u * t**2 * P[2] + t**3 * P[3])
                cs.append((float(p[0]), float(p[1])))
        elif kind == "fast-2x":          # trained paths, double speed
            pfn = PATHS[rng.integers(0, len(PATHS))][1]
            s = rng.integers(0, F - 2 * T + 2)
            oy, ox = rng.uniform(-6, 6), rng.uniform(-6, 6)
            cs = [pfn((s + 2 * f) / (F - 1)) for f in range(T)]
            cs = [(cx + ox, cy + oy) for cx, cy in cs]
        elif kind == "wide-offset":      # trained paths, offsets beyond +-6
            pfn = PATHS[rng.integers(0, len(PATHS))][1]
            s = rng.integers(0, F - T + 1)
            oy = rng.uniform(6, 12) * rng.choice([-1, 1])
            ox = rng.uniform(6, 12) * rng.choice([-1, 1])
            cs = [pfn((s + f) / (F - 1)) for f in range(T)]
            cs = [(cx + ox, cy + oy) for cx, cy in cs]
        else:
            raise ValueError(kind)
        for f, (cx, cy) in enumerate(cs):
            cx = float(np.clip(cx, 4, 60))
            cy = float(np.clip(cy, 4, 60))
            X[i, f] = sfn(cx, cy).reshape(32, 2, 32, 2).mean((1, 3))
    return _finish(X, rng), y


ABLATIONS = [
    ("control", "the ten trained paths, fresh draw — the anchor"),
    ("static", "no motion at all"),
    ("random-walk", "Brownian jitter, no path structure"),
    ("bezier", "smooth random curves — an unseen path family"),
    ("fast-2x", "trained paths at double speed"),
    ("wide-offset", "trained paths pushed 6-12 px outside the trained range"),
]


from anim_infer import count_params


def _features(torch, tp, env, ckpt, N, test):
    G = rk.GRID
    cols, prev = [], None
    for si, genomes in enumerate(ckpt["spaces"]):
        if si == 0:
            f = lambda g: rk.feature_r0(torch, tp, env, g,
                                        test=test).view(N, T).mean(1)
            gr = lambda g: rk.feature_r0(torch, tp, env, g, test=test,
                                         want_grid=True).view(N, T, G, G)
        else:
            f = lambda g: rk.feature_grid_g(torch, tp, prev, g)
            gr = lambda g: rk.feature_grid_g(torch, tp, prev, g, want_grid=True)
        cols.extend(f(g) for g in genomes)
        if si + 1 < len(ckpt["spaces"]):
            bank = [gr(g) for g in genomes]
            prev = (torch.cat(bank, 1) if si == 0
                    else torch.stack(bank, 1)).half()
    return torch.stack(cols, 1).float()


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()

    with open(os.path.join(_HERE, "radial_data", "anim_model_shape.json")) as f:
        ckpt = json.load(f)
    rk.GRID = int(ckpt["grid"])

    z = np.load(os.path.join(_HERE, "radial_data", "anim_seq.npz"))
    XtrF = (z["Xtr"].astype(np.float32) / 255.0).reshape(-1, 32, 32, 3)
    ytr = z["ytr_shape"]
    Ntr = len(ytr)

    # head fit ONCE on the original training set; reused frozen everywhere
    env0 = Env(torch, dev, XtrF, XtrF[: T], max_cached=6)
    Ftr = _features(torch, tp, env0, ckpt, Ntr, test=False)
    ytr_t = torch.tensor(ytr, device=dev)
    Y = -torch.ones((Ntr, 10), device=dev)
    Y[torch.arange(Ntr), ytr_t] = 1.0

    def _fit(Xf, Yf, lam):
        n, d = Xf.shape
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        A = torch.hstack([(Xf - mu) / sd, torch.ones(n, 1, device=dev)])
        W = torch.linalg.solve(A.T @ A + lam * torch.eye(d + 1, device=dev),
                               A.T @ Yf)
        return mu, sd, W

    n_fit = int(Ntr * 0.8)
    best = (None, -1.0)
    for lam in (1.0, 3.0, 10.0, 30.0):
        mu, sd, W = _fit(Ftr[:n_fit], Y[:n_fit], lam)
        s = torch.hstack([(Ftr[n_fit:] - mu) / sd,
                          torch.ones(Ntr - n_fit, 1, device=dev)]) @ W
        acc = float((s.argmax(1) == ytr_t[n_fit:]).float().mean())
        if acc > best[1]:
            best = (lam, acc)
    mu, sd, W = _fit(Ftr, Y, best[0])
    print(f"head fit: lam {best[0]} ({round(time.time()-t0)}s)", flush=True)

    results = []
    for kind, desc in ABLATIONS:
        X8, y = gen(kind)
        envA = Env(torch, dev, XtrF,
                   (X8.astype(np.float32) / 255.0).reshape(-1, 32, 32, 3),
                   max_cached=6)
        Fte = _features(torch, tp, envA, ckpt, len(y), test=True)
        s = torch.hstack([(Fte - mu) / sd,
                          torch.ones(len(y), 1, device=dev)]) @ W
        preds = s.argmax(1).cpu().numpy()
        acc = float((preds == y).mean())
        per_class = {SHAPE_NAMES[c]: round(float((preds[y == c] == c).mean()), 4)
                     for c in range(len(SHAPE_NAMES))}
        results.append({"kind": kind, "desc": desc, "n": len(y),
                        "acc": round(acc, 4), "per_class": per_class})
        print(f"  {kind:12s} acc {acc:.4f}  ({round(time.time()-t0)}s)",
              flush=True)

    n_genomes = sum(len(sp) for sp in ckpt["spaces"])
    genome_params = sum(count_params(g) for sp in ckpt["spaces"] for g in sp)
    head_params = int(W.numel())
    out = {"model": "shape", "chance": 0.1, "n_per_set": N_AB,
           "note": "genomes and head FROZEN from the shape checkpoint; "
                   "nothing adapts to the new motion regimes",
           "n_genomes": n_genomes,
           "genome_params": genome_params,
           "head_params": head_params,
           "total_params": genome_params + head_params,
           "results": results, "seconds": round(time.time() - t0)}
    op = os.path.join(_HERE, "radial_data", "anim_ablation.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[anim-ablate] DONE -> {op}", flush=True)


if __name__ == "__main__":
    main()
