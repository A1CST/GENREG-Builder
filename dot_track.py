"""dot_track.py — FIRST model of the red-dot line.

Goal: track (localize) a single RED DOT. A red dot sits at a random position on
a plain black background; the model must output its (x, y). This is the lab's
first REGRESSION task — the radial features are evolved gradient-free exactly as
before (patch-PCA environment, grammar-v2 spatial genomes, freeze-and-compose),
but the read-out is a closed-form ridge REGRESSION to position, and fitness is
how much a feature explains the position residual (greedy residual boosting).
The evolved soft spatial window is a natural position detector, so features
encode "where" for free. Reports mean pixel error + R^2. No gradients.

  python dot_track.py            # GPU if free
"""
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _STOP
from radial_evo2 import Env, new_genome, mutate, feature
import radial_anim as ra

try:
    from genreg_train import animation_data as ad
except ImportError:
    import animation_data as ad

_HERE = os.path.dirname(os.path.abspath(__file__))


# non-red distractor colors (the red cursor stays uniquely red)
_DCOL = np.array([[0.2, 0.85, 0.25], [0.2, 0.45, 1.0], [0.1, 0.85, 0.9],
                  [0.95, 0.9, 0.2], [0.75, 0.35, 0.95], [0.6, 0.6, 0.6]], np.float32)


def _rand_color(rng):
    """A random visible NON-red color. The target shape AND the distractors both
    draw from this, so the target is indistinguishable by color — the red cursor
    on it is the ONLY cue that marks it as the thing to track."""
    while True:
        c = rng.uniform(0.15, 1.0, 3).astype(np.float32)
        if not (c[0] > 0.6 and c[1] < 0.4 and c[2] < 0.4):   # keep red for the cursor
            return c


def gen_dot(n, res=32, seed=0, noise=0.03, with_shape=True, distractors=0):
    """n frames (res,res,3): a red CURSOR dot pinned to the centre of a (white)
    shape; targets = the normalised (x, y) in [0,1]. The model must output the
    RED cursor's location. `distractors` adds that many non-red shapes of random
    color / shape / size — the model must ignore them. Composited at native 64,
    area-downscaled."""
    rng = np.random.default_rng(seed)
    S = ad.SIZE
    SH = ra.SHAPES
    X = np.zeros((n, 1, S, S, 3), np.float32)
    pos = np.zeros((n, 2), np.float32)
    for i in range(n):
        x = float(rng.uniform(10, S - 10))
        y = float(rng.uniform(10, S - 10))
        for k in range(distractors):               # colored distractor shapes (behind)
            dsfn = SH[int(rng.integers(len(SH)))]
            da = dsfn(float(rng.uniform(8, S - 8)), float(rng.uniform(8, S - 8)),
                      r=float(rng.uniform(3.0, 9.0)))
            col = _rand_color(rng)
            X[i, 0] = col[None, None, :] * da[..., None] + X[i, 0] * (1.0 - da[..., None])
        if with_shape:                              # target shape — a random color, like the
            sfn = SH[int(rng.integers(len(SH)))]    # distractors; only the cursor marks it
            sa = sfn(x, y, r=float(rng.uniform(4.5, 7.0)))
            tcol = _rand_color(rng)
            X[i, 0] = tcol[None, None, :] * sa[..., None] + X[i, 0] * (1.0 - sa[..., None])
        da = ad.circle(x, y, r=float(rng.uniform(2.5, 3.8)))   # red cursor on top
        red = np.array([1.0, 0.0, 0.0], np.float32)
        X[i, 0] = red[None, None, :] * da[..., None] + X[i, 0] * (1.0 - da[..., None])
        pos[i] = [x / S, y / S]
    X = ra._downscale(X, res)[:, 0]                 # (n, res, res, 3)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    return X.astype(np.float32), pos


def run(n_train=6000, n_test=1500, res=32, rounds=60, pop=64, gens=10,
        freeze_top=8, seed=0, out_path=None, verbose=True, distractors=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    Xtr, ptr = gen_dot(n_train, res, seed=1, distractors=distractors)
    Xte, pte = gen_dot(n_test, res, seed=2, distractors=distractors)
    env = Env(torch, dev, Xtr, Xte, max_cached=6)
    Y = torch.tensor(ptr, device=dev)
    Yte = torch.tensor(pte, device=dev)
    n_fit = int(n_train * 0.8)
    rng = np.random.default_rng(seed)

    def zc(F):                                      # standardise + bias column
        mu, sd = F[:n_fit].mean(0), F[:n_fit].std(0) + 1e-6
        A = torch.cat([(F - mu) / sd, torch.ones(len(F), 1, device=dev)], 1)
        return A

    def ridge(Ftr, Ytr, Fev):
        A = zc(Ftr)[:n_fit]
        W = torch.linalg.solve(A.T @ A + 3.0 * torch.eye(A.shape[1], device=dev),
                               A.T @ Ytr[:n_fit])
        return zc(Fev) @ W if Fev is Ftr else None

    def r2_err(F):
        A = zc(F)
        W = torch.linalg.solve(A[:n_fit].T @ A[:n_fit] + 3.0 * torch.eye(A.shape[1], device=dev),
                               A[:n_fit].T @ Y[:n_fit])
        pv = A[n_fit:] @ W
        yv = Y[n_fit:]
        ss = ((pv - yv) ** 2).sum(0)
        tot = ((yv - yv.mean(0)) ** 2).sum(0)
        r2 = float((1 - ss / tot).mean())
        px = float((((pv - yv) * ad.SIZE) ** 2).sum(1).sqrt().mean())
        return r2, px

    frozen, fcols = [], []
    for rnd in range(rounds):
        # current residual on the FIT split (what the frozen bank can't explain)
        if fcols:
            base = torch.stack(fcols, 1)
            A = zc(base)
            W = torch.linalg.solve(A[:n_fit].T @ A[:n_fit] + 3.0 * torch.eye(A.shape[1], device=dev),
                                   A[:n_fit].T @ Y[:n_fit])
            resid = Y - A @ W                       # (N, 2)
        else:
            resid = Y - Y[:n_fit].mean(0)

        def fit_pop(gs):
            cols = [feature(torch, tp, env, g) for g in gs]
            score = np.full(len(gs), -1e9)
            for i, c in enumerate(cols):
                if float(c.std()) < 1e-6 or not bool(torch.isfinite(c).all()):
                    continue
                cz = (c[:n_fit] - c[:n_fit].mean()) / (c[:n_fit].std() + 1e-6)
                # how much this feature explains the fit residual (both axes)
                score[i] = float((cz.view(-1, 1) * resid[:n_fit]).mean(0).abs().sum())
            return score, cols

        pgen = [new_genome(rng) for _ in range(pop)]
        scales = np.full(pop, 0.25)
        fits, cols = fit_pop(pgen)
        for _ in range(gens):
            order = np.argsort(fits)[::-1]
            keep = list(order[:6])
            kids, ksc = [], []
            while len(kids) < pop - 6:
                cand = rng.choice(pop, 3)
                pi = cand[int(np.argmax(fits[cand]))]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                kids.append(mutate(rng, pgen[pi], sc))
                ksc.append(sc)
            kf, kc = fit_pop(kids)
            pgen = [pgen[i] for i in keep] + kids
            scales = np.concatenate([scales[keep], ksc])
            fits = np.concatenate([fits[keep], kf])
            cols = [cols[i] for i in keep] + kc
        # freeze top decorrelated positive contributors
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= 1e-5 or added >= freeze_top:
                break
            c = cols[idx]
            cz = (c - c.mean()) / (c.std() + 1e-9)
            dup = any(float(torch.abs((cz * ((fc - fc.mean()) / (fc.std() + 1e-9))).mean())) > 0.95
                      for fc in fcols[-60:])
            if not dup:
                frozen.append(pgen[idx]); fcols.append(c); added += 1
        r2, px = r2_err(torch.stack(fcols, 1))
        if verbose:
            print(f"  round {rnd:3d}  +{added} (feats {len(frozen)})  val R2 {r2:.4f}  "
                  f"err {px:.2f}px  ({round(time.time()-t0)}s)", flush=True)
        if os.path.exists(_STOP) or (added == 0 and rnd > 3):
            break

    # honest test: touched once
    Ftr = torch.stack(fcols, 1)
    Fte = torch.stack([feature(torch, tp, env, g, test=True) for g in frozen], 1)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    A = torch.cat([(Ftr - mu) / sd, torch.ones(len(Ftr), 1, device=dev)], 1)
    B = torch.cat([(Fte - mu) / sd, torch.ones(len(Fte), 1, device=dev)], 1)
    W = torch.linalg.solve(A.T @ A + 3.0 * torch.eye(A.shape[1], device=dev), A.T @ Y)
    pred = B @ W
    ss = ((pred - Yte) ** 2).sum(0)
    tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
    test_r2 = float((1 - ss / tot).mean())
    err_px = float((((pred - Yte) * ad.SIZE) ** 2).sum(1).sqrt().mean())
    med_px = float((((pred - Yte) * ad.SIZE) ** 2).sum(1).sqrt().median())
    out = {"experiment": "red-cursor tracker", "task": "localize the red cursor (x,y regression)",
           "res": res, "distractors": distractors,
           "n_train": n_train, "n_test": n_test, "n_feats": len(frozen),
           "test_r2": round(test_r2, 4), "mean_err_px": round(err_px, 3),
           "median_err_px": round(med_px, 3), "frame_px": ad.SIZE,
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(_HERE, "radial_data", "dot_track.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    # save the model so inference (following a moving cursor) can reload it
    model = {"genomes": frozen, "res": res, "distractors": distractors,
             "W": W.cpu().numpy().tolist(),
             "mu": mu.cpu().numpy().tolist(), "sd": sd.cpu().numpy().tolist()}
    with open(os.path.join(_HERE, "radial_data", "dot_model.json"), "w") as f:
        json.dump(model, f)
    print(f"[dot-track] DONE: {len(frozen)} feats, test R2 {test_r2:.4f}, "
          f"mean err {err_px:.2f}px (median {med_px:.2f}px) on a {ad.SIZE}px frame "
          f"({out['seconds']}s)", flush=True)
    return out


if __name__ == "__main__":
    run(res=64, distractors=3)
