"""replicate_climb.py — module 55: is the AUGMENTED-MEASUREMENT axis a LEVER
that CLIMBS, or does it saturate too?

The R1-over-frozen-R0 work capped at +0.5pt because a readout of FIXED
information cannot climb (re-readings earn 0). The only axis that has ever
compounded in this project is NEW MEASUREMENTS: augmented multi-view replays
(0.77 -> 0.7967). The user's criterion for a real lever: accuracy must CLIMB
with effort, not go up 0.5 and flatten. So measure the curve directly —
accuracy vs number of augmented measurements K — and see if it ladders.

Each view is a fresh random augmentation (crop/flip/brightness) of every
image; the FROZEN R0 genomes re-measure it (a replay = a new measurement, not
a re-reading of the same pixels). Features are averaged across views (a better
estimate of each image's true response) for both train and test, then a ridge
reads them. Report TEST accuracy at K = 1,2,4,8,16.

CRITICAL (pca-basis-not-portable law): augmented views reuse R0's FROZEN
patch-PCA basis (seeded into _SVD_CACHE), never rebuild it.

    python3 replicate/replicate_climb.py --views 16
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

from radial_evo import _tprims
import radial_evo2 as re2
from radial_evo2 import Env, SCALES
import radial_stack as rk
from replicate_r1build import union_genomes

RD = os.path.join(_HERE, "radial_data")
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def _fp(Xnp):
    return (Xnp.shape, float(Xnp[:64].sum()), float(Xnp[-64:].sum()),
            float(Xnp.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=16)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    tp = _tprims(torch)
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr0 = z["Xtr"].astype(np.float32) / 255.0
    Xte0 = z["Xte"].astype(np.float32) / 255.0
    ytr = torch.tensor(z["ytr"].astype(np.int64), device=dev)
    yte = torch.tensor(z["yte"].astype(np.int64), device=dev)
    N = len(z["ytr"])
    gs = union_genomes()
    n_fit = int(N * 0.8)
    yv = ytr[n_fit:]
    Y = -torch.ones((N, 10), device=dev)
    Y[torch.arange(N), ytr] = 1.0
    log(f"[climb] {len(gs)} frozen R0 genomes, {torch.cuda.get_device_name(0)}")

    # clean Env -> freeze the patch-PCA basis for every scale
    env0 = Env(torch, dev, Xtr0, Xte0, max_cached=len(SCALES))
    for ps in SCALES:
        env0.maps(ps)
    clean_basis = {ps: re2._SVD_CACHE[(_fp(Xtr0), ps)] for ps in SCALES}
    log(f"[climb] froze basis for scales {SCALES} ({round(time.time()-t0)}s)")

    def feats(env):
        ftr = torch.stack([rk.feature_r0(torch, tp, env, g) for g in gs], 1)
        fte = torch.stack([rk.feature_r0(torch, tp, env, g, test=True)
                           for g in gs], 1)
        return ftr, fte

    def test_of(Ftr, Fte):
        n, d = Ftr.shape
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        A = torch.hstack([((Ftr - mu) / sd)[:n_fit], torch.ones(n_fit, 1, device=dev)])
        Gm = (A.T @ A).double()
        Av = torch.hstack([((Ftr - mu) / sd)[n_fit:],
                           torch.ones(n - n_fit, 1, device=dev)])
        best = (-1, None)
        for lam in (1.0, 3.0, 10.0, 30.0, 100.0):
            W = torch.linalg.solve(Gm + lam * torch.eye(d + 1, device=dev,
                                   dtype=torch.float64),
                                   (A.T @ Y[:n_fit]).double()).float()
            va = float(((Av @ W).argmax(1) == yv).float().mean())
            if va > best[0]:
                best = (va, lam)
        lam = best[1]
        Af = torch.hstack([(Ftr - mu) / sd, torch.ones(n, 1, device=dev)])
        W = torch.linalg.solve((Af.T @ Af).double() + lam * torch.eye(
            d + 1, device=dev, dtype=torch.float64),
            (Af.T @ Y).double()).float()
        At = torch.hstack([(Fte - mu) / sd, torch.ones(len(Fte), 1, device=dev)])
        ta = float(((At @ W).argmax(1) == yte).float().mean())
        return ta, best[0]

    gen = torch.Generator(device=dev).manual_seed(0)

    def augment(Xnp, view):
        # PHOTOMETRIC only — brightness / contrast / per-channel color / gamma /
        # noise. These preserve spatial alignment, so averaging features across
        # views is a clean denoise (geometric augs would misalign the genomes'
        # spatial windows and blur the average).
        if view == 0:
            return Xnp                                   # view 0 = clean
        t = torch.tensor(Xnp, device=dev).permute(0, 3, 1, 2)
        n = len(t)
        b = 1 + (torch.rand(n, 1, 1, 1, generator=gen, device=dev) - 0.5) * 0.5
        con = 1 + (torch.rand(n, 1, 1, 1, generator=gen, device=dev) - 0.5) * 0.5
        col = 1 + (torch.rand(n, 3, 1, 1, generator=gen, device=dev) - 0.5) * 0.3
        gam = 0.7 + torch.rand(n, 1, 1, 1, generator=gen, device=dev) * 0.6
        mean = t.mean((2, 3), keepdim=True)
        t = ((t - mean) * con + mean) * b * col
        t = t.clamp(1e-4, 1) ** gam
        t = t + torch.randn(t.shape, generator=gen, device=dev) * 0.03
        return t.clamp(0, 1).permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def env_for(Xa, Xb):
        e = Env(torch, dev, Xa, Xb, max_cached=len(SCALES))
        afp = _fp(Xa)
        for ps in SCALES:
            re2._SVD_CACHE[(afp, ps)] = clean_basis[ps]
        return e

    # accumulate running mean of features across views; test at K checkpoints
    ckpts = [k for k in (1, 2, 4, 8, 16, 32) if k <= args.views]
    sum_tr = torch.zeros((N, len(gs)), device=dev)
    sum_te = torch.zeros((len(yte), len(gs)), device=dev)
    curve = []
    for v in range(max(ckpts) if ckpts else 1):
        Xa = augment(Xtr0, v)
        Xb = augment(Xte0, v)
        env = env_for(Xa, Xb) if v > 0 else env0
        ftr, fte = feats(env)
        sum_tr += ftr
        sum_te += fte
        if v > 0:
            del env
        torch.cuda.empty_cache()
        K = v + 1
        if K in ckpts:
            ta, va = test_of(sum_tr / K, sum_te / K)
            curve.append((K, va, ta))
            base = curve[0][2]
            log(f"[climb] K={K:2d} views: val {va:.4f} TEST {ta:.4f} "
                f"(vs K=1 {base:.4f}, {ta - base:+.4f}) ({round(time.time()-t0)}s)")

    log("[climb] CURVE (K, val, test): " +
        " | ".join(f"{k}:{t:.4f}" for k, _, t in curve))
    if len(curve) >= 3:
        early = curve[1][2] - curve[0][2]
        late = curve[-1][2] - curve[-2][2]
        log(f"[climb] early slope (K1->K2) {early:+.4f} vs late slope "
            f"(last step) {late:+.4f} -> "
            f"{'CLIMBING' if late > 0.002 else 'saturating'}")
    json.dump({"module": "climb", "curve": [(k, round(t, 4)) for k, _, t in curve]},
              open(os.path.join(RD, "replicate_climb.json"), "w"))


if __name__ == "__main__":
    main()
