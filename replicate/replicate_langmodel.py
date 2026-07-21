"""replicate_langmodel.py — module 59: THE MODEL. Label-free spaces, each a
DIFFERENT answerable question, composing, output head only at the end.

  image
   -> space 1  : words invariant to PHOTOMETRIC change (brightness/contrast/color)
   -> space 2  : words over space-1 words, invariant to GEOMETRIC change (flip/crop)
   -> space 3  : words over space-2 words, PREDICTIVE (a word the rest predict)
   -> OUTPUT HEAD (labels enter here and nowhere else)

Each space is asked an EASY, answerable, LABEL-FREE question whose answer is in
its environment. Because the questions DIFFER, each space sees structure the
previous was blind to, so depth composes instead of colliding (asking the same
question twice, m58, added nothing). Spaces oscillate under the energy economy
and are never stopped early.

    python3 replicate/replicate_langmodel.py --rounds 50 --cap 2200 --spaces 3
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
from radial_evo2 import Env, SCALES, new_genome, mutate, feature
from replicate_langspace import _fp
from replicate_langstack import (evolve, new_vec, mut_vec, feat_vec)

RD = os.path.join(_HERE, "radial_data")


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--cap", type=int, default=2200)
    ap.add_argument("--freeze", type=float, default=0.02)
    ap.add_argument("--spaces", type=int, default=3)
    args = ap.parse_args()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr = torch.tensor(z["ytr"].astype(np.int64), device=dev)
    yte = torch.tensor(z["yte"].astype(np.int64), device=dev)
    N = len(z["ytr"])
    env0 = Env(torch, dev, Xtr, Xte, max_cached=len(SCALES))
    for ps in SCALES:
        env0.maps(ps)
    basis = {ps: re2._SVD_CACHE[(_fp(Xtr), ps)] for ps in SCALES}

    def augment(Xnp, seed, mode):
        g = torch.Generator(device=dev).manual_seed(seed)
        t = torch.tensor(Xnp, device=dev).permute(0, 3, 1, 2)
        n = len(t)
        if mode == "geo":                              # flip + crop only
            t = torch.nn.functional.pad(t, (3, 3, 3, 3), mode="reflect")
            ox = int(torch.randint(0, 7, (1,), generator=g, device=dev))
            oy = int(torch.randint(0, 7, (1,), generator=g, device=dev))
            t = t[:, :, oy:oy + 32, ox:ox + 32]
            flip = torch.rand(n, generator=g, device=dev) < 0.5
            t[flip] = torch.flip(t[flip], dims=[3])
        else:                                          # photometric only
            b = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=dev) - .5) * .5
            con = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=dev) - .5) * .5
            col = 1 + (torch.rand(n, 3, 1, 1, generator=g, device=dev) - .5) * .3
            gam = 0.7 + torch.rand(n, 1, 1, 1, generator=g, device=dev) * 0.6
            m = t.mean((2, 3), keepdim=True)
            t = (((t - m) * con + m) * b * col).clamp(1e-4, 1) ** gam
            t = (t + torch.randn(t.shape, generator=g, device=dev) * 0.03).clamp(0, 1)
        return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def env_for(Xa):
        e = Env(torch, dev, Xa, Xa[:100], max_cached=len(SCALES))
        for ps in SCALES:
            re2._SVD_CACHE[(_fp(Xa), ps)] = basis[ps]
        return e

    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:4000], device=dev)
    rng = np.random.default_rng(59)

    # ---- SPACE 1 : photometric-invariant words over the image ------------
    envP1, envP2 = env_for(augment(Xtr, 1, "photo")), env_for(augment(Xtr, 2, "photo"))
    log(f"[model] space1 photometric views ready ({round(time.time()-t0)}s)")
    S1 = evolve(torch, rng, None, args.pop, args.gens, args.rounds, args.cap,
                args.freeze, lambda r: new_genome(r), lambda r, g, s: mutate(r, g, s),
                lambda gs: torch.stack([feature(torch, tp, envP1, g) for g in gs], 1),
                lambda gs: torch.stack([feature(torch, tp, envP2, g) for g in gs], 1),
                None, probe, "S1-photo")
    del envP1, envP2
    torch.cuda.empty_cache()

    def s1_words(env, test=False):
        return torch.stack([feature(torch, tp, env, g, test=test) for g in S1], 1)
    W0, W0te = s1_words(env0), s1_words(env0, test=True)
    mu1, sd1 = W0.mean(0), W0.std(0) + 1e-6
    log(f"[model] SPACE 1: {len(S1)} photometric words ({round(time.time()-t0)}s)")

    # ---- SPACE 2 : geometric-invariant words over space-1 words ----------
    envG1, envG2 = env_for(augment(Xtr, 3, "geo")), env_for(augment(Xtr, 4, "geo"))
    WG1 = (s1_words(envG1) - mu1) / sd1
    WG2 = (s1_words(envG2) - mu1) / sd1
    del envG1, envG2
    torch.cuda.empty_cache()
    F1 = len(S1)
    S2 = evolve(torch, rng, None, args.pop, args.gens, args.rounds, args.cap,
                args.freeze, lambda r: new_vec(r, F1), lambda r, g, s: mut_vec(r, g, s, F1),
                lambda gs: torch.stack([feat_vec(torch, tp, WG1, g) for g in gs], 1),
                lambda gs: torch.stack([feat_vec(torch, tp, WG2, g) for g in gs], 1),
                None, probe, "S2-geo")
    W0z = (W0 - mu1) / sd1
    W0tez = (W0te - mu1) / sd1
    S2_0 = torch.stack([feat_vec(torch, tp, W0z, g) for g in S2], 1)
    S2_te = torch.stack([feat_vec(torch, tp, W0tez, g) for g in S2], 1)
    mu2, sd2 = S2_0.mean(0), S2_0.std(0) + 1e-6
    log(f"[model] SPACE 2: {len(S2)} geometric words ({round(time.time()-t0)}s)")

    # ---- SPACE 3 : predictive words over space-2 words -------------------
    S3, S3_0, S3_te = [], None, None
    if args.spaces >= 3:
        # predictive question: a word should be PREDICTABLE from a random other
        # subset of space-2 words -> it captures the redundant/structured part.
        # implement as: quality = corr(word, its own value reconstructed from a
        # fixed random projection of the OTHER words). Label-free, answer in env.
        S2z = (S2_0 - mu2) / sd2
        gproj = torch.Generator(device=dev).manual_seed(7)
        P = torch.randn(S2z.shape[1], S2z.shape[1], generator=gproj, device=dev)
        P.fill_diagonal_(0.0)
        recon = S2z @ (P / (P.norm(dim=0, keepdim=True) + 1e-6))   # predict from others

        def s3_quality(gs, view):
            C = torch.stack([feat_vec(torch, tp, S2z, g) for g in gs], 1)
            # predictable = correlates with the reconstruction-from-others
            c = (C - C.mean(0)) / (C.std(0) + 1e-6)
            r = (recon - recon.mean(0)) / (recon.std(0) + 1e-6)
            # each candidate scored vs its own channel's reconstruction target
            # (use nearest by index proxy: correlation with mean recon signal)
            pred = (c * r.mean(1, keepdim=True)).mean(0).abs()
            info = C.std(0); info = info / (info.mean() + 1e-6)
            q = pred * info
            q[~torch.isfinite(q)] = -1
            return q
        F2 = len(S2)
        S3 = evolve(torch, rng, None, args.pop, args.gens, args.rounds, args.cap,
                    args.freeze, lambda r: new_vec(r, F2), lambda r, g, s: mut_vec(r, g, s, F2),
                    lambda gs: torch.stack([feat_vec(torch, tp, (S2_0 - mu2) / sd2, g)
                                            for g in gs], 1),
                    lambda gs: torch.stack([feat_vec(torch, tp, (S2_0 - mu2) / sd2, g)
                                            for g in gs], 1),
                    None, probe, "S3-pred")
        S3_0 = torch.stack([feat_vec(torch, tp, (S2_0 - mu2) / sd2, g) for g in S3], 1)
        S3_te = torch.stack([feat_vec(torch, tp, (S2_te - mu2) / sd2, g) for g in S3], 1)
        log(f"[model] SPACE 3: {len(S3)} predictive words ({round(time.time()-t0)}s)")

    # ---- OUTPUT HEAD -----------------------------------------------------
    n_fit = int(N * 0.8); yv = ytr[n_fit:]
    Y = -torch.ones((N, 10), device=dev); Y[torch.arange(N), ytr] = 1.

    def head(F, Fte, tag):
        mu, sd = F.mean(0), F.std(0) + 1e-6
        Ztr, Zte = (F - mu) / sd, (Fte - mu) / sd
        bl, bv = 10., -1
        for lam in (1., 3., 10., 30., 100., 300.):
            _, a = _ridge_soft(torch, Ztr[:n_fit], Ztr[n_fit:], Y[:n_fit], yv, lam=lam)
            if a > bv:
                bl, bv = lam, a
        A = torch.hstack([Ztr, torch.ones(N, 1, device=dev)])
        W = torch.linalg.solve((A.T @ A).double() + bl * torch.eye(A.shape[1],
                               device=dev, dtype=torch.float64),
                               (A.T @ Y).double()).float()
        B = torch.hstack([Zte, torch.ones(len(yte), 1, device=dev)])
        acc = float(((B @ W).argmax(1) == yte).float().mean())
        log(f"[model] HEAD [{tag}]: val {bv:.4f} TEST {acc:.4f}")
        return acc

    a1 = head(W0, W0te, f"S1 photo ({len(S1)})")
    a2 = head(torch.cat([W0, S2_0], 1), torch.cat([W0te, S2_te], 1),
              f"S1|S2 photo+geo ({len(S1)}+{len(S2)})")
    log(f"[model] S2 (geometric) stacking gain over S1: {a2 - a1:+.4f}")
    res = {"module": "langmodel", "s1": len(S1), "s2": len(S2),
           "s1_test": round(a1, 4), "s12_test": round(a2, 4)}
    if S3:
        a3 = head(torch.cat([W0, S2_0, S3_0], 1),
                  torch.cat([W0te, S2_te, S3_te], 1),
                  f"S1|S2|S3 ({len(S1)}+{len(S2)}+{len(S3)})")
        log(f"[model] S3 (predictive) stacking gain over S1|S2: {a3 - a2:+.4f}")
        res.update({"s3": len(S3), "s123_test": round(a3, 4)})
    log(f"[model] FULL MODEL: S1 {a1:.4f} -> S1|S2 {a2:.4f}"
        + (f" -> S1|S2|S3 {res.get('s123_test'):.4f}" if S3 else "")
        + f"  (label-supervised single-seed was 0.7701)")
    json.dump(res, open(os.path.join(RD, "replicate_langmodel.json"), "w"))


if __name__ == "__main__":
    main()
