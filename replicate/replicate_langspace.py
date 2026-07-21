"""replicate_langspace.py — module 57: the INTERNAL LANGUAGE. Label-free
spaces, output head only at the end.

The architecture the user has been pointing at the whole time:
  image -> space1 -> space1 words -> space2 -> space2 words -> OUTPUT HEAD
Only the output head ever sees labels. Every space before it evolves its own
internal language for features (contrastive self-supervision), no labels
leaking in. My earlier stacks let labels into every space (make_scorer scores
each space on classification), which forces every space to re-speak the
classifier's question, so deeper spaces are redundant by construction and the
stack correctly reports it can't earn. Take the labels out of the spaces and
each one is free to invent genuinely new words.

A word = a genome's output. A GOOD word (label-free fitness):
  - STABLE: two augmentations of the same image get the same code
    (invariance = the word names the feature, not the nuisance)
  - INFORMATIVE + DISTINCT: high variance across images, decorrelated from the
    words already in the bank

The space is evolved under the REAL energy economy (starvation turnover,
tournament selection, emergent cap) — the machinery, not a greedy loop. Then
a single ridge output head translates the internal words to English and we
measure. Competitive with the label-supervised 0.77? Then stack space 2.

    python3 replicate/replicate_langspace.py --pop 96 --gens 12 --rounds 60
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
import radial_evo2 as re2
from radial_evo2 import Env, SCALES, new_genome, mutate, feature
import radial_stack as rk

RD = os.path.join(_HERE, "radial_data")
# energy economy (from radial_stack), scale-free turnover dynamics
E_DECAY, OUT_COST, RESTORE, E_GAIN, E_FLOOR, E_MAX = 0.75, 0.05, 0.04, 6.0, 0.2, 1.5
MIN_TURNOVER = 12


def log(m):
    print(m, flush=True)


def _fp(X):
    return (X.shape, float(X[:64].sum()), float(X[-64:].sum()), float(X.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--freeze", type=float, default=0.02,
                    help="min word quality (invariance*info) to freeze")
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

    # clean env (frozen patch-PCA basis) + two augmented envs sharing that basis
    env0 = Env(torch, dev, Xtr, Xte, max_cached=len(SCALES))
    for ps in SCALES:
        env0.maps(ps)
    basis = {ps: re2._SVD_CACHE[(_fp(Xtr), ps)] for ps in SCALES}
    gen = torch.Generator(device=dev).manual_seed(0)

    def augment(Xnp, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        t = torch.tensor(Xnp, device=dev).permute(0, 3, 1, 2)
        n = len(t)
        t = torch.nn.functional.pad(t, (3, 3, 3, 3), mode="reflect")
        ox = int(torch.randint(0, 7, (1,), generator=g, device=dev))
        oy = int(torch.randint(0, 7, (1,), generator=g, device=dev))
        t = t[:, :, oy:oy + 32, ox:ox + 32]
        flip = torch.rand(n, generator=g, device=dev) < 0.5
        t[flip] = torch.flip(t[flip], dims=[3])
        b = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=dev) - .5) * .4
        col = 1 + (torch.rand(n, 3, 1, 1, generator=g, device=dev) - .5) * .3
        t = (t * b * col).clamp(0, 1)
        return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def env_for(Xa):
        e = Env(torch, dev, Xa, Xa[:100], max_cached=len(SCALES))
        for ps in SCALES:
            re2._SVD_CACHE[(_fp(Xa), ps)] = basis[ps]
        return e
    # two fixed augmented views of the TRAIN set (the contrastive pair)
    envA = env_for(augment(Xtr, 1))
    envB = env_for(augment(Xtr, 2))
    log(f"[lang] frozen basis + 2 augmented train views built "
        f"({round(time.time()-t0)}s, {dev})")

    def words(env, pop, test=False):
        return torch.stack([feature(torch, tp, env, g, test=test) for g in pop], 1)

    def quality(CA, CB):
        """label-free word quality per genome: invariance(view A vs B) x info.
        CA,CB: (N, K) genome outputs on the two augmented views."""
        a = (CA - CA.mean(0)) / (CA.std(0) + 1e-6)
        b = (CB - CB.mean(0)) / (CB.std(0) + 1e-6)
        inv = (a * b).mean(0).clamp(min=0)           # cross-view correlation
        info = CA.std(0)                             # spread across images
        info = info / (info.mean() + 1e-6)
        q = inv * info
        q[~torch.isfinite(q)] = -1
        return q, inv

    # ---- evolve the internal-language space under the energy economy -------
    rng = np.random.default_rng(57)
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:4000], device=dev)
    frozen, fsig = [], []
    for rnd in range(args.rounds):
        if len(frozen) >= args.cap:
            break
        pop = [new_genome(rng) for _ in range(args.pop)]
        scales = np.full(args.pop, 0.25)

        def fit_pop(gs):
            CA = words(envA, gs); CB = words(envB, gs)
            q, inv = quality(CA, CB)
            return q.cpu().numpy(), CA
        fits, CA = fit_pop(pop)
        energy = np.ones(args.pop)
        starved_total = 0
        for g_ in range(args.gens):
            valid = fits > -1e8
            energy = np.clip(energy * E_DECAY - OUT_COST + RESTORE * valid
                             + E_GAIN * np.maximum(fits - np.median(fits), 0.0),
                             0.0, E_MAX)
            starved = energy < E_FLOOR
            starved_total += int(starved.sum())
            dead = list(np.where(starved)[0])
            if len(dead) < MIN_TURNOVER:
                living = [i for i in np.argsort(fits) if i not in set(dead)]
                dead += living[:MIN_TURNOVER - len(dead)]
            alive = [i for i in range(args.pop) if i not in set(dead)] or \
                    list(np.argsort(fits)[::-1][:4])
            n_fresh = max(1, len(dead) // 4)
            kids, ksc = [], []
            for k in range(len(dead)):
                if k < n_fresh:
                    kids.append(new_genome(rng)); ksc.append(0.25)
                else:
                    cand = rng.choice(alive, 3)
                    pi = cand[int(np.argmax(fits[cand]))]
                    sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]),
                                       0.03, 0.6))
                    kids.append(mutate(rng, pop[pi], sc)); ksc.append(sc)
            kf, kCA = fit_pop(kids)
            for slot, k in zip(dead, range(len(kids))):
                pop[slot] = kids[k]; scales[slot] = ksc[k]; fits[slot] = kf[k]
                energy[slot] = 1.0
        # freeze every high-quality decorrelated WORD (no labels used)
        CAf = words(envA, pop)
        order = np.argsort(fits)[::-1]
        added = 0
        for i in order:
            if fits[i] < args.freeze:
                break
            s = CAf[probe, i] - CAf[probe, i].mean()
            s = s / (s.norm() + 1e-8)
            if any(float(torch.abs(s @ t)) > 0.9 for t in fsig[-120:]):
                continue
            frozen.append(pop[i]); fsig.append(s); added += 1
        spg = round(starved_total / max(args.gens, 1), 1)
        log(f"[lang] round {rnd:3d}  +{added} (lang {len(frozen)})  "
            f"starved/gen {spg}  best-q {float(fits.max()):.3f} "
            f"({round(time.time()-t0)}s)")
        if added == 0 and rnd > 3:
            log("[lang] no new words admitted — space full"); break

    # ---- OUTPUT HEAD at the end: translate internal words -> English -------
    n_fit = int(N * 0.8)
    yv = ytr[n_fit:]
    Y = -torch.ones((N, 10), device=dev); Y[torch.arange(N), ytr] = 1.
    Wtr = words(env0, frozen)                       # clean-view words, train
    Wte = words(env0, frozen, test=True)
    mu, sd = Wtr.mean(0), Wtr.std(0) + 1e-6
    Ztr, Zte = (Wtr - mu) / sd, (Wte - mu) / sd
    bl, bv = 10.0, -1
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
    log(f"[lang] OUTPUT HEAD on {len(frozen)} internal words (LABEL-FREE space): "
        f"val {bv:.4f} TEST {acc:.4f}  (label-supervised R0 was 0.7701)")


if __name__ == "__main__":
    main()
