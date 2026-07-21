"""replicate_langstack.py — module 58: the 2-space LABEL-FREE stack.

The decisive test of the internal-language architecture. Space 1 evolves words
over the image (patch-PCA), label-free (contrastive: view-stable x informative
x diverse). Space 2 evolves higher words over SPACE 1's words, same label-free
question. Only the output head at the very end sees labels.

The question: does space 2 EARN over space 1, where label-supervised depth
(m56) earned exactly 0? If yes, depth was never dead -- it was being asked the
classifier's question in every space. Each space here is asked an EASY,
answerable, label-free question (its answer is present in the previous space's
output, and one RS can solve it), so the spaces compose instead of colliding.

    python3 replicate/replicate_langstack.py --s1_rounds 40 --s2_rounds 40
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
from radial_evo2 import (Env, SCALES, new_genome, mutate, feature,
                         _PRIMS, _OPS)
from replicate_langspace import _fp
tp_prims = ["id", "abs", "relu", "tanh", "gauss", "sq", "soft", "sin"]

RD = os.path.join(_HERE, "radial_data")
E_DECAY, OUT_COST, RESTORE, E_GAIN, E_FLOOR, E_MAX = 0.75, 0.05, 0.04, 6.0, 0.2, 1.5
MIN_TURNOVER = 12


def log(m):
    print(m, flush=True)


# --- vector grammar for space 2+ : a word is a nonlinear combo of prev words -
def new_vec(rng, F):
    order = 2 if rng.random() < 0.6 else 3
    return {"terms": [{"c": int(rng.integers(F)),
                       "prog": [(int(rng.integers(len(_PRIMS))),
                                 float(rng.uniform(0.5, 2.5)),
                                 float(rng.uniform(-1, 1)))
                                for _ in range(1 if rng.random() < 0.7 else 2)]}
                      for _ in range(order)],
            "op": int(rng.integers(len(_OPS)))}


def mut_vec(rng, g, sc, F):
    c = json.loads(json.dumps(g))
    for t in c["terms"]:
        if rng.random() < 0.15:
            t["c"] = int(rng.integers(F))
        prog = [list(s) for s in t["prog"]]
        for s in prog:
            if rng.random() < 0.1:
                s[0] = int(rng.integers(len(_PRIMS)))
            s[1] = float(np.clip(s[1] + rng.normal(0, sc), 0.1, 4.0))
            s[2] = float(np.clip(s[2] + rng.normal(0, sc), -2.0, 2.0))
        t["prog"] = [tuple(s) for s in prog]
    if rng.random() < 0.1:
        c["op"] = int(rng.integers(len(_OPS)))
    return c


def feat_vec(torch, tp, Wz, g):
    z = None
    for t in g["terms"]:
        v = Wz[:, t["c"] % Wz.shape[1]]
        for prim, a, b in t["prog"]:
            v = tp[tp_prims[prim]](a * v + b)
        if z is None:
            z = v
        else:
            op = _OPS[g["op"]]
            z = (z * v if op == "mult" else torch.minimum(z, v)
                 if op == "min" else torch.abs(z - v))
    return z


def quality(torch, CA, CB):
    a = (CA - CA.mean(0)) / (CA.std(0) + 1e-6)
    b = (CB - CB.mean(0)) / (CB.std(0) + 1e-6)
    inv = (a * b).mean(0).clamp(min=0)
    info = CA.std(0); info = info / (info.mean() + 1e-6)
    q = inv * info
    q[~torch.isfinite(q)] = -1
    return q


def evolve(torch, rng, pop, args_pop, gens, rounds, cap, freeze, new_fn,
           mut_fn, feat_A, feat_B, feat_clean, probe, tag):
    """Generic label-free contrastive space under the energy economy.
    feat_A/B(genlist)->(N,K) on the two views; feat_clean for the head later."""
    frozen, fsig = [], []
    for rnd in range(rounds):
        if len(frozen) >= cap:
            break
        pop = [new_fn(rng) for _ in range(args_pop)]
        scales = np.full(args_pop, 0.25)

        def fit_pop(gs):
            CA = feat_A(gs); CB = feat_B(gs)
            return quality(torch, CA, CB).cpu().numpy()
        fits = fit_pop(pop)
        energy = np.ones(args_pop); starved_total = 0
        for _g in range(gens):
            valid = fits > -1e8
            energy = np.clip(energy * E_DECAY - OUT_COST + RESTORE * valid
                             + E_GAIN * np.maximum(fits - np.median(fits), 0.),
                             0., E_MAX)
            starved = energy < E_FLOOR
            starved_total += int(starved.sum())
            dead = list(np.where(starved)[0])
            if len(dead) < MIN_TURNOVER:
                living = [i for i in np.argsort(fits) if i not in set(dead)]
                dead += living[:MIN_TURNOVER - len(dead)]
            alive = [i for i in range(args_pop) if i not in set(dead)] or \
                    list(np.argsort(fits)[::-1][:4])
            n_fresh = max(1, len(dead) // 4)
            kids, ksc = [], []
            for k in range(len(dead)):
                if k < n_fresh:
                    kids.append(new_fn(rng)); ksc.append(0.25)
                else:
                    cand = rng.choice(alive, 3)
                    pi = cand[int(np.argmax(fits[cand]))]
                    sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]),
                                       0.03, 0.6))
                    kids.append(mut_fn(rng, pop[pi], sc)); ksc.append(sc)
            kf = fit_pop(kids)
            for slot, k in zip(dead, range(len(kids))):
                pop[slot] = kids[k]; scales[slot] = ksc[k]; fits[slot] = kf[k]
                energy[slot] = 1.0
        CAf = feat_A(pop)
        order = np.argsort(fits)[::-1]; added = 0
        for i in order:
            if fits[i] < freeze:
                break
            s = CAf[probe, i] - CAf[probe, i].mean()
            s = s / (s.norm() + 1e-8)
            if any(float(torch.abs(s @ t)) > 0.9 for t in fsig[-120:]):
                continue
            frozen.append(pop[i]); fsig.append(s); added += 1
        spg = round(starved_total / max(gens, 1), 1)
        log(f"[{tag}] round {rnd:3d}  +{added} ({tag} {len(frozen)})  "
            f"starved/gen {spg}  best-q {float(fits.max()):.2f}")
        # NO early stop: these spaces oscillate (added dips to 1-4, taps 0) but
        # never flatline -- there is always another word. Run the full budget.
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--s1_rounds", type=int, default=40)
    ap.add_argument("--s2_rounds", type=int, default=40)
    ap.add_argument("--s1_cap", type=int, default=1200)
    ap.add_argument("--s2_cap", type=int, default=1200)
    ap.add_argument("--freeze", type=float, default=0.02)
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

    def augment(Xnp, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        t = torch.tensor(Xnp, device=dev).permute(0, 3, 1, 2); n = len(t)
        t = torch.nn.functional.pad(t, (3, 3, 3, 3), mode="reflect")
        ox = int(torch.randint(0, 7, (1,), generator=g, device=dev))
        oy = int(torch.randint(0, 7, (1,), generator=g, device=dev))
        t = t[:, :, oy:oy + 32, ox:ox + 32]
        flip = torch.rand(n, generator=g, device=dev) < 0.5
        t[flip] = torch.flip(t[flip], dims=[3])
        b = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=dev) - .5) * .4
        col = 1 + (torch.rand(n, 3, 1, 1, generator=g, device=dev) - .5) * .3
        return (t * b * col).clamp(0, 1).permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def env_for(Xa):
        e = Env(torch, dev, Xa, Xa[:100], max_cached=len(SCALES))
        for ps in SCALES:
            re2._SVD_CACHE[(_fp(Xa), ps)] = basis[ps]
        return e
    envA, envB = env_for(augment(Xtr, 1)), env_for(augment(Xtr, 2))
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:4000], device=dev)
    rng = np.random.default_rng(58)
    log(f"[stack] envs ready ({round(time.time()-t0)}s, {dev})")

    # ---- SPACE 1 : words over the image, label-free ----------------------
    S1 = evolve(torch, rng, None, args.pop, args.gens, args.s1_rounds,
                args.s1_cap, args.freeze,
                lambda r: new_genome(r), lambda r, g, s: mutate(r, g, s),
                lambda gs: torch.stack([feature(torch, tp, envA, g) for g in gs], 1),
                lambda gs: torch.stack([feature(torch, tp, envB, g) for g in gs], 1),
                None, probe, "S1")

    def s1_words(env, test=False):
        return torch.stack([feature(torch, tp, env, g, test=test) for g in S1], 1)
    WA, WB = s1_words(envA), s1_words(envB)
    W0, W0te = s1_words(env0), s1_words(env0, test=True)
    zc = lambda W, mu, sd: (W - mu) / sd
    muA, sdA = WA.mean(0), WA.std(0) + 1e-6
    WAz, WBz = zc(WA, muA, sdA), zc(WB, muA, sdA)
    log(f"[stack] SPACE 1 done: {len(S1)} words ({round(time.time()-t0)}s)")

    # ---- SPACE 2 : words over space-1 words, label-free ------------------
    F1 = len(S1)
    S2 = evolve(torch, rng, None, args.pop, args.gens, args.s2_rounds,
                args.s2_cap, args.freeze,
                lambda r: new_vec(r, F1), lambda r, g, s: mut_vec(r, g, s, F1),
                lambda gs: torch.stack([feat_vec(torch, tp, WAz, g) for g in gs], 1),
                lambda gs: torch.stack([feat_vec(torch, tp, WBz, g) for g in gs], 1),
                None, probe, "S2")
    log(f"[stack] SPACE 2 done: {len(S2)} words ({round(time.time()-t0)}s)")

    # ---- OUTPUT HEAD : translate internal words -> English ---------------
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
        log(f"[stack] HEAD [{tag}]: val {bv:.4f} TEST {acc:.4f}")
        return acc
    S2_0 = torch.stack([feat_vec(torch, tp, zc(W0, muA, sdA), g) for g in S2], 1)
    S2_te = torch.stack([feat_vec(torch, tp, zc(W0te, muA, sdA), g) for g in S2], 1)
    a1 = head(W0, W0te, f"SPACE 1 only ({len(S1)})")
    aj = head(torch.cat([W0, S2_0], 1), torch.cat([W0te, S2_te], 1),
              f"SPACE 1 | SPACE 2 ({len(S1)}+{len(S2)})")
    log(f"[stack] STACKING GAIN (label-free depth): {aj - a1:+.4f} "
        f"(S1 {a1:.4f} -> S1|S2 {aj:.4f}); label-supervised depth earned 0")
    json.dump({"module": "langstack", "s1": len(S1), "s2": len(S2),
               "s1_test": round(a1, 4), "joint_test": round(aj, 4)},
              open(os.path.join(RD, "replicate_langstack.json"), "w"))


if __name__ == "__main__":
    main()
