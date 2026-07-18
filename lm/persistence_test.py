"""persistence_test.py - the PERSISTENCE operator (user's definition).

Persistence = a signal that REPEATS over time; you ACCUMULATE it; the
accumulated signal is the persistence, and it is what the static space
consumes. Operationally: a per-frame detector applied at every view, its
response SUMMED across the stream. A pattern that recurs (the letter's real
structure, invariant across viewpoints) accumulates; independent per-view
corruption averages out. Accumulation is ORDER-INVARIANT by design - this is
NOT the transition/grammar operator tested before.

Testbed matching the dog-across-angles / text-across-viewpoints examples: the
SAME letter rendered K times under heavy, independent per-view corruption
(noise + occlusion + size/jitter), so a single view is unreliable. Arms, all
evolving the same R0 detectors, same budget:
  SINGLE     - feature from ONE corrupted view (no accumulation)
  PIXELMEAN  - detector on the pixel-averaged view (accumulate in PIXEL space)
  ACCUM      - detector response ACCUMULATED over the K views (persistence:
               accumulate in FEATURE space - the operator under test)
If ACCUM >> SINGLE (and beats PIXELMEAN), accumulating the repeating signal
recovers identity a single viewpoint loses - persistence works.

  python persistence_test.py [--smoke]
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
import os
import sys
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, new_genome, mutate
import radial_stack as rk
from radial_kid import render_letter, LETTERS

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")


def make_views(n, K, rng):
    """n letters, each rendered K times under heavy INDEPENDENT per-view
    corruption. The letter is the repeating signal; the corruption is not."""
    y = rng.integers(0, 26, n)
    X = np.zeros((n, K, 32, 32), np.float32)
    for i in range(n):
        for k in range(K):
            im = render_letter(LETTERS[y[i]], rng, size=int(rng.integers(12, 28)))
            im = im + rng.normal(0, 0.45, im.shape).astype(np.float32)
            ox, oy = int(rng.integers(0, 24)), int(rng.integers(0, 24))
            im[oy:oy + 9, ox:ox + 9] = 0.0          # random occlusion patch
            X[i, k] = np.clip(im, 0, 1)
    X = np.repeat(X[..., None], 3, axis=4)           # (n,K,32,32,3)
    return X, y.astype(np.int64)


def run(smoke=False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    rk.GRID = 8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    rng = np.random.default_rng(0)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    log_lines = []

    def log(m, v=True):
        log_lines.append(m); print(m, flush=True)

    K = 8
    Ntr, Nte = (1500, 400) if smoke else (8000, 2000)
    pop, gens, rounds = (48, 8, 20) if smoke else (64, 12, 60)
    log(f"[persist] rendering {Ntr}x{K} + {Nte}x{K} corrupted views "
        f"(noise .45 + 9x9 occlusion + size/jitter)")
    Xtr, ytr = make_views(Ntr, K, rng)
    Xte, yte = make_views(Nte, K, rng)

    # three feature environments over the SAME corrupted data
    envK = Env(torch, dev, Xtr.reshape(Ntr * K, 32, 32, 3),
               Xte.reshape(Nte * K, 32, 32, 3), max_cached=6)   # all views
    env1 = Env(torch, dev, Xtr[:, 0], Xte[:, 0], max_cached=6)  # one view
    envM = Env(torch, dev, Xtr.mean(1), Xte.mean(1), max_cached=6)  # pixel-mean

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 26), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, 26), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    def readout(Ftr, Fte):
        best = 0.0
        for lam in (1.0, 3.0, 10.0):
            _, a = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
            best = max(best, a)
        return round(float(best), 4)

    def arm(feat_tr, feat_te, tag):
        base0 = torch.zeros((Ntr, 0), device=dev)
        new_fn = new_genome
        mut_fn = lambda r, g, sc: mutate(r, g, sc)
        frozen, fcols = rk._evolve_space(torch, rng, pop, gens, rounds, n_fit,
                                         Yf, yv, base0, new_fn, mut_fn,
                                         feat_tr, log, True)
        if not frozen:
            log(f"[{tag}] earned nothing")
            return {"test": 0.0, "genomes": 0}
        Ftr = torch.stack(fcols, 1)
        Fte = torch.stack([feat_te(g) for g in frozen], 1)
        acc = readout(Ftr, Fte)
        log(f"[{tag}] TEST {acc} ({len(frozen)} genomes)")
        return {"test": acc, "genomes": len(frozen)}

    def _fin(c):
        return torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e6, 1e6)

    res = {"K": K, "Ntr": Ntr, "Nte": Nte, "chance": round(1 / 26, 4)}
    log("=== SINGLE: one corrupted view, no accumulation ===")
    res["single"] = arm(lambda g: _fin(rk.feature_r0(torch, tp, env1, g)),
                        lambda g: _fin(rk.feature_r0(torch, tp, env1, g, test=True)),
                        "single")
    log("=== PIXELMEAN: accumulate in PIXEL space, then detect ===")
    res["pixelmean"] = arm(lambda g: _fin(rk.feature_r0(torch, tp, envM, g)),
                          lambda g: _fin(rk.feature_r0(torch, tp, envM, g, test=True)),
                          "pixelmean")
    log("=== ACCUM: detect per view, ACCUMULATE over K (persistence) ===")
    res["accum"] = arm(
        lambda g: _fin(rk.feature_r0(torch, tp, envK, g).view(Ntr, K).mean(1)),
        lambda g: _fin(rk.feature_r0(torch, tp, envK, g, test=True).view(Nte, K).mean(1)),
        "accum")

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "persistence_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("PERSISTENCE RESULT: " + json.dumps(res))
    s = res["single"]["test"]; p = res["pixelmean"]["test"]; a = res["accum"]["test"]
    log(f"VERDICT: single {s} | pixel-mean {p} | ACCUM(persistence) {a} "
        f"-> accumulation {'RECOVERS identity' if a > s else 'does NOT beat single view'}")
    print("PERSISTENCE DONE", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
