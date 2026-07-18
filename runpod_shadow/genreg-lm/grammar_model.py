"""grammar_model.py - the GRAMMAR SPECIALIST (user's architecture call).

One model, one question: IS THIS WORD ORDER PROPER? Not next-word, not
topic - just grammar, exactly like the topic specialist (module 32) knows
only topics. The union happens at decode time.

Substrate: the DIRECTIONAL RS spaces (embed_rs_prev + embed_rs_next,
5000 x 64 each -> 128-d syntactic role vector per word) - built 2026-07-16
precisely because grammatical role lives in what precedes/follows a word.
Module 30 already proved temporal genomes are the ONLY earner on
real-vs-shuffled and that semantic embed_rs capped the signal at 0.62;
this is the same question on the syntactic substrate.

Task: real 10-word sequences vs LOCALLY-shuffled copies (1-4 adjacent
swaps, mixed - position stats preserved, transitions broken; the honest
control from module 30). Temporal genomes only, genome-only head, saved
as a frozen artifact (kid_grammar_model.json) with everything needed to
score arbitrary text.

  python lm/grammar_model.py [--smoke]
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401

import json
import os
import sys
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
import radial_stack as rk
import radial_temporal as rt
from radial_lm import _clean

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_ROOT, "radial_data")
T = 10                                    # words per sequence


def build_stream(n_train, n_test):
    """Real T-word sequences as syntactic role vectors (prev|next)."""
    zp = np.load(os.path.join(RD, "embed_rs_prev.npz"), allow_pickle=True)
    zn = np.load(os.path.join(RD, "embed_rs_next.npz"), allow_pickle=True)
    vocab = {str(w): i for i, w in enumerate(zp["vocab"])}
    E = np.concatenate([zp["feat"], zn["feat"]], 1).astype(np.float32)
    D = E.shape[1]
    with open(os.path.join(_ROOT, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        f.seek(20_000_000)
        toks = _clean(f.read(28_000_000)).split()
    rng = np.random.default_rng(0)
    n = n_train + n_test
    X = np.zeros((n, T, D), np.float32)
    got = 0
    for p in rng.permutation(len(toks) - T):
        ids = [vocab.get(w) for w in toks[p:p + T]]
        if all(i is not None for i in ids):
            for t, i in enumerate(ids):
                X[got, t] = E[i]
            got += 1
            if got == n:
                break
    return X[:got], vocab, E, D


def local_shuffle(rng, X, n_swaps):
    X = X.copy()
    for i in range(len(X)):
        for _ in range(n_swaps):
            j = rng.integers(0, T - 1)
            X[i, j], X[i, j + 1] = X[i, j + 1].copy(), X[i, j].copy()
    return X


def run(smoke=False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    def log(m, v=True):
        print(m, flush=True)

    n_train, n_test = (3000, 800) if smoke else (40000, 8000)
    pop, gens, rounds = (48, 8, 20) if smoke else (96, 12, 200)
    Xr, vocab, E, D = build_stream(n_train, n_test)
    n_train = min(n_train, len(Xr) - n_test)
    log(f"[gram] {len(Xr)} real sequences, T={T}, D={D} (prev|next role)")

    # z-score per channel over (N,T) on REAL train (shuffle only reorders
    # time, so real and shuffled share channel stats - no leak)
    flat = Xr[:n_train].reshape(-1, D)
    cmu, csd = flat.mean(0), flat.std(0) + 1e-6
    Xz = np.clip((Xr - cmu) / csd, -8, 8).astype(np.float32)

    def make_pairs(X0, seed):
        r = np.random.default_rng(seed)
        n_sw = r.integers(1, 5, len(X0))      # 1-4 adjacent swaps, mixed
        Xs = X0.copy()
        for i in range(len(X0)):
            for _ in range(n_sw[i]):
                j = r.integers(0, T - 1)
                Xs[i, j], Xs[i, j + 1] = Xs[i, j + 1].copy(), Xs[i, j].copy()
        X = np.concatenate([X0, Xs])
        y = np.concatenate([np.ones(len(X0)), np.zeros(len(X0))]).astype(np.int64)
        perm = r.permutation(len(X))
        return X[perm], y[perm]

    Xtr, ytr = make_pairs(Xz[:n_train], 1)
    Xte, yte = make_pairs(Xz[n_train:n_train + n_test], 2)
    Ntr, Nte = len(ytr), len(yte)
    F_tr = torch.tensor(Xtr, device=dev)
    F_te = torch.tensor(Xte, device=dev)
    log(f"[gram] pairs: Ntr={Ntr} Nte={Nte} (chance 0.5)")

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 2), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, 2), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    # anchor: ridge on the position-concat (grammar should NOT be linear)
    cat_tr = F_tr.reshape(Ntr, T * D)
    _, anchor = _ridge_soft(torch, cat_tr[:n_fit], cat_tr[n_fit:], Yf, yv)
    log(f"[gram] linear anchor (concat ridge): val {anchor:.4f}")

    new_fn = lambda r: rt.new_temporal(r, D, T)
    mut_fn = lambda r, g, sc: rt.mutate_temporal(r, g, sc, D, T)
    feat_tr = lambda g: rt._finite(torch, rt.temporal_feat(torch, F_tr, g))
    feat_te = lambda g: rt._finite(torch, rt.temporal_feat(torch, F_te, g))
    base0 = torch.zeros((Ntr, 0), device=dev)
    frozen, fcols = rk._evolve_space(torch, rng, pop, gens, rounds, n_fit,
                                     Yf, yv, base0, new_fn, mut_fn,
                                     feat_tr, log, True)
    if not frozen:
        log("[gram] earned nothing - no model saved")
        return
    Ftr = torch.stack(fcols, 1)
    Fte = torch.stack([feat_te(g) for g in frozen], 1)
    fmu, fsd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = ((Ftr - fmu) / fsd).clamp(-8, 8)
    Fte = ((Fte - fmu) / fsd).clamp(-8, 8)

    best = (1.0, -1.0)
    for lam in (1.0, 3.0, 10.0):
        _, a = _ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf, yv, lam=lam)
        if a > best[1]:
            best = (lam, float(a))
    hm2 = Ftr.mean(0); hs2 = Ftr.std(0) + 1e-6
    A = torch.hstack([(Ftr - hm2) / hs2, torch.ones(Ntr, 1, device=dev)])
    G = (A.T @ A).double() + best[0] * torch.eye(Ftr.shape[1] + 1, device=dev,
                                                dtype=torch.float64)
    Wm = torch.linalg.solve(G, (A.T @ Yfull).double()).float()
    s = torch.hstack([(Fte - hm2) / hs2,
                      torch.ones(Nte, 1, device=dev)]) @ Wm
    test = round(float((s.argmax(1) == yte_t).float().mean()), 4)
    log(f"[gram] TEST {test} ({len(frozen)} temporal genomes, "
        f"anchor {anchor:.4f}, chance 0.5)")

    with open(os.path.join(RD, "kid_grammar_model.json"), "w") as f:
        json.dump({"genomes": frozen, "T": T, "D": D,
                   "cmu": cmu.tolist(), "csd": csd.tolist(),
                   "fmu": fmu.tolist(), "fsd": fsd.tolist(),
                   "head_mu": hm2.tolist(), "head_sd": hs2.tolist(),
                   "head_W": Wm.tolist(), "lam": best[0],
                   "val_acc": best[1], "test_acc": test,
                   "anchor": round(float(anchor), 4)}, f)
    res = {"test_acc": test, "val_acc": best[1], "n_genomes": len(frozen),
           "anchor": round(float(anchor), 4), "chance": 0.5,
           "Ntr": Ntr, "Nte": Nte, "T": T, "D": D,
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "grammar_model_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("GRAM RESULT: " + json.dumps(res))
    print("GRAM DONE", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
