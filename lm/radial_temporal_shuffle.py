"""radial_temporal_shuffle.py - Fork B: the STRUCTURAL task.

Next-word is recency-dominated, which favours absolute-position (static)
composition and makes it a poor test of temporal composition. The
real-vs-shuffled discriminator is the opposite: the ONLY thing distinguishing
a real stream from a time-permuted copy of ITSELF is sequence order, so a
detector must model dependency structure. Same rows for real and shuffled
(identical content, only order differs) => content cannot leak; order is the
whole signal.

Runs the same 3 arms on the char stream, binary (real=1, shuffled=0):
  RAW static (concat, absolute position) vs TEMPORAL (relative-time) vs the
  linear anchor. If temporal beats static HERE, relative-time composition
  detects structure that position-tagged static cannot.

  python radial_temporal_shuffle.py --cache wf_char_stream.pt
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
import radial_stack as rk
import radial_temporal as rt

RD = rt.RD


def shuffle_time(torch, F):
    """Per-row random permutation of the time axis. (M,T,C)->(M,T,C)."""
    idx = torch.argsort(torch.rand(F.shape[0], F.shape[1], device=F.device), 1)
    return torch.gather(F, 1, idx.unsqueeze(-1).expand(-1, -1, F.shape[2]))


def local_shuffle(torch, F, n_swaps):
    """n_swaps random ADJACENT transpositions per row: each element moves at
    most a step or two, so absolute position (and its channel statistics) is
    nearly preserved, but LOCAL transitions t-1->t are broken. This kills the
    position-stats shortcut that made the full-scramble task linearly easy, so
    only a relative-time (transition) detector can win."""
    M, T, _ = F.shape
    F = F.clone()
    rows = torch.arange(M, device=F.device)
    for _ in range(n_swaps):
        i = torch.randint(0, T - 1, (M,), device=F.device)
        a = F[rows, i].clone()
        F[rows, i] = F[rows, i + 1]
        F[rows, i + 1] = a
    return F


def run(cache, local=0):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    rng = np.random.default_rng(0)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    log_lines = []

    def log(m, v=True):
        log_lines.append(m); print(m, flush=True)

    blob = torch.load(os.path.join(RD, cache), map_location=dev)
    F_tr = torch.stack([t.float().to(dev) for t in blob["tr"]], 1)  # (N,T,C)
    F_te = torch.stack([t.float().to(dev) for t in blob["te"]], 1)
    T, C = F_tr.shape[1], F_tr.shape[2]
    # z-score per channel over (N,T) BEFORE shuffling (shuffle only reorders
    # time, so real and shuffled share channel stats - no normalization leak)
    cmu = F_tr.reshape(-1, C).mean(0); csd = F_tr.reshape(-1, C).std(0) + 1e-6
    F_tr = ((F_tr - cmu) / csd).clamp(-8, 8)
    F_te = ((F_te - cmu) / csd).clamp(-8, 8)

    def perturb(F):
        return local_shuffle(torch, F, local) if local else shuffle_time(torch, F)

    def real_vs_shuf(F):
        X = torch.cat([F, perturb(F)], 0)
        y = torch.cat([torch.ones(len(F), device=dev),
                       torch.zeros(len(F), device=dev)]).long()
        perm = torch.randperm(len(X), device=dev)
        return X[perm].contiguous(), y[perm]

    Xtr, ytr = real_vs_shuf(F_tr)
    Xte, yte = real_vs_shuf(F_te)
    Ntr, Nte = Xtr.shape[0], Xte.shape[0]
    mode = f"LOCAL x{local} adjacent swaps" if local else "FULL scramble"
    log(f"[shuffle] real-vs-shuffled ({mode}): T={T} C={C} "
        f"Ntr={Ntr} Nte={Nte} (chance 0.5)")

    cat_tr = Xtr.reshape(Ntr, T * C)
    cat_te = Xte.reshape(Nte, T * C)
    smu, ssd = cat_tr.mean(0), cat_tr.std(0) + 1e-6
    cat_tr = ((cat_tr - smu) / ssd).clamp(-8, 8)
    cat_te = ((cat_te - smu) / ssd).clamp(-8, 8)

    n_fit = int(Ntr * 0.8)
    yv = ytr[n_fit:]
    yte_t = yte
    Yf = -torch.ones((n_fit, 2), device=dev)
    Yf[torch.arange(n_fit), ytr[:n_fit]] = 1.0
    Yfull = -torch.ones((Ntr, 2), device=dev)
    Yfull[torch.arange(Ntr), ytr] = 1.0

    pop, gens, rounds, spaces = 96, 12, 200, 3

    def acc(Ftr, Fte):
        a1, _ = rt.readout(torch, Ftr, Fte, Yfull, yte_t, Ntr, Nte, dev)
        return a1

    _, anchor = _ridge_soft(torch, cat_tr[:n_fit], cat_tr[n_fit:], Yf, yv)
    log(f"[shuffle] linear anchor (concat->ridge): val {anchor:.4f}")
    res = {"anchor": round(float(anchor), 4), "T": T, "C": C,
           "Ntr": Ntr, "Nte": Nte, "chance": 0.5}

    log("=== RAW: static genomes over the concat (absolute position) ===")
    Rtr, Rte = rt.stack_static(torch, tp, rng, cat_tr.clone(), cat_te.clone(),
                               n_fit, Yf, yv, log, pop, gens, rounds, spaces,
                               tag="raw")
    if Rtr is not None:
        res["raw"] = {"acc": acc(Rtr, Rte), "genomes": Rtr.shape[1]}
        log(f"[RAW] TEST acc {res['raw']['acc']} ({Rtr.shape[1]} genomes)")

    log("=== TEMPORAL: relative-time genomes, then static on frozen ===")
    C_ = C
    new_fn = lambda r: rt.new_temporal(r, C_, T)
    mut_fn = lambda r, g, sc: rt.mutate_temporal(r, g, sc, C_, T)
    feat_tr = lambda g: rt._finite(torch, rt.temporal_feat(torch, Xtr, g))
    feat_te = lambda g: rt._finite(torch, rt.temporal_feat(torch, Xte, g))
    base0 = torch.zeros((Ntr, 0), device=dev)
    frozenT, fcolsT = rk._evolve_space(torch, rng, pop, gens, rounds, n_fit,
                                       Yf, yv, base0, new_fn, mut_fn,
                                       feat_tr, log, True)
    log(f"[TEMPORAL space 0] froze {len(frozenT)} relative-time genomes")
    if frozenT:
        tf_tr = torch.stack(fcolsT, 1)
        tf_te = torch.stack([feat_te(g) for g in frozenT], 1)
        tmu, tsd = tf_tr.mean(0), tf_tr.std(0) + 1e-6
        tf_tr = ((tf_tr - tmu) / tsd).clamp(-8, 8)
        tf_te = ((tf_te - tmu) / tsd).clamp(-8, 8)
        res["temporal_only"] = {"acc": acc(tf_tr, tf_te), "genomes": tf_tr.shape[1]}
        log(f"[TEMPORAL-only] TEST acc {res['temporal_only']['acc']} "
            f"({tf_tr.shape[1]} genomes)")
        Ttr, Tte = rt.stack_static(torch, tp, rng, tf_tr.clone(), tf_te.clone(),
                                   n_fit, Yf, yv, log, pop, gens, rounds, spaces,
                                   seed_cols_tr=tf_tr, seed_cols_te=tf_te,
                                   tag="on-temporal")
        res["temporal_static"] = {"acc": acc(Ttr, Tte), "genomes": Ttr.shape[1]}
        log(f"[TEMPORAL+STATIC] TEST acc {res['temporal_static']['acc']} "
            f"({Ttr.shape[1]} genomes)")
    else:
        res["temporal_only"] = res["temporal_static"] = None

    res["seconds"] = round(time.time() - t0)
    res["local_swaps"] = local
    name = f"temporal_shuffle_local{local}.json" if local else \
        "temporal_shuffle_result.json"
    with open(os.path.join(RD, name), "w") as f:
        json.dump(res, f, indent=1)
    log("SHUFFLE RESULT: " + json.dumps(res))
    raw = res.get("raw", {}).get("acc")
    ts = (res.get("temporal_static") or {}).get("acc")
    to = (res.get("temporal_only") or {}).get("acc")
    if raw is not None and ts is not None:
        best = max(ts, to or 0)
        log(f"VERDICT: {'TEMPORAL HELPS' if best > raw else 'temporal does NOT beat static'} "
            f"| raw {raw} vs temporal_static {ts} temporal_only {to} "
            f"(anchor {res['anchor']}, chance 0.5)")
    print("SHUFFLE DONE", flush=True)


if __name__ == "__main__":
    cache = "wf_char_stream.pt"
    if "--cache" in sys.argv:
        cache = sys.argv[sys.argv.index("--cache") + 1]
    local = 0
    if "--local" in sys.argv:
        local = int(sys.argv[sys.argv.index("--local") + 1])
    run(cache, local=local)
