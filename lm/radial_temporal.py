"""radial_temporal.py - the temporal->static freeze-and-stack for language.

The design (converged with the user):

  space 0  TEMPORAL  - reads the context words as an ORDERED STREAM and
                       composes over RELATIVE-TIME offsets (position-invariant:
                       "channel a at t interacts with channel b at t-o, pooled
                       over all t"). Same next-word fitness. Weak, expected.
  space 1+ STATIC    - reads the temporal space's FROZEN output as its
                       environment and does the same job, surfacing which
                       temporal signals help predict the next word.

Persistence is the operator: a relative-time pattern that holds across the
window is what a temporal genome fires on, and freezing it crystallizes that
pattern into a static feature the next space classifies on. Freeze == the
temporal->static graduation. Head is genome-only throughout (no ridge head
reads the raw environment).

THE TEST (the only number that matters): does static-on-temporal beat
static-on-raw at next-word? If relative-time composition surfaces signal a
static-chunk model cannot, temporal > raw. Reuses today's cached word features
(no eye sweep) so it is cheap.

  python radial_temporal.py [--smoke]
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

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
V_B = 500
CACHE = "wf_kid_next_g8_A515_B297_N50000x10000.pt"
# pooling modes a temporal term may use. Default {mean, max} is fully
# position-invariant (throws away recency). --recency adds {last,
# recency-weighted} so a genome can KEEP recency - next-word is
# recency-dominated, so the flat-pool version lost to static-position.
N_POOL = 2


# -- the temporal genome: relative-time offset interactions ----------------
def new_temporal(rng, C, T):
    nterms = 1 + int(rng.random() < 0.6) + int(rng.random() < 0.3)   # 1..3
    return {"terms": [{
        "a": int(rng.integers(C)), "b": int(rng.integers(C)),
        "o": int(rng.integers(1, T)),          # relative offset, >=1
        "op": int(rng.integers(3)),            # 0 prod, 1 diff, 2 min
        "pool": int(rng.integers(N_POOL)),     # 0 mean 1 max [2 last 3 recency]
        "w": float(rng.uniform(-1, 1)),
    } for _ in range(nterms)]}


def mutate_temporal(rng, g, sc, C, T):
    c = json.loads(json.dumps(g))
    for t in c["terms"]:
        if rng.random() < 0.20: t["a"] = int(rng.integers(C))
        if rng.random() < 0.20: t["b"] = int(rng.integers(C))
        if rng.random() < 0.15: t["o"] = int(rng.integers(1, T))
        if rng.random() < 0.15: t["op"] = int(rng.integers(3))
        if rng.random() < 0.15: t["pool"] = int(rng.integers(N_POOL))
        t["w"] += float(rng.normal(0, sc))
    if rng.random() < 0.15 and len(c["terms"]) < 3:
        c["terms"].append(new_temporal(rng, C, T)["terms"][0])
    if rng.random() < 0.15 and len(c["terms"]) > 1:
        c["terms"].pop(int(rng.integers(len(c["terms"]))))
    return c


def temporal_feat(torch, F, g):
    """F: (N, T, C) -> (N,). Each term pools a relative-time interaction over
    all valid t, so the same (a, b, offset) pattern is detected regardless of
    absolute position - the property a static-concat genome cannot express."""
    N, T, C = F.shape
    z = None
    for t in g["terms"]:
        o = min(t["o"], T - 1)
        xa = F[:, o:, t["a"] % C]              # values at t = o..T-1
        xb = F[:, :T - o, t["b"] % C]          # values at t-o
        if t["op"] == 0:
            v = xa * xb
        elif t["op"] == 1:
            v = xa - xb
        else:
            v = torch.minimum(xa, xb)
        p = t["pool"]
        if p == 1:
            v = v.amax(1)                       # max over time (pos-invariant)
        elif p == 2:
            v = v[:, -1]                        # LAST interaction (recency)
        elif p == 3:                            # recency-weighted mean
            wt = torch.arange(1, v.shape[1] + 1, device=v.device,
                              dtype=v.dtype)
            v = (v * wt).sum(1) / wt.sum()
        else:
            v = v.mean(1)                       # mean over time (pos-invariant)
        v = t["w"] * v
        z = v if z is None else z + v
    return z


# -- helpers ---------------------------------------------------------------
def _finite(torch, c):
    return torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e6, 1e6)


def readout(torch, Ftr, Fte, Yfull, yte_t, Ntr, Nte, dev):
    """genome-only ridge head with a small lambda sweep -> (top1, top5)."""
    best1 = best5 = 0.0
    for lam in (1.0, 3.0, 10.0):
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        Am = torch.hstack([(Ftr - mu) / sd, torch.ones(Ntr, 1, device=dev)])
        Gm = (Am.T @ Am).double() + lam * torch.eye(
            Ftr.shape[1] + 1, device=dev, dtype=torch.float64)
        Wm = torch.linalg.solve(Gm, (Am.T @ Yfull).double()).float()
        sc = torch.hstack([(Fte - mu) / sd, torch.ones(Nte, 1, device=dev)]) @ Wm
        a1 = float((sc.argmax(1) == yte_t).float().mean())
        k5 = min(5, sc.shape[1])               # binary tasks have <5 classes
        a5 = float((sc.topk(k5, 1).indices == yte_t.view(-1, 1)).any(1).float().mean())
        if a1 > best1:
            best1, best5 = a1, a5
    return round(best1, 4), round(best5, 4)


def stack_static(torch, tp, rng, bank_tr, bank_te, n_fit, Yf, yv, log,
                 pop, gens, rounds, max_spaces, seed_cols_tr=None,
                 seed_cols_te=None, tag="static"):
    """Evolve genome-only static vec spaces over `bank`. Returns the full
    per-genome feature matrices (train, test) = seed cols + every frozen col."""
    all_tr = [] if seed_cols_tr is None else list(seed_cols_tr.T)
    all_te = [] if seed_cols_te is None else list(seed_cols_te.T)
    dev = bank_tr.device
    for si in range(max_spaces):
        C = bank_tr.shape[1]
        new_fn = lambda r: rk.new_vec_genome(r, C)
        mut_fn = lambda r, g, sc: rk.mutate_vec(r, g, sc, C)
        feat_tr = lambda g, b=bank_tr: _finite(torch, rk.feature_vec(torch, tp, b, g))
        feat_te = lambda g, b=bank_te: _finite(torch, rk.feature_vec(torch, tp, b, g))
        base_prev = (torch.stack(all_tr, 1) if all_tr
                     else torch.zeros((bank_tr.shape[0], 0), device=dev))
        log(f"  [{tag} space {si}] opening - bank {C} ch, base {base_prev.shape[1]}")
        frozen, fcols = rk._evolve_space(torch, rng, pop, gens, rounds, n_fit,
                                         Yf, yv, base_prev, new_fn, mut_fn,
                                         feat_tr, log, True)
        if not frozen:
            log(f"  [{tag} space {si}] produced nothing - stop")
            break
        rk.bake_gate_stats(torch, tp, frozen, bank_tr)
        f_tr = torch.stack(fcols, 1)
        f_te = torch.stack([feat_te(g) for g in frozen], 1)
        fmu, fsd = f_tr.mean(0), f_tr.std(0) + 1e-6
        f_tr = ((f_tr - fmu) / fsd).clamp(-8, 8)
        f_te = ((f_te - fmu) / fsd).clamp(-8, 8)
        all_tr.extend(f_tr[:, j] for j in range(f_tr.shape[1]))
        all_te.extend(f_te[:, j] for j in range(f_te.shape[1]))
        bank_tr = torch.cat([bank_tr, f_tr], 1)
        bank_te = torch.cat([bank_te, f_te], 1)
        _, va = _ridge_soft(torch, torch.stack(all_tr, 1)[:n_fit],
                            torch.stack(all_tr, 1)[n_fit:], Yf, yv)
        log(f"  [{tag} space {si}] FULL: {len(frozen)} genomes, val {va:.4f}")
    Ftr = torch.stack(all_tr, 1) if all_tr else None
    Fte = torch.stack(all_te, 1) if all_te else None
    return Ftr, Fte


def run(smoke=False, cache=CACHE):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    rng = np.random.default_rng(0)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    log_lines = []

    def log(m, v=True):
        log_lines.append(m)
        print(m, flush=True)

    log(f"[temporal] cache = {cache}")
    blob = torch.load(os.path.join(RD, cache), map_location=dev)
    wf_tr = [t.float().to(dev) for t in blob["tr"]]   # P_C x (Ntr, C)
    wf_te = [t.float().to(dev) for t in blob["te"]]
    z = np.load(os.path.join(RD, "kid_next.npz"))
    ytr, yte = z["ytr"], z["yte"]
    T = len(wf_tr)
    Ntr, C = wf_tr[0].shape
    Nte = wf_te[0].shape[0]
    log(f"[temporal] loaded cache: T={T} words, C={C} channels/word, "
        f"Ntr={Ntr} Nte={Nte}")

    if smoke:
        k = 4000
        wf_tr = [w[:k] for w in wf_tr]; wf_te = [w[:1000] for w in wf_te]
        ytr, yte = ytr[:k], yte[:1000]
        Ntr, Nte = k, 1000
        pop, gens, rounds, spaces = 32, 6, 8, 2
        log("[temporal] SMOKE mode (subset + short budget)")
    else:
        pop, gens, rounds, spaces = 96, 12, 400, 4

    # (N, T, C) sequence; z-score PER CHANNEL over (N,T) so a channel is
    # comparable across time (required for diff/prod to mean anything).
    F_tr = torch.stack(wf_tr, 1)                      # (Ntr, T, C)
    F_te = torch.stack(wf_te, 1)
    cmu = F_tr.reshape(-1, C).mean(0)
    csd = F_tr.reshape(-1, C).std(0) + 1e-6
    F_tr = ((F_tr - cmu) / csd).clamp(-8, 8)
    F_te = ((F_te - cmu) / csd).clamp(-8, 8)

    # static concat bank (N, T*C), z-scored per column (the RAW-chunk view)
    cat_tr = F_tr.reshape(Ntr, T * C)
    cat_te = F_te.reshape(Nte, T * C)
    smu, ssd = cat_tr.mean(0), cat_tr.std(0) + 1e-6
    cat_tr = ((cat_tr - smu) / ssd).clamp(-8, 8)
    cat_te = ((cat_te - smu) / ssd).clamp(-8, 8)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, V_B), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, V_B), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    # reference linear anchor: ridge on the raw concat, NO genomes
    _, anchor = _ridge_soft(torch, cat_tr[:n_fit], cat_tr[n_fit:], Yf, yv)
    log(f"[temporal] linear anchor (concat -> ridge, no genomes): "
        f"val {anchor:.4f}")

    res = {"anchor": round(float(anchor), 4), "T": T, "C": C,
           "Ntr": Ntr, "Nte": Nte}

    # ---- ARM 1: STATIC ON RAW (the ablation floor) ----
    log("=== ARM RAW: static genome spaces over the concat chunk ===")
    Rtr, Rte = stack_static(torch, tp, rng, cat_tr.clone(), cat_te.clone(),
                            n_fit, Yf, yv, log, pop, gens, rounds, spaces,
                            tag="raw")
    if Rtr is not None:
        r1, r5 = readout(torch, Rtr, Rte, Yfull, yte_t, Ntr, Nte, dev)
        res["raw"] = {"test": r1, "top5": r5, "genomes": Rtr.shape[1]}
        log(f"[RAW] TEST top1 {r1} top5 {r5}  ({Rtr.shape[1]} genomes)")

    # ---- ARM 2: TEMPORAL space, then STATIC on its frozen output ----
    log("=== ARM TEMPORAL: relative-time genomes, then static on frozen ===")
    new_fn = lambda r: new_temporal(r, C, T)
    mut_fn = lambda r, g, sc: mutate_temporal(r, g, sc, C, T)
    feat_tr = lambda g: _finite(torch, temporal_feat(torch, F_tr, g))
    feat_te = lambda g: _finite(torch, temporal_feat(torch, F_te, g))
    base0 = torch.zeros((Ntr, 0), device=dev)
    frozenT, fcolsT = rk._evolve_space(torch, rng, pop, gens, rounds, n_fit,
                                       Yf, yv, base0, new_fn, mut_fn,
                                       feat_tr, log, True)
    log(f"[TEMPORAL space 0] froze {len(frozenT)} relative-time genomes")
    if not frozenT:
        log("[TEMPORAL] space 0 earned nothing - no temporal signal at this "
            "budget/representation")
        res["temporal_only"] = None
        res["temporal_static"] = None
    else:
        tf_tr = torch.stack(fcolsT, 1)
        tf_te = torch.stack([feat_te(g) for g in frozenT], 1)
        tmu, tsd = tf_tr.mean(0), tf_tr.std(0) + 1e-6
        tf_tr = ((tf_tr - tmu) / tsd).clamp(-8, 8)
        tf_te = ((tf_te - tmu) / tsd).clamp(-8, 8)
        # temporal-only readout (how much the temporal genomes carry alone)
        t1, t5 = readout(torch, tf_tr, tf_te, Yfull, yte_t, Ntr, Nte, dev)
        res["temporal_only"] = {"test": t1, "top5": t5, "genomes": tf_tr.shape[1]}
        log(f"[TEMPORAL-only] TEST top1 {t1} top5 {t5}  ({tf_tr.shape[1]} genomes)")
        # static spaces reading ONLY the frozen temporal bank (enforcement:
        # the static genomes see temporal features, never the raw tokens)
        Ttr, Tte = stack_static(torch, tp, rng, tf_tr.clone(), tf_te.clone(),
                                n_fit, Yf, yv, log, pop, gens, rounds, spaces,
                                seed_cols_tr=tf_tr, seed_cols_te=tf_te,
                                tag="on-temporal")
        s1, s5 = readout(torch, Ttr, Tte, Yfull, yte_t, Ntr, Nte, dev)
        res["temporal_static"] = {"test": s1, "top5": s5,
                                  "genomes": Ttr.shape[1]}
        log(f"[TEMPORAL+STATIC] TEST top1 {s1} top5 {s5}  ({Ttr.shape[1]} genomes)")

    res["seconds"] = round(time.time() - t0)
    res["cache"] = cache
    tag = "char" if "char" in cache else "word"
    with open(os.path.join(RD, f"temporal_result_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("TEMPORAL RESULT: " + json.dumps(res))
    # the verdict
    raw = res.get("raw", {}).get("test")
    ts = (res.get("temporal_static") or {}).get("test")
    to = (res.get("temporal_only") or {}).get("test")
    if raw is not None and ts is not None:
        best = max(ts, to or 0)
        verdict = ("TEMPORAL HELPS" if best > raw else
                   "temporal does NOT beat static-on-raw")
        log(f"VERDICT: {verdict} | raw {raw} vs temporal_static {ts} "
            f"temporal_only {to} (anchor {res['anchor']})")
    print("TEMPORAL DONE", flush=True)
    return res


if __name__ == "__main__":
    cache = CACHE
    if "--cache" in sys.argv:
        cache = sys.argv[sys.argv.index("--cache") + 1]
    if "--recency" in sys.argv:
        N_POOL = 4                              # enable last + recency-weighted
        print("[temporal] RECENCY pooling enabled (modes 0..3)", flush=True)
    run(smoke="--smoke" in sys.argv, cache=cache)
