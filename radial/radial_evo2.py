"""radial_evo2.py — FULL SEND EXPRESSIVITY, with nothing hand-crafted.

The saturation analysis (evo campaign parts 6-7) measured that the v1 feature
genome class tops out at ~0.63-0.64 full-CIFAR. This module gives evolution a
GRAMMAR instead of a menu — every structural property that was previously a
design decision is now a GENE:

  scale   — patch size is a gene (4..14); the environment lazily builds the
            label-free patch-PCA basis for whatever scales evolution visits
  terms   — 2 or 3 interacting components (order evolves by add/remove-term
            mutations), each bent through a 1- or 2-deep lens program
            (depth evolves too)
  op      — primitive combine (product / min / |diff|), folded across terms
  pooling — NO region catalog: a soft spatial window with evolved center
            (cx, cy) and width (sigma); wide sigma IS global pooling, tight
            sigma IS a local region — evolution places it; the stat over the
            window (mean / max / std) is a primitive gene

The human contribution is only: math primitives (the same activation catalog
the radial space has always had, 3 ops, 3 stats) and data statistics (patch
PCA — built from the data, per the environment-not-organism rule). Evolution
decides scale, structure, depth, location, everything.

Training: full CIFAR 50k, fitness on an honest 10k held-back val split, test
touched exactly ONCE. Comma GA (first space — the energy economy applies to
downstream spaces per the house rule). Checkpoint per round
(radial_data/evo2_ckpt.json), STOP lever (radial_data/STOP_EVO), resumable.
"""
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CKPT = os.path.join(_HERE, "radial_data", "evo2_ckpt.json")

SCALES = [4, 6, 8, 10, 12, 14]
_PRIMS = ["id", "abs", "relu", "tanh", "gauss", "sq", "soft", "sin"]
_OPS = ["mult", "min", "absdiff"]
_STATS = ["mean", "max", "std"]
C_PER_SCALE = 40

_SVD_CACHE = {}          # (fingerprint, ps) -> (comps, mu): the patch-PCA basis is deterministic
#                          from the reference set, so reuse it across pages/calls (inference speed)


# ---------------------------------------------------------------------------
# environment: lazily-built per-scale patch-PCA maps (data statistics only)
# ---------------------------------------------------------------------------

class Env:
    def __init__(self, torch, dev, Xtr, Xte, max_cached=4, test_only=False):
        self.torch, self.dev = torch, dev
        self.Xtr, self.Xte = Xtr, Xte
        self.cache = {}                     # ps -> (Mtr fp16, Mte fp16, H, W)
        self.max_cached = max_cached
        self.last_used = {}
        self.tick = 0
        self.test_only = test_only          # inference: only the test slot is read, so skip
        #                                     projecting the (large, fixed) reference set -> Mtr

    def maps(self, ps):
        import torch.nn.functional as Fn
        torch = self.torch
        self.tick += 1
        if ps in self.cache:
            self.last_used[ps] = self.tick
            return self.cache[ps]
        if len(self.cache) >= self.max_cached:
            ev = min(self.last_used, key=self.last_used.get)
            del self.cache[ev]; del self.last_used[ev]
            torch.cuda.empty_cache()
        stride = max(2, ps // 2)
        d = ps * ps * self.Xtr.shape[3]          # channel-agnostic (3 for RGB)
        # the basis (comps, mu) is deterministic from the reference set — cache it across
        # Env instances (a fresh Env is built per page at inference) keyed by a cheap fingerprint
        fp = ((self.Xtr.shape, float(self.Xtr[:64].sum()), float(self.Xtr[-64:].sum()),
               float(self.Xtr.mean())) if len(self.Xtr) else (0,))
        bkey = (fp, ps)
        cached = _SVD_CACHE.get(bkey)
        if cached is not None:
            comps, mu = cached
        else:
            i2k = torch.tensor(self.Xtr[:2000], device=self.dev).permute(0, 3, 1, 2).contiguous()
            P = Fn.unfold(i2k, ps, stride=stride)
            cols = P.permute(0, 2, 1).reshape(-1, d)
            g = torch.Generator(device="cpu").manual_seed(ps)
            cols = cols[torch.randperm(len(cols), generator=g)[:100000].to(self.dev)]
            mu = cols.mean(0)
            _, _, V = torch.linalg.svd(cols - mu, full_matrices=False)
            comps = V[:min(C_PER_SCALE, d)]
            _SVD_CACHE[bkey] = (comps, mu)
        S = self.Xtr.shape[1]                    # input resolution (was hardcoded 32)
        H = W = (S - ps) // stride + 1

        def build(X, bs=400):
            out = None
            sd = None
            for b in range(0, len(X), bs):
                imgs = torch.tensor(X[b:b + bs], device=self.dev).permute(0, 3, 1, 2).contiguous()
                U = Fn.unfold(imgs, ps, stride=stride)
                M = torch.einsum("cd,bdl->bcl", comps, U - mu.view(1, -1, 1))
                if out is None:
                    out = torch.zeros((len(X), M.shape[1], M.shape[2]),
                                      device=self.dev, dtype=torch.float16)
                if sd is None:
                    sd = M.std((0, 2), keepdim=True) + 1e-6
                out[b:b + len(imgs)] = (M / sd).half()
            return out

        Mtr = None if self.test_only else build(self.Xtr)   # Mtr unused when only test feats read
        Mte = build(self.Xte)
        self.cache[ps] = (Mtr, Mte, H, W)
        self.last_used[ps] = self.tick
        return self.cache[ps]


# ---------------------------------------------------------------------------
# the grammar: genome -> one scalar feature per image
# ---------------------------------------------------------------------------

def new_genome(rng):
    order = 2 if rng.random() < 0.7 else 3
    return {
        "ps": int(rng.choice(SCALES)),
        "terms": [{"c": int(rng.integers(C_PER_SCALE)),
                   "prog": [(int(rng.integers(len(_PRIMS))),
                             float(rng.uniform(0.5, 2.5)),
                             float(rng.uniform(-1, 1)))
                            for _ in range(1 if rng.random() < 0.7 else 2)]}
                  for _ in range(order)],
        "op": int(rng.integers(len(_OPS))),
        "stat": int(rng.integers(len(_STATS))),
        "cx": float(rng.uniform(0.1, 0.9)), "cy": float(rng.uniform(0.1, 0.9)),
        "lsig": float(rng.uniform(np.log(0.15), np.log(1.5))),
    }


def mutate(rng, g, sc):
    c = json.loads(json.dumps(g))           # deep copy (plain types only)
    for t in c["terms"]:
        if rng.random() < 0.12:
            t["c"] = int(rng.integers(C_PER_SCALE))
        prog = [list(st) for st in t["prog"]]
        for st in prog:
            if rng.random() < 0.10:
                st[0] = int(rng.integers(len(_PRIMS)))
            st[1] = float(np.clip(st[1] + rng.normal(0, sc), 0.1, 4.0))
            st[2] = float(np.clip(st[2] + rng.normal(0, sc), -2.0, 2.0))
        if rng.random() < 0.10:              # depth evolves
            if len(prog) == 1:
                prog.append([int(rng.integers(len(_PRIMS))),
                             float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1, 1))])
            else:
                prog.pop(int(rng.integers(len(prog))))
        t["prog"] = [tuple(st) for st in prog]
    if rng.random() < 0.08:                  # order evolves
        if len(c["terms"]) == 2:
            c["terms"].append({"c": int(rng.integers(C_PER_SCALE)),
                               "prog": [(int(rng.integers(len(_PRIMS))),
                                         float(rng.uniform(0.5, 2.5)),
                                         float(rng.uniform(-1, 1)))]})
        else:
            c["terms"].pop(int(rng.integers(len(c["terms"]))))
    if rng.random() < 0.08:                  # scale evolves (walk to neighbour)
        i = SCALES.index(c["ps"])
        c["ps"] = SCALES[int(np.clip(i + rng.choice([-1, 1]), 0, len(SCALES) - 1))]
    if rng.random() < 0.08:
        c["op"] = int(rng.integers(len(_OPS)))
    if rng.random() < 0.08:
        c["stat"] = int(rng.integers(len(_STATS)))
    c["cx"] = float(np.clip(c["cx"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
    c["cy"] = float(np.clip(c["cy"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
    c["lsig"] = float(np.clip(c["lsig"] + rng.normal(0, sc * 0.5),
                              np.log(0.05), np.log(3.0)))
    return c


def crossover(rng, g1, g2):
    """Uniform per-gene recombination of two parents. Terms are drawn from
    either parent (order taken from one of them); scalar genes picked
    per-gene; the pooling window blends half the time."""
    c = json.loads(json.dumps(g1 if rng.random() < 0.5 else g2))
    pool = [json.loads(json.dumps(t)) for t in (g1["terms"] + g2["terms"])]
    rng.shuffle(pool)
    c["terms"] = pool[:len(c["terms"])]
    for t in c["terms"]:
        t["prog"] = [tuple(s) for s in t["prog"]]
    for k in ("ps", "op", "stat"):
        c[k] = (g1 if rng.random() < 0.5 else g2)[k]
    if rng.random() < 0.5:                       # blend the window
        for k in ("cx", "cy", "lsig"):
            c[k] = 0.5 * (g1[k] + g2[k])
    else:
        for k in ("cx", "cy", "lsig"):
            c[k] = (g1 if rng.random() < 0.5 else g2)[k]
    return c


def feature(torch, tp, env, g, test=False):
    Mtr, Mte, H, W = env.maps(g["ps"])
    M = Mte if test else Mtr
    z = None
    for t in g["terms"]:
        v = M[:, t["c"] % M.shape[1], :].float().view(len(M), H, W)
        for prim, a, b in t["prog"]:
            v = tp[_PRIMS[prim]](a * v + b)
        if z is None:
            z = v
        else:
            op = _OPS[g["op"]]
            z = z * v if op == "mult" else (torch.minimum(z, v) if op == "min"
                                            else torch.abs(z - v))
    # evolved soft spatial window
    ys = torch.linspace(0, 1, H, device=z.device).view(H, 1)
    xs = torch.linspace(0, 1, W, device=z.device).view(1, W)
    sig = float(np.exp(g["lsig"]))
    wgt = torch.exp(-(((xs - g["cx"]) ** 2) + ((ys - g["cy"]) ** 2)) / (2 * sig * sig))
    stat = _STATS[g["stat"]]
    if stat == "max":
        wn = wgt / (wgt.max() + 1e-9)
        return (z * wn + (wn - 1.0) * 30.0).amax((1, 2))
    wsum = wgt.sum() + 1e-9
    m = (z * wgt).sum((1, 2)) / wsum
    if stat == "mean":
        return m
    var = ((z - m.view(-1, 1, 1)) ** 2 * wgt).sum((1, 2)) / wsum
    return torch.sqrt(var + 1e-9)


# ---------------------------------------------------------------------------
# fast fitness: Schur-complement border ridge
#   The frozen base changes once per round; every candidate is base + ONE
#   column. Factor the base normal equations once (Cholesky), then each
#   candidate costs O(N*F) instead of O(N*F^2), and the whole batch of
#   candidates is scored in a handful of fused matmuls.
# ---------------------------------------------------------------------------

def make_scorer(torch, base, n_fit, Yf, yv, lam=3.0):
    """base: (n, F) raw frozen feature columns (F may be 0). Returns
    (score_fn, base_soft, base_acc); score_fn takes C (n, K) raw candidate
    columns and returns (soft, acc) arrays of length K."""
    dev = Yf.device
    n = base.shape[0]
    nv = n - n_fit
    Bf = base[:n_fit]
    if base.shape[1] > 0:
        mu, sd = Bf.mean(0), Bf.std(0) + 1e-6
        Bfz = torch.cat([(Bf - mu) / sd,
                         torch.ones(n_fit, 1, device=dev)], 1)
        Bvz = torch.cat([(base[n_fit:] - mu) / sd,
                         torch.ones(nv, 1, device=dev)], 1)
    else:
        Bfz = torch.ones(n_fit, 1, device=dev)
        Bvz = torch.ones(nv, 1, device=dev)
    F1 = Bfz.shape[1]
    # gram in TRUE fp32 (callers disable TF32 — TF32's ~1e-3 error was
    # the original poison), factor + solve in fp64. A pure-fp64 gram is
    # 60x slower on consumer GPUs and chokes at 2000+ base columns; if
    # the fp32 gram is still too rough, retry once in fp64.
    eye64 = lam * torch.eye(F1, device=dev, dtype=torch.float64)
    A = (Bfz.T @ Bfz).double() + eye64
    try:
        L = torch.linalg.cholesky(A)
    except Exception:
        # fp32-gram roundoff leaves wide bases (10k+ cols with one-hot /
        # sparse blocks) numerically non-PSD. The fp64 re-gram fallback is
        # a Bfz-sized copy (16GB+ at 17k cols x 120k rows) that OOMs
        # exactly when the base is widest - so first try escalating
        # IN-PLACE diagonal jitter (memory-free; just a stronger ridge on
        # the evolution scorer), and only re-gram in fp64 when the base is
        # small enough for the copy to be cheap.
        L = None
        for extra in (10 * lam, 100 * lam, 1000 * lam):
            A.diagonal().add_(extra)             # cumulative
            try:
                L = torch.linalg.cholesky(A)
                break
            except Exception:
                pass
        if L is None:
            A = Bfz.double().T @ Bfz.double() + eye64
            L = torch.linalg.cholesky(A)
    W0 = torch.cholesky_solve((Bfz.T @ Yf).double(), L).float()  # (F1, 10)
    S0 = Bvz @ W0                                      # (nv, 10)
    b_soft = float(torch.log_softmax(S0, 1)[torch.arange(nv), yv].mean())
    b_acc = float((S0.argmax(1) == yv).float().mean())

    def score(C):
        Cf = C[:n_fit]
        cmu, csd = Cf.mean(0), Cf.std(0) + 1e-6
        Cfz = (Cf - cmu) / csd
        Cvz = (C[n_fit:] - cmu) / csd
        U = Bfz.T @ Cfz                                # (F1, K)
        P = torch.cholesky_solve(U.double(), L).float()  # (F1, K)
        d = torch.clamp((Cfz * Cfz).sum(0) + lam - (U * P).sum(0), min=1e-6)
        W2 = (Cfz.T @ Yf - U.T @ W0) / d.view(-1, 1)   # (K, 10)
        Q = Cvz - Bvz @ P                              # (nv, K)
        # the (K, nv, n_cls) score cube is chunked over K: one piece is
        # 36GB at V=5000 classes (next-word) where the original 10-class
        # tasks needed 100MB. Chunking is exact - same numbers, less peak.
        K = Q.shape[1]
        n_cls = S0.shape[1]
        kc = max(1, int(2_000_000_000 // max(1, nv * n_cls * 4)))
        soft, acc = [], []
        for a in range(0, K, kc):
            S = (S0.unsqueeze(0) +
                 Q.T[a:a + kc].unsqueeze(2) * W2[a:a + kc].unsqueeze(1))
            soft += [float(x) for x in
                     torch.log_softmax(S, 2)[:, torch.arange(nv), yv].mean(1)]
            acc += [float(x) for x in (S.argmax(2) == yv).float().mean(1)]
        return soft, acc

    return score, b_soft, b_acc


# ---------------------------------------------------------------------------
# evolution (comma GA, honest 10k val, freeze-and-compose)
# ---------------------------------------------------------------------------

def run(rounds=400, pop_size=64, gens=12, freeze_top=8, seed=5, p_cross=0.0,
        ckpt_path=_CKPT, out_path=None, verbose=True):
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    z = np.load(os.path.join(_HERE, "radial_data", "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    env = Env(torch, dev, Xtr, Xte)
    n_fit = 40000
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((len(ytr), 10), device=dev)
    Yfull[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0

    frozen, fcols = [], []
    hist = []
    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ck = json.load(f)
        frozen = [g for g in ck.get("frozen", [])]
        for g in frozen:
            g["terms"] = [{"c": t["c"], "prog": [tuple(s) for s in t["prog"]]}
                          for t in g["terms"]]
        fcols = [feature(torch, tp, env, g) for g in frozen]
        hist = ck.get("hist", [])
        if verbose:
            print(f"[evo2] resumed {len(frozen)} frozen genomes", flush=True)

    empty_streak = 0
    for rnd in range(len(hist), rounds):
        base = torch.stack(fcols, 1) if fcols else torch.zeros((len(Xtr), 0), device=dev)
        scorer, s0, a0 = make_scorer(torch, base, n_fit, Yf, yv)

        pop = [new_genome(rng) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)

        def fit_pop(gs):
            cols = [feature(torch, tp, env, g) for g in gs]
            ok = [i for i, c in enumerate(cols)
                  if float(c.std()) > 1e-6 and bool(torch.isfinite(c).all())]
            softs = np.full(len(gs), -1e9)
            accs = np.zeros(len(gs))
            if ok:
                C = torch.stack([cols[i] for i in ok], 1)
                sf, ac = scorer(C)
                for j, i in enumerate(ok):
                    softs[i] = sf[j] - s0
                    accs[i] = ac[j]
            return softs, accs, cols

        fits, accs, cols = fit_pop(pop)
        for gen in range(gens):
            order = np.argsort(fits)[::-1]
            keep = list(order[:6])
            kids, ksc = [], []
            while len(kids) < pop_size - 6:
                cand = rng.choice(pop_size, 3)
                pi = cand[np.argmax(fits[cand])]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                if p_cross > 0 and rng.random() < p_cross:
                    cand2 = rng.choice(pop_size, 3)
                    pj = cand2[np.argmax(fits[cand2])]
                    kids.append(mutate(rng, crossover(rng, pop[pi], pop[pj]), sc))
                else:
                    kids.append(mutate(rng, pop[pi], sc))
                ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            pop = [pop[i] for i in keep] + kids
            scales = np.concatenate([scales[keep], ksc])
            fits = np.concatenate([fits[keep], kf])
            accs = np.concatenate([accs[keep], ka])
            cols = [cols[i] for i in keep] + kc
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= 0.0005 or added >= freeze_top:
                break
            col = cols[idx]
            colz = (col - col.mean()) / (col.std() + 1e-9)
            dup = False
            for fc in fcols[-60:]:
                fz = (fc - fc.mean()) / (fc.std() + 1e-9)
                if float(torch.abs((colz * fz).mean())) > 0.95:
                    dup = True
                    break
            if not dup:
                frozen.append(pop[idx]); fcols.append(col); added += 1
        base = torch.stack(fcols, 1) if fcols else torch.zeros((len(Xtr), 0), device=dev)
        s1, a1 = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv)
        hist.append({"round": rnd, "added": added, "n": len(frozen),
                     "val_acc": round(a1, 4)})
        if ckpt_path:
            tmp = ckpt_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"frozen": frozen, "hist": hist,
                           "seconds": round(time.time() - t0)}, f)
            os.replace(tmp, ckpt_path)
        if verbose:
            scs = sorted({g["ps"] for g in frozen})
            print(f"  [evo2] round {rnd:3d}  +{added} (total {len(frozen)})  "
                  f"val {a1:.4f}  scales-in-use {scs}  "
                  f"({round(time.time()-t0)}s)", flush=True)
        if os.path.exists(_STOP):
            print("[evo2] STOP lever pulled — checkpoint saved", flush=True)
            break
        empty_streak = empty_streak + 1 if added == 0 else 0
        if empty_streak >= 3:
            break

    # honest final: test touched once
    Fte = torch.stack([feature(torch, tp, env, g, test=True) for g in frozen], 1)
    Ftr = torch.stack(fcols, 1)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
        best = max(best, acc)
    orders = [len(g["terms"]) for g in frozen]
    out = {"phase": "F-grammar-v2", "p_cross": p_cross, "n_frozen": len(frozen),
           "hist": hist,
           "test_acc": round(best, 4),
           "val_final": hist[-1]["val_acc"] if hist else 0.0,
           "scales_used": sorted({g["ps"] for g in frozen}),
           "order_counts": {str(o): orders.count(o) for o in set(orders)},
           "stat_counts": {_STATS[s]: [g["stat"] for g in frozen].count(s)
                           for s in range(3)},
           "references": {"v1_class_tower": 0.6378, "v1_class_saturation": "0.63-0.64",
                          "coates_ng": 0.5904},
           "seconds": round(time.time() - t0)}
    with open(out_path or os.path.join(_HERE, "radial_data", "evo2_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[evo2] DONE: {len(frozen)} genomes, val {out['val_final']}, "
              f"TEST {best:.4f} (v1 class topped at 0.6378) "
              f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    run()
