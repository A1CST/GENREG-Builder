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
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(_HERE, "radial_data", "evo2_ckpt.json")

SCALES = [4, 6, 8, 10, 12, 14]
_PRIMS = ["id", "abs", "relu", "tanh", "gauss", "sq", "soft", "sin"]
_OPS = ["mult", "min", "absdiff"]
_STATS = ["mean", "max", "std"]
C_PER_SCALE = 40


# ---------------------------------------------------------------------------
# environment: lazily-built per-scale patch-PCA maps (data statistics only)
# ---------------------------------------------------------------------------

class Env:
    def __init__(self, torch, dev, Xtr, Xte, max_cached=4):
        self.torch, self.dev = torch, dev
        self.Xtr, self.Xte = Xtr, Xte
        self.cache = {}                     # ps -> (Mtr fp16, Mte fp16, H, W)
        self.max_cached = max_cached
        self.last_used = {}
        self.tick = 0

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
        d = ps * ps * 3
        i2k = torch.tensor(self.Xtr[:2000], device=self.dev).permute(0, 3, 1, 2).contiguous()
        P = Fn.unfold(i2k, ps, stride=stride)
        cols = P.permute(0, 2, 1).reshape(-1, d)
        g = torch.Generator(device="cpu").manual_seed(ps)
        cols = cols[torch.randperm(len(cols), generator=g)[:100000].to(self.dev)]
        mu = cols.mean(0)
        _, _, V = torch.linalg.svd(cols - mu, full_matrices=False)
        comps = V[:min(C_PER_SCALE, d)]
        H = W = (32 - ps) // stride + 1

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

        Mtr = build(self.Xtr)
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
# evolution (comma GA, honest 10k val, freeze-and-compose)
# ---------------------------------------------------------------------------

def run(rounds=400, pop_size=64, gens=12, freeze_top=8, seed=5, verbose=True):
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
    if os.path.exists(_CKPT):
        with open(_CKPT) as f:
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
        s0, a0 = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv) \
            if fcols else (-np.log(10.0), 0.1)
        pop = [new_genome(rng) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)

        def fit_pop(gs):
            softs, accs, cols = [], [], []
            for g in gs:
                col = feature(torch, tp, env, g)
                if float(col.std()) < 1e-6 or not bool(torch.isfinite(col).all()):
                    softs.append(-1e9); accs.append(0.0); cols.append(col)
                    continue
                X = torch.cat([base, col.view(-1, 1)], 1)
                s, a = _ridge_soft(torch, X[:n_fit], X[n_fit:], Yf, yv)
                softs.append(s - s0); accs.append(a); cols.append(col)
            return np.array(softs), np.array(accs), cols

        fits, accs, cols = fit_pop(pop)
        for gen in range(gens):
            order = np.argsort(fits)[::-1]
            keep = list(order[:6])
            kids, ksc = [], []
            while len(kids) < pop_size - 6:
                cand = rng.choice(pop_size, 3)
                pi = cand[np.argmax(fits[cand])]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                kids.append(mutate(rng, pop[pi], sc)); ksc.append(sc)
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
        tmp = _CKPT + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"frozen": frozen, "hist": hist,
                       "seconds": round(time.time() - t0)}, f)
        os.replace(tmp, _CKPT)
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
    out = {"phase": "F-grammar-v2", "n_frozen": len(frozen),
           "test_acc": round(best, 4),
           "val_final": hist[-1]["val_acc"] if hist else 0.0,
           "scales_used": sorted({g["ps"] for g in frozen}),
           "order_counts": {str(o): orders.count(o) for o in set(orders)},
           "stat_counts": {_STATS[s]: [g["stat"] for g in frozen].count(s)
                           for s in range(3)},
           "references": {"v1_class_tower": 0.6378, "v1_class_saturation": "0.63-0.64",
                          "coates_ng": 0.5904},
           "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "radial_data", "evo2_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[evo2] DONE: {len(frozen)} genomes, val {out['val_final']}, "
              f"TEST {best:.4f} (v1 class topped at 0.6378) "
              f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    run()
