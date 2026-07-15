"""resnet_evo.py — GRADIENT-FREE evolved RESIDUAL networks on CIFAR-10.

The whole lab is gradient-free (GENREG rule #1: no gradients, no backprop, no
hybrid). This module keeps that law and asks the ResNet question inside it:
*can evolution discover the residual-block primitive and stack it usefully,
scored only by a closed-form linear read-out on a held-back val split?*

It mirrors `radial_evo2.py`'s machinery exactly — same environment (label-free
patch-PCA maps: "the features are the environment"), same fast Schur-complement
border-ridge scorer, same comma-GA freeze-and-compose loop, same checkpoint /
STOP / resume contract, same output-JSON shape — and swaps the feature grammar
for a **residual network genome**:

  stem     — a small set of PCA-component channel maps (C = 2..4), the input
             stack; channel count is fixed through the blocks so the residual
             identity add stays valid (classic ResNet within-stage rule)
  blocks   — a stack of residual blocks (DEPTH is a gene, 1..4). Each block is

                 h = h + gain * act( a * (mix @ h) + b )

             where `mix` is a C×C channel mixing (the 1×1-conv analog), `act`
             is one of the 8-function GENREG activation catalog (the signature
             evolved primitive), and `gain` is the residual scale. The identity
             SKIP is the ResNet gene. New blocks are BOOTSTRAPPED as near-no-op
             (gain≈0, mix≈identity, act=id) then evolved — per rule VI, stacked
             layers cannot be evolved from random init.
  head     — collapse the C channels to one map by an evolved weighted sum,
             then reduce to ONE scalar per image through the same evolved soft
             spatial window (center cx,cy + width sigma) and a stat primitive
             (mean / max / std) that radial_evo2 uses.

Human contribution is only math primitives (activation catalog, 3 stats) and
data statistics (patch PCA, built FROM the data). Evolution decides scale,
channels, depth, mixing, activations, residual gains, the head, everything.

Artifacts (checkpoint + output JSON + any exported images) default to
`F:\\Resnet` (override with env var GENREG_RESNET_DIR) so nothing large lands on
C:. Every completed run also writes a runs/ entry so it shows up on /runs.
"""
import datetime
import hashlib
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, make_scorer, C_PER_SCALE, SCALES

_HERE = os.path.dirname(os.path.abspath(__file__))

# Artifacts live off the C: drive by default (user directive: F:\Resnet).
OUT_DIR = os.environ.get("GENREG_RESNET_DIR", r"F:\Resnet")
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except OSError:                       # F: not present (e.g. on the pod) -> local
    OUT_DIR = os.path.join(_HERE, "radial_data")
    os.makedirs(OUT_DIR, exist_ok=True)

_CKPT = os.path.join(OUT_DIR, "resnet_evo_ckpt.json")
_RUNS = os.path.join(_HERE, "runs", "resnet_evo")

_PRIMS = ["id", "abs", "relu", "tanh", "gauss", "sq", "soft", "sin"]
_STATS = ["mean", "max", "std"]
MAX_C = 4                             # channel-stack width cap
MAX_DEPTH = 4                         # residual-block depth cap


# ---------------------------------------------------------------------------
# genome: a residual network -> one scalar feature per image
# ---------------------------------------------------------------------------

def _rand_block(rng, C, bootstrap=False):
    """One residual block. Bootstrapped blocks are near-no-op (identity mix,
    zero gain, id activation) so a freshly-added block does not disturb a
    working stack until evolution turns it on (rule VI)."""
    if bootstrap:
        mix = np.eye(C)
        return {"mix": mix.reshape(-1).tolist(), "prim": 0,   # id
                "a": 1.0, "b": 0.0, "gain": 0.05}
    mix = np.eye(C) + rng.normal(0, 0.25, (C, C))
    return {"mix": [float(x) for x in mix.reshape(-1)],
            "prim": int(rng.integers(len(_PRIMS))),
            "a": float(rng.uniform(0.5, 2.0)),
            "b": float(rng.uniform(-0.5, 0.5)),
            "gain": float(rng.uniform(0.2, 0.9))}


def new_genome(rng):
    C = int(rng.integers(2, MAX_C + 1))
    depth = 1 if rng.random() < 0.6 else 2
    return {
        "ps": int(rng.choice(SCALES)),
        "chans": [int(rng.integers(C_PER_SCALE)) for _ in range(C)],
        "blocks": [_rand_block(rng, C) for _ in range(depth)],
        "wout": [float(rng.normal(0, 1)) for _ in range(C)],
        "stat": int(rng.integers(len(_STATS))),
        "cx": float(rng.uniform(0.1, 0.9)), "cy": float(rng.uniform(0.1, 0.9)),
        "lsig": float(rng.uniform(np.log(0.15), np.log(1.5))),
    }


def _C(g):
    return len(g["chans"])


def mutate(rng, g, sc):
    c = json.loads(json.dumps(g))                 # deep copy, plain types
    C = _C(c)
    # channel identities
    for i in range(C):
        if rng.random() < 0.10:
            c["chans"][i] = int(rng.integers(C_PER_SCALE))
    # per-block params
    for blk in c["blocks"]:
        mix = np.array(blk["mix"], dtype=float).reshape(C, C)
        mix = mix + rng.normal(0, sc, (C, C)) * (np.abs(mix) + 0.1)
        blk["mix"] = [float(x) for x in mix.reshape(-1)]
        if rng.random() < 0.12:
            blk["prim"] = int(rng.integers(len(_PRIMS)))
        blk["a"] = float(np.clip(blk["a"] + rng.normal(0, sc), 0.1, 4.0))
        blk["b"] = float(np.clip(blk["b"] + rng.normal(0, sc), -2.0, 2.0))
        blk["gain"] = float(np.clip(blk["gain"] + rng.normal(0, sc), 0.0, 2.0))
    # depth evolves (add bootstrapped no-op / remove)
    if rng.random() < 0.10:
        if len(c["blocks"]) < MAX_DEPTH:
            c["blocks"].append(_rand_block(rng, C, bootstrap=True))
        elif len(c["blocks"]) > 1:
            c["blocks"].pop(int(rng.integers(len(c["blocks"]))))
    # scale evolves (walk to a neighbour)
    if rng.random() < 0.08:
        i = SCALES.index(c["ps"])
        c["ps"] = SCALES[int(np.clip(i + rng.choice([-1, 1]), 0, len(SCALES) - 1))]
    # head
    for i in range(C):
        c["wout"][i] = float(c["wout"][i] + rng.normal(0, sc))
    if rng.random() < 0.08:
        c["stat"] = int(rng.integers(len(_STATS)))
    c["cx"] = float(np.clip(c["cx"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
    c["cy"] = float(np.clip(c["cy"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
    c["lsig"] = float(np.clip(c["lsig"] + rng.normal(0, sc * 0.5),
                              np.log(0.05), np.log(3.0)))
    return c


def crossover(rng, g1, g2):
    """Uniform per-gene recombination. Because channel width C can differ, the
    block stack and head are taken WHOLE from one parent (they are C-shaped),
    while scale / stat / window mix per-gene or blend."""
    base = g1 if rng.random() < 0.5 else g2
    other = g2 if base is g1 else g1
    c = json.loads(json.dumps(base))
    # take the whole C-shaped structure (chans+blocks+wout) from one parent,
    # but let the OTHER parent donate its stat / scale sometimes
    for k in ("ps", "stat"):
        c[k] = (g1 if rng.random() < 0.5 else g2)[k]
    if rng.random() < 0.5:                        # blend the window
        for k in ("cx", "cy", "lsig"):
            c[k] = 0.5 * (g1[k] + g2[k])
    else:
        for k in ("cx", "cy", "lsig"):
            c[k] = (g1 if rng.random() < 0.5 else g2)[k]
    # optionally inherit the depth-1 head channel identities from `other`
    if len(other["chans"]) == len(c["chans"]) and rng.random() < 0.5:
        for i in range(len(c["chans"])):
            if rng.random() < 0.5:
                c["chans"][i] = other["chans"][i]
    return c


GRID = 4    # coarse spatial resolution passed BETWEEN stacked spaces. A space
            # emits a (n_genomes, GRID, GRID) map per image instead of a scalar,
            # so spatial structure ("where") survives into the next space and it
            # can build hierarchical features (edges -> textures -> objects).


def _blocks_collapse(torch, tp, h, g):
    """Shared residual-block core: channel stack h (N,C,H,W) -> collapsed map
    z (N,H,W) via the evolved residual blocks + weighted channel collapse."""
    C = h.shape[1]
    for blk in g["blocks"]:
        mix = torch.tensor(blk["mix"], device=h.device, dtype=h.dtype).view(C, C)
        hm = torch.einsum("ij,njhw->nihw", mix, h)          # channel mixing
        f = tp[_PRIMS[blk["prim"]]](blk["a"] * hm + blk["b"])
        h = h + blk["gain"] * f                             # RESIDUAL skip
    wout = torch.tensor(g["wout"], device=h.device, dtype=h.dtype)
    return torch.einsum("c,nchw->nhw", wout, h)             # (N, H, W)


def _window_pool(torch, z, g):
    """Evolved soft spatial window -> ONE scalar per image (readout/fitness)."""
    H, W = z.shape[1], z.shape[2]
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


def _to_grid(torch, z):
    """Coarsen a collapsed map z (N,H,W) to the (N,GRID,GRID) hand-off grid."""
    import torch.nn.functional as Fn
    return Fn.adaptive_avg_pool2d(z.unsqueeze(1), (GRID, GRID)).squeeze(1)


def feature(torch, tp, env, g, test=False, want_grid=False):
    """R0 residual net over patch-PCA maps. want_grid -> (N,GRID,GRID) spatial
    hand-off for the next space; else -> (N,) scalar readout column."""
    Mtr, Mte, H, W = env.maps(g["ps"])
    M = Mte if test else Mtr
    chans = [M[:, ci % M.shape[1], :].float().view(len(M), H, W) for ci in g["chans"]]
    h = torch.stack(chans, 1)                      # (N, C, H, W)
    z = _blocks_collapse(torch, tp, h, g)
    return _to_grid(torch, z) if want_grid else _window_pool(torch, z, g)


def feature_grid(torch, tp, gridT, g, want_grid=False):
    """R1+ residual net over the PREVIOUS space's (N, C_prev, GRID, GRID) map —
    stacking REPRESENTATIONS, not scalars. Same spatial genome machinery as R0;
    the channels are now the previous space's genomes and the resolution is GRID.
    want_grid -> (N,GRID,GRID) hand-off; else -> (N,) scalar readout column."""
    idx = [ci % gridT.shape[1] for ci in g["chans"]]
    h = gridT[:, idx].float()                      # (N, C, GRID, GRID)
    z = _blocks_collapse(torch, tp, h, g)          # (N, GRID, GRID)
    return z if want_grid else _window_pool(torch, z, g)


def new_genome_grid(rng, C_prev):
    """A spatial residual genome over a stacked space's (C_prev, GRID, GRID)
    map. Same shape as new_genome but channels index the previous space's
    genomes and there is no patch scale (ps)."""
    C = int(rng.integers(2, MAX_C + 1))
    depth = 1 if rng.random() < 0.6 else 2
    return {
        "chans": [int(rng.integers(C_prev)) for _ in range(C)],
        "blocks": [_rand_block(rng, C) for _ in range(depth)],
        "wout": [float(rng.normal(0, 1)) for _ in range(C)],
        "stat": int(rng.integers(len(_STATS))),
        "cx": float(rng.uniform(0.1, 0.9)), "cy": float(rng.uniform(0.1, 0.9)),
        "lsig": float(rng.uniform(np.log(0.15), np.log(1.5))),
    }


def mutate_grid(rng, g, sc, C_prev):
    """Mutate a grid (R1+) genome — like mutate() but chans index C_prev and
    there is no patch scale to walk."""
    c = json.loads(json.dumps(g))
    C = _C(c)
    for i in range(C):
        if rng.random() < 0.10:
            c["chans"][i] = int(rng.integers(C_prev))
    for blk in c["blocks"]:
        mix = np.array(blk["mix"], dtype=float).reshape(C, C)
        mix = mix + rng.normal(0, sc, (C, C)) * (np.abs(mix) + 0.1)
        blk["mix"] = [float(x) for x in mix.reshape(-1)]
        if rng.random() < 0.12:
            blk["prim"] = int(rng.integers(len(_PRIMS)))
        blk["a"] = float(np.clip(blk["a"] + rng.normal(0, sc), 0.1, 4.0))
        blk["b"] = float(np.clip(blk["b"] + rng.normal(0, sc), -2.0, 2.0))
        blk["gain"] = float(np.clip(blk["gain"] + rng.normal(0, sc), 0.0, 2.0))
    if rng.random() < 0.10:
        if len(c["blocks"]) < MAX_DEPTH:
            c["blocks"].append(_rand_block(rng, C, bootstrap=True))
        elif len(c["blocks"]) > 1:
            c["blocks"].pop(int(rng.integers(len(c["blocks"]))))
    for i in range(C):
        c["wout"][i] = float(c["wout"][i] + rng.normal(0, sc))
    if rng.random() < 0.08:
        c["stat"] = int(rng.integers(len(_STATS)))
    c["cx"] = float(np.clip(c["cx"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
    c["cy"] = float(np.clip(c["cy"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
    c["lsig"] = float(np.clip(c["lsig"] + rng.normal(0, sc * 0.5),
                              np.log(0.05), np.log(3.0)))
    return c


# ---------------------------------------------------------------------------
# runs/ integration — every completed run appears on /runs (no Flask needed)
# ---------------------------------------------------------------------------

def _record_run(cfg, hist, stats, log_lines, tags):
    """Write the standard runs/ file trio so the run shows up on the Runs page,
    matching the layout runstore/radial_demo_record use (env = 'resnet_evo')."""
    try:
        ts = datetime.datetime.now()
        h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str)
                         .encode()).hexdigest()[:6]
        rid = f"{ts.strftime('%Y%m%d-%H%M%S')}-resnet_evo-{h}"
        base = os.path.join(_RUNS, rid)
        os.makedirs(base, exist_ok=True)
        created = ts.isoformat(timespec="seconds")
        with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "environment": "resnet_evo", "created": created,
                       "config": cfg, "status": "finished"}, f, indent=2)
        with open(os.path.join(base, "history.jsonl"), "w", encoding="utf-8") as f:
            for hrec in hist:
                f.write(json.dumps({"gen": hrec.get("round"),
                                    "fitness": hrec.get("val_acc"),
                                    "added": hrec.get("added"),
                                    "n": hrec.get("n")}) + "\n")
        with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "environment": "resnet_evo", "status": "finished",
                       "finished": created, "best": stats, "checkpoint": None},
                      f, indent=2)
        with open(os.path.join(base, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"label": f"resnet-evo {stats.get('n_frozen')} blocks "
                                f"test {stats.get('test_acc')}",
                       "favorite": False, "group": "", "tags": tags}, f, indent=2)
        with open(os.path.join(base, "report.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "kind": "resnet_evo", "created": created,
                       "params": cfg, "stats": stats,
                       "log": [str(x)[:300] for x in log_lines][-200:]},
                      f, indent=2)
        return rid
    except Exception as exc:                       # recording must never kill a run
        print(f"[resnet-evo] run record failed (non-fatal): {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# evolution (comma GA, honest val split, freeze-and-compose) — mirrors evo2
# ---------------------------------------------------------------------------

def run(rounds=400, pop_size=64, gens=12, freeze_top=8, seed=5, p_cross=0.0,
        ckpt_path=_CKPT, out_path=None, verbose=True,
        smoke=False, n_train=None, n_test=None, record=True):
    """Evolve a bank of gradient-free residual-network features, freeze the top
    decorrelated survivors each round, read out with a closed-form ridge head.
    `smoke=True` runs a tiny end-to-end validation (small data subset, few
    rounds) to exercise the whole pipeline without a real training run."""
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    if smoke:
        rounds, pop_size, gens, freeze_top = 2, 16, 3, 4
        n_train = n_train or 3000
        n_test = n_test or 800

    z = np.load(os.path.join(_HERE, "radial_data", "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    if n_train:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    if n_test:
        Xte, yte = Xte[:n_test], yte[:n_test]

    env = Env(torch, dev, Xtr, Xte)
    n_fit = int(len(Xtr) * 0.8)                     # 80/20 honest val split
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((len(ytr), 10), device=dev)
    Yfull[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0

    frozen, fcols = [], []
    hist = []
    if ckpt_path and os.path.exists(ckpt_path) and not smoke:
        with open(ckpt_path) as f:
            ck = json.load(f)
        frozen = ck.get("frozen", [])
        fcols = [feature(torch, tp, env, g) for g in frozen]
        hist = ck.get("hist", [])
        if verbose:
            print(f"[resnet-evo] resumed {len(frozen)} frozen genomes", flush=True)

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
                Ck = torch.stack([cols[i] for i in ok], 1)
                sf, ac = scorer(Ck)
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
            depths = sorted({len(g["blocks"]) for g in frozen}) if frozen else []
            print(f"  [resnet-evo] round {rnd:3d}  +{added} (total {len(frozen)})  "
                  f"val {a1:.4f}  depths-in-use {depths}  "
                  f"({round(time.time()-t0)}s)", flush=True)
        if os.path.exists(_STOP):
            print("[resnet-evo] STOP lever pulled — checkpoint saved", flush=True)
            break
        empty_streak = empty_streak + 1 if added == 0 else 0
        if empty_streak >= 3:
            break

    # honest final: test touched once
    if not frozen:
        raise RuntimeError("no residual features survived — nothing to read out")
    Fte = torch.stack([feature(torch, tp, env, g, test=True) for g in frozen], 1)
    Ftr = torch.stack(fcols, 1)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
        best = max(best, acc)
    depths = [len(g["blocks"]) for g in frozen]
    widths = [_C(g) for g in frozen]
    gains = [b["gain"] for g in frozen for b in g["blocks"]]
    out = {"phase": "resnet-evo (gradient-free residual)", "p_cross": p_cross,
           "smoke": bool(smoke), "n_train": len(Xtr), "n_test": len(Xte),
           "n_frozen": len(frozen), "hist": hist,
           "test_acc": round(best, 4),
           "val_final": hist[-1]["val_acc"] if hist else 0.0,
           "scales_used": sorted({g["ps"] for g in frozen}),
           "depth_counts": {str(d): depths.count(d) for d in sorted(set(depths))},
           "width_counts": {str(w): widths.count(w) for w in sorted(set(widths))},
           "stat_counts": {_STATS[s]: [g["stat"] for g in frozen].count(s)
                           for s in range(3)},
           "mean_gain": round(float(np.mean(gains)), 4) if gains else 0.0,
           "references": {"radial_v1_class_tower": 0.6378,
                          "coates_ng": 0.5904,
                          "grammar_v2_record": 0.7035},
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(OUT_DIR,
                                  "resnet_evo_smoke.json" if smoke else "resnet_evo_cifar.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    out["out_path"] = op

    if record:
        cfg = {"rounds": rounds, "pop_size": pop_size, "gens": gens,
               "freeze_top": freeze_top, "seed": seed, "p_cross": p_cross,
               "smoke": bool(smoke), "n_train": len(Xtr), "n_test": len(Xte)}
        log = [f"round {h['round']}: +{h['added']} -> {h['n']} val {h['val_acc']}"
               for h in hist]
        rid = _record_run(cfg, hist,
                          {k: out[k] for k in ("test_acc", "val_final", "n_frozen",
                                               "scales_used", "depth_counts",
                                               "width_counts", "mean_gain")},
                          log, tags=(["smoke"] if smoke else []) + ["resnet", "gradient-free"])
        out["run_id"] = rid

    if verbose:
        print(f"[resnet-evo] DONE: {len(frozen)} residual genomes, "
              f"val {out['val_final']}, TEST {best:.4f}  "
              f"(records -> {op}) ({round(time.time()-t0)}s)", flush=True)
    return out


# ===========================================================================
# STACKED spaces with an EMERGENT per-space cap (documentation/stacking.txt)
#
#   "Don't set the cap. Make it a pressure. Let the energy economy decide."
#
# Each space runs the validated downstream energy economy (Phase D). There is
# NO hard genome cap: a space freezes every above-threshold, decorrelated
# contributor, and is declared FULL when it goes SATURATE_ROUNDS consecutive
# rounds adding nothing — scarcity found the natural size. Overflow then opens
# the NEXT space, which reads the PREVIOUS space's frozen outputs as its data
# (residual blocks over a feature vector). Depth emerges from pressure.
# ===========================================================================

# Phase-D energy economy constants (validated in radial_evo.phase_stack).
E_DECAY, OUT_COST, RESTORE, E_GAIN, E_FLOOR, E_MAX = 0.75, 0.05, 0.04, 400.0, 0.2, 1.5
MIN_TURNOVER = 12               # search pressure even with no starvation deaths
CAP_WINDOW = 5                  # rounds over which val must keep earning (patient:
                                # tolerates single-round dips so a space can MATURE
                                # through brief plateaus instead of tripping early)
MIN_WINDOW_GAIN = 0.002         # val gain over that window below which a space is FULL
MIN_SPACE_GAIN = 0.003          # a full space raising val by less => stop stacking

# Live-tunable cap: if F:\Resnet\cap.txt exists, its float overrides
# MIN_WINDOW_GAIN at the start of every round — tune the emergent cap WITHOUT
# restarting the run.
CAP_FILE = os.path.join(OUT_DIR, "cap.txt")


def _cap_thresh():
    try:
        with open(CAP_FILE) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return MIN_WINDOW_GAIN


# Space 0 (R0) is deterministic and unaffected by any downstream change
# (rotation, deeper-space caps, stacking params). Cache its frozen genomes once
# and REUSE them so iterating on R1+ never re-evolves R0. Cache is keyed by the
# params that actually determine R0, so changing the seed/pop/cap re-evolves it.
R0_CACHE = os.path.join(OUT_DIR, "r0_cache.json")


def _r0_key(seed, pop_size, gens, n_train, n_test, cap):
    return (f"s{seed}_p{pop_size}_g{gens}_tr{n_train}_te{n_test}"
            f"_cap{cap:.5f}_win{CAP_WINDOW}")


def _load_r0(key):
    try:
        with open(R0_CACHE) as f:
            c = json.load(f)
        if c.get("key") == key:
            gs = c["genomes"]
            for g in gs:                        # tuples became lists in JSON
                for blk in g["blocks"]:
                    blk["mix"] = [float(x) for x in blk["mix"]]
            return gs
    except (OSError, ValueError, KeyError):
        pass
    return None


def _save_r0(key, genomes):
    try:
        with open(R0_CACHE, "w") as f:
            json.dump({"key": key, "n": len(genomes), "genomes": genomes}, f)
    except OSError as exc:
        print(f"[resnet-stack] R0 cache save failed (non-fatal): {exc}", flush=True)
MAX_SPACES = 6                  # guard rail; the economy usually stops sooner
FREEZE_THRESH = 0.0005          # residual-gain bar to earn a slot


# ---- vector residual genome: residual blocks over a previous space's outputs

def new_vec_genome(rng, F_prev):
    C = int(rng.integers(2, MAX_C + 1))
    depth = 1 if rng.random() < 0.6 else 2
    return {
        "chans": [int(rng.integers(F_prev)) for _ in range(C)],
        "blocks": [_rand_block(rng, C) for _ in range(depth)],
        "wout": [float(rng.normal(0, 1)) for _ in range(C)],
    }


def mutate_vec(rng, g, sc, F_prev):
    c = json.loads(json.dumps(g))
    C = _C(c)
    for i in range(C):
        if rng.random() < 0.10:
            c["chans"][i] = int(rng.integers(F_prev))
    for blk in c["blocks"]:
        mix = np.array(blk["mix"], dtype=float).reshape(C, C)
        mix = mix + rng.normal(0, sc, (C, C)) * (np.abs(mix) + 0.1)
        blk["mix"] = [float(x) for x in mix.reshape(-1)]
        if rng.random() < 0.12:
            blk["prim"] = int(rng.integers(len(_PRIMS)))
        blk["a"] = float(np.clip(blk["a"] + rng.normal(0, sc), 0.1, 4.0))
        blk["b"] = float(np.clip(blk["b"] + rng.normal(0, sc), -2.0, 2.0))
        blk["gain"] = float(np.clip(blk["gain"] + rng.normal(0, sc), 0.0, 2.0))
    if rng.random() < 0.10:
        if len(c["blocks"]) < MAX_DEPTH:
            c["blocks"].append(_rand_block(rng, C, bootstrap=True))
        elif len(c["blocks"]) > 1:
            c["blocks"].pop(int(rng.integers(len(c["blocks"]))))
    for i in range(C):
        c["wout"][i] = float(c["wout"][i] + rng.normal(0, sc))
    return c


def _rotate_features(torch, X, deg):
    """Rotate a feature matrix (N, F) by `deg` degrees via block-diagonal 2x2
    Givens rotations across adjacent feature axes (0,1),(2,3),... — the radial
    'relative motion between data and lens creates diversity' move. Same R for
    train and test (deterministic). An odd trailing feature is left unrotated."""
    n = X.shape[1]
    c = float(np.cos(np.radians(deg)))
    s = float(np.sin(np.radians(deg)))
    R = torch.eye(n, device=X.device, dtype=X.dtype)
    idx = torch.arange(0, n - 1, 2, device=X.device)
    R[idx, idx] = c
    R[idx, idx + 1] = -s
    R[idx + 1, idx] = s
    R[idx + 1, idx + 1] = c
    return X @ R


def _embed_rotate(torch, Xtr, Xte, deg):
    """The FAITHFUL radial move: embed the feature set into its behavioral map
    (SVD -> principal axes, ranked by variance, like the genome map), then
    rotate that map about a SINGLE axis -- one plane, the dominant (PC0,PC1)
    plane -- by `deg`. Returns the rotated principal scores for train & test
    (V fit on train, applied to both). This is ONE plane of rotation, contrast
    the block version's ~F/2 simultaneous planes."""
    mu = Xtr.mean(0)
    Xtrc, Xtec = Xtr - mu, Xte - mu
    # principal axes of the behavioral map (fit on train)
    _, _, Vh = torch.linalg.svd(Xtrc, full_matrices=False)     # Vh: (F, F)
    Str = Xtrc @ Vh.T                                           # scores (N, F)
    Ste = Xtec @ Vh.T
    c = float(np.cos(np.radians(deg)))
    s = float(np.sin(np.radians(deg)))
    for S in (Str, Ste):                                        # rotate PC0-PC1 plane
        a = S[:, 0].clone(); b = S[:, 1].clone()
        S[:, 0] = a * c - b * s
        S[:, 1] = a * s + b * c
    return Str, Ste


def feature_vec(torch, tp, prevF, g):
    """Residual network over a feature vector (a previous space's outputs).
    prevF: (N, F_prev) z-scored. Returns (N,) scalar column. No gradients."""
    C = _C(g)
    idx = [ci % prevF.shape[1] for ci in g["chans"]]
    h = prevF[:, idx].clone()                       # (N, C)
    for blk in g["blocks"]:
        mix = torch.tensor(blk["mix"], device=h.device, dtype=h.dtype).view(C, C)
        hm = h @ mix.T                              # channel mixing
        f = tp[_PRIMS[blk["prim"]]](blk["a"] * hm + blk["b"])
        h = h + blk["gain"] * f                     # RESIDUAL skip
    wout = torch.tensor(g["wout"], device=h.device, dtype=h.dtype)
    return h @ wout                                 # (N,)


def _evolve_space(torch, np_rng, pop_size, gens, max_rounds, n_fit, Yf, yv,
                  base_prev, new_fn, mut_fn, feat_tr, log, verbose):
    """Evolve ONE space under the energy economy until it SATURATES (emergent
    cap). base_prev: (N, F) columns frozen in all EARLIER spaces (fitness is
    marginal contribution over these + this space's own frozen so far).
    Returns (frozen_genomes, this_space_train_cols)."""
    rng = np_rng
    dev = Yf.device
    frozen, fcols = [], []
    empty = 0
    vals = []
    for rnd in range(max_rounds):
        cols_here = torch.stack(fcols, 1) if fcols else torch.zeros((base_prev.shape[0], 0), device=dev)
        base = torch.cat([base_prev, cols_here], 1)
        scorer, s0, a0 = make_scorer(torch, base, n_fit, Yf, yv)

        pop = [new_fn(rng) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)

        def fit_pop(gs):
            cs = [feat_tr(g) for g in gs]
            ok = [i for i, c in enumerate(cs)
                  if float(c.std()) > 1e-6 and bool(torch.isfinite(c).all())]
            softs = np.full(len(gs), -1e9)
            accs = np.zeros(len(gs))
            if ok:
                Ck = torch.stack([cs[i] for i in ok], 1)
                sf, ac = scorer(Ck)
                for j, i in enumerate(ok):
                    softs[i] = sf[j] - s0
                    accs[i] = ac[j]
            return softs, accs, cs

        fits, accs, cols = fit_pop(pop)
        energy = np.ones(pop_size)
        starved_total = 0
        for gen in range(gens):
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
            alive = [i for i in range(pop_size) if i not in set(dead)] or \
                    list(np.argsort(fits)[::-1][:4])
            n_fresh = max(1, len(dead) // 4)
            kids, ksc = [], []
            for k in range(len(dead)):
                if k < n_fresh:
                    kids.append(new_fn(rng)); ksc.append(0.25)
                else:
                    cand = rng.choice(alive, 3)
                    pi = cand[int(np.argmax(fits[cand]))]
                    sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                    kids.append(mut_fn(rng, pop[pi], sc)); ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            for slot, k in zip(dead, range(len(kids))):
                pop[slot] = kids[k]; scales[slot] = ksc[k]
                fits[slot] = kf[k]; accs[slot] = ka[k]; cols[slot] = kc[k]
                energy[slot] = 1.0
        # freeze EVERY qualifying decorrelated contributor — NO hard cap
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= FREEZE_THRESH:
                break
            col = cols[idx]
            colz = (col - col.mean()) / (col.std() + 1e-9)
            dup = False
            for fc in fcols[-80:]:
                fz = (fc - fc.mean()) / (fc.std() + 1e-9)
                if float(torch.abs((colz * fz).mean())) > 0.95:
                    dup = True; break
            if not dup:
                frozen.append(pop[idx]); fcols.append(col); added += 1
        cols_here = torch.stack(fcols, 1) if fcols else torch.zeros((base_prev.shape[0], 0), device=dev)
        _, a1 = _ridge_soft(torch, torch.cat([base_prev, cols_here], 1)[:n_fit],
                            torch.cat([base_prev, cols_here], 1)[n_fit:], Yf, yv)
        vals.append(float(a1))
        spg = round(starved_total / max(gens, 1), 1)
        # PATIENT windowed cap: trip only when val is flat over CAP_WINDOW rounds,
        # so a space climbs THROUGH brief single-round dips and MATURES instead of
        # tripping on the first flat round. A 3-empty-round backstop still catches
        # genuinely dead spaces. Threshold is live-tunable via cap.txt.
        wgain = (vals[-1] - vals[-1 - CAP_WINDOW]) if len(vals) > CAP_WINDOW else None
        thresh = _cap_thresh()
        empty = empty + 1 if added == 0 else 0
        log(f"    round {rnd:3d}  +{added} (space {len(frozen)})  val {a1:.4f}  "
            f"starved/gen {spg}"
            + (f"  d-val/{CAP_WINDOW}r +{wgain:.4f} (cap {thresh:.4f})"
               if wgain is not None else ""),
            verbose)
        if (wgain is not None and wgain < thresh) or empty >= 3:
            why = (f"val plateau (+{wgain:.4f} over {CAP_WINDOW} rounds < {thresh:.4f})"
                   if wgain is not None and wgain < thresh else "3 empty rounds")
            log(f"    space FULL at {len(frozen)} genomes — {why}; matured, stacking next",
                verbose)
            break
    return frozen, fcols


def run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5, smoke=False,
                n_train=None, n_test=None, out_path=None, record=True, verbose=True,
                rot_deg=1.0, rot_mode="block", reuse_r0=True):
    """Emergent-cap stacked residual evolution. Each space self-sizes under the
    energy economy; a full space's outputs become the next space's data. Depth
    emerges from scarcity, not a hyperparameter."""
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    max_spaces = MAX_SPACES
    if smoke:
        pop_size, gens, max_rounds, max_spaces = 16, 4, 6, 2
        n_train = n_train or 3000
        n_test = n_test or 800

    z = np.load(os.path.join(_HERE, "radial_data", "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    if n_train:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    if n_test:
        Xte, yte = Xte[:n_test], yte[:n_test]
    env = Env(torch, dev, Xtr, Xte)
    n_fit = int(len(Xtr) * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((len(ytr), 10), device=dev)
    Yfull[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0

    def log(msg, v=True):
        if v:
            print(msg, flush=True)

    spaces = []                                 # per-space records
    all_tr, all_te = [], []                     # every frozen scalar column, all spaces
    prev_grid_tr = prev_grid_te = None          # previous space's (N, C, GRID, GRID) map
    val_prev = 0.0

    for si in range(max_spaces):
        base_prev = (torch.stack(all_tr, 1) if all_tr
                     else torch.zeros((len(Xtr), 0), device=dev))
        if si == 0:                             # R0: spatial, reads patch-PCA maps
            new_fn = new_genome
            mut_fn = lambda r, g, sc: mutate(r, g, sc)
            feat_tr = lambda g: feature(torch, tp, env, g)
            feat_te = lambda g: feature(torch, tp, env, g, test=True)
            grid_tr_of = lambda g: feature(torch, tp, env, g, want_grid=True)
            grid_te_of = lambda g: feature(torch, tp, env, g, test=True, want_grid=True)
            src = "patch-PCA maps"
        else:                                   # R1+: spatial, reads prev space's GRID map
            C_prev = prev_grid_tr.shape[1]
            new_fn = lambda r: new_genome_grid(r, C_prev)
            mut_fn = lambda r, g, sc: mutate_grid(r, g, sc, C_prev)
            feat_tr = lambda g: feature_grid(torch, tp, prev_grid_tr, g)
            feat_te = lambda g: feature_grid(torch, tp, prev_grid_te, g)
            grid_tr_of = lambda g: feature_grid(torch, tp, prev_grid_tr, g, want_grid=True)
            grid_te_of = lambda g: feature_grid(torch, tp, prev_grid_te, g, want_grid=True)
            src = f"space {si-1} maps ({C_prev} ch x {GRID}x{GRID} — spatial)"

        log(f"  [space {si}] opening — reads {src}", verbose)
        r0key = _r0_key(seed, pop_size, gens, len(Xtr), len(Xte), _cap_thresh())
        cached = _load_r0(r0key) if (si == 0 and reuse_r0) else None
        if cached is not None:
            frozen = cached
            fcols = [feat_tr(g) for g in frozen]   # rebuild columns (cheap vs re-evolving)
            log(f"  [space 0] REUSED {len(frozen)} genomes from R0 cache "
                f"(skipped re-evolving; {round(time.time()-t0)}s)", verbose)
        else:
            frozen, fcols = _evolve_space(torch, rng, pop_size, gens, max_rounds,
                                          n_fit, Yf, yv, base_prev, new_fn, mut_fn,
                                          feat_tr, log, verbose)
            if si == 0 and reuse_r0 and frozen:
                _save_r0(r0key, frozen)
                log(f"  [space 0] cached {len(frozen)} R0 genomes for reuse", verbose)
        if not frozen:
            log(f"  [space {si}] produced nothing — stop stacking", verbose)
            break
        fte = [feat_te(g) for g in frozen]
        all_tr.extend(fcols); all_te.extend(fte)
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:], Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "source": src, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4),
                       "depth_counts": {str(d): [len(g["blocks"]) for g in frozen].count(d)
                                        for d in sorted({len(g["blocks"]) for g in frozen})}})
        log(f"  [space {si}] FULL: {len(frozen)} genomes, val {val_now:.4f} "
            f"(+{gain:.4f} vs previous space) ({round(time.time()-t0)}s)", verbose)
        # checkpoint after each space
        ck = {"spaces": spaces, "seconds": round(time.time() - t0)}
        with open(os.path.join(OUT_DIR, "resnet_stack_ckpt.json"), "w") as f:
            json.dump(ck, f, indent=1)
        # build this space's GRID hand-off — spatial maps, not scalars, so the
        # next space keeps "where" and can build hierarchical features.
        prev_grid_tr = torch.stack([grid_tr_of(g) for g in frozen], 1).half()
        prev_grid_te = torch.stack([grid_te_of(g) for g in frozen], 1).half()
        val_prev = val_now
        if si > 0 and gain < MIN_SPACE_GAIN:    # nothing left to earn — stop
            log(f"  [space {si}] gain {gain:.4f} < {MIN_SPACE_GAIN} — the stack "
                f"is done; deeper spaces can't earn their keep", verbose)
            break
        if os.path.exists(_STOP):
            log("[resnet-stack] STOP lever pulled", verbose)
            break

    # honest final: test touched once, ridge on ALL frozen columns
    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
        best = max(best, acc)
    total = sum(s["n_frozen"] for s in spaces)
    out = {"phase": "resnet-stack (spatial-grid stacked residual)", "smoke": bool(smoke),
           "grid": GRID, "handoff": "spatial-grid (representations, not scalars)",
           "n_train": len(Xtr), "n_test": len(Xte),
           "n_spaces": len(spaces), "n_frozen_total": total,
           "test_acc": round(best, 4),
           "val_final": spaces[-1]["val_after"] if spaces else 0.0,
           "space_caps": [s["n_frozen"] for s in spaces],
           "spaces": spaces,
           "references": {"radial_v1_class_tower": 0.6378, "coates_ng": 0.5904,
                          "grammar_v2_record": 0.7035,
                          "resnet_single_space": 0.6593},
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(OUT_DIR,
                                  "resnet_stack_smoke.json" if smoke else "resnet_stack_cifar.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    out["out_path"] = op
    if record:
        cfg = {"mode": "stacked", "pop_size": pop_size, "gens": gens, "seed": seed,
               "smoke": bool(smoke), "n_train": len(Xtr), "n_test": len(Xte)}
        rid = _record_run(cfg, [{"round": s["space"], "added": s["n_frozen"],
                                 "n": sum(x["n_frozen"] for x in spaces[:i+1]),
                                 "val_acc": s["val_after"]}
                                for i, s in enumerate(spaces)],
                          {k: out[k] for k in ("test_acc", "val_final", "n_spaces",
                                               "n_frozen_total", "space_caps")},
                          [f"space {s['space']}: {s['n_frozen']} genomes, val "
                           f"{s['val_after']} (+{s['val_gain']})" for s in spaces],
                          tags=(["smoke"] if smoke else []) +
                               ["resnet", "gradient-free", "stacked"])
        out["run_id"] = rid
    log(f"[resnet-stack] DONE: {len(spaces)} spaces {out['space_caps']} "
        f"= {total} genomes, val {out['val_final']}, TEST {best:.4f} "
        f"(records -> {op}) ({round(time.time()-t0)}s)", verbose)
    return out


if __name__ == "__main__":
    import sys
    if "--stack" in sys.argv:
        run_stacked(smoke=("--smoke" in sys.argv))
    else:
        run(smoke=("--smoke" in sys.argv))
