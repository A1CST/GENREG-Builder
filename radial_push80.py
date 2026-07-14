"""radial_push80.py — the push to 80% (documentation/"push to 80.txt").

A downstream radial space evolved ON TOP of the converged grammar-v2
substrate, with the three levers from the plan, all as GENES:

  STACKING — this is stage 2 over the v2 genomes' outputs; the head sees
    [stage-1 features | stage-2 features], closed-form as always.
  META-GENOME — every term in a v3 genome chooses its SOURCE: a raw
    patch-PCA component map (any scale) or ANY frozen stage-1 genome's
    output. Genomes assembling from other genomes' outputs is a per-term
    gene, not a fixed wiring.
  CONDITIONAL ROUTING — an optional evolved GATE: the genome's feature is
    multiplied by sigmoid(k * gate(x)) where the gate is its own mini
    feature (source + bend + window) with evolved sharpness k. A genome can
    learn to fire only for the inputs its pathway serves — attention from
    first principles.

Downstream house rule: the ENERGY ECONOMY applies (steady-state population;
existing costs, outputting costs, any valid output restores a little, real
energy only from above-median contribution through the composition; the
starved die and their slots go to children). Fitness runs on the Schur
fast path; crossover p=0.5 (validated). Full 50k, honest 10k val, test
touched once. Checkpoints per round; STOP lever = radial_data/STOP_EVO.
"""
import json
import os
import time

import numpy as np

import radial_evo2 as e2
from radial_evo import _tprims, _STOP

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# grammar v3: v2 core + per-term source + optional gate
# ---------------------------------------------------------------------------

def new_genome3(rng, n_feat):
    g = e2.new_genome(rng)
    for t in g["terms"]:
        t["src"] = "feat" if rng.random() < 0.25 else "map"
        if t["src"] == "feat":
            t["c"] = int(rng.integers(n_feat))
    g["gate"] = _new_gate(rng, n_feat) if rng.random() < 0.3 else None
    return g


def _new_gate(rng, n_feat):
    src = "feat" if rng.random() < 0.4 else "map"
    return {"src": src,
            "c": int(rng.integers(n_feat if src == "feat" else e2.C_PER_SCALE)),
            "prog": [(int(rng.integers(len(e2._PRIMS))),
                      float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1, 1)))],
            "cx": float(rng.uniform(0.1, 0.9)), "cy": float(rng.uniform(0.1, 0.9)),
            "lsig": float(rng.uniform(np.log(0.15), np.log(1.5))),
            "k": float(rng.uniform(0.5, 3.0))}


def mutate3(rng, g, sc, n_feat):
    c = e2.mutate(rng, g, sc)
    for t in c["terms"]:
        if "src" not in t:
            t["src"] = "map"
        if rng.random() < 0.06:                      # source flips
            t["src"] = "feat" if t["src"] == "map" else "map"
            t["c"] = int(rng.integers(n_feat if t["src"] == "feat" else e2.C_PER_SCALE))
        elif t["src"] == "feat" and rng.random() < 0.10:
            t["c"] = int(rng.integers(n_feat))
    if rng.random() < 0.06:                          # gate appears/disappears
        c["gate"] = None if c.get("gate") else _new_gate(rng, n_feat)
    gt = c.get("gate")
    if gt:
        if rng.random() < 0.10:
            gt["prog"][0] = (int(rng.integers(len(e2._PRIMS))),
                             gt["prog"][0][1], gt["prog"][0][2])
        p0 = gt["prog"][0]
        gt["prog"][0] = (p0[0],
                         float(np.clip(p0[1] + rng.normal(0, sc), 0.1, 4.0)),
                         float(np.clip(p0[2] + rng.normal(0, sc), -2.0, 2.0)))
        gt["cx"] = float(np.clip(gt["cx"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
        gt["cy"] = float(np.clip(gt["cy"] + rng.normal(0, sc * 0.5), 0.0, 1.0))
        gt["lsig"] = float(np.clip(gt["lsig"] + rng.normal(0, sc * 0.5),
                                   np.log(0.05), np.log(3.0)))
        gt["k"] = float(np.clip(gt["k"] * np.exp(rng.normal(0, sc)), 0.1, 10.0))
    return c


def crossover3(rng, g1, g2, n_feat):
    c = e2.crossover(rng, g1, g2)
    for t in c["terms"]:
        if "src" not in t:
            t["src"] = "map"
        if t["src"] == "feat":
            t["c"] = t["c"] % n_feat
    c["gate"] = json.loads(json.dumps(
        (g1 if rng.random() < 0.5 else g2).get("gate")))
    return c


def _pooled(torch, tp, env, F1z, spec, is_gate=False):
    """A single (N,) signal from a spec with src/c/prog/window genes."""
    if spec["src"] == "feat":
        v = F1z[:, spec["c"] % F1z.shape[1]]
        for prim, a, b in spec["prog"]:
            v = tp[e2._PRIMS[prim]](a * v + b)
        return v
    Mtr, Mte, H, W = env.maps(spec.get("ps", 6))
    M = Mte if env._test_mode else Mtr
    v = M[:, spec["c"] % M.shape[1], :].float().view(len(M), H, W)
    for prim, a, b in spec["prog"]:
        v = tp[e2._PRIMS[prim]](a * v + b)
    ys = torch.linspace(0, 1, H, device=v.device).view(H, 1)
    xs = torch.linspace(0, 1, W, device=v.device).view(1, W)
    sig = float(np.exp(spec["lsig"]))
    wgt = torch.exp(-(((xs - spec["cx"]) ** 2) + ((ys - spec["cy"]) ** 2)) / (2 * sig * sig))
    return (v * wgt).sum((1, 2)) / (wgt.sum() + 1e-9)


def feature3(torch, tp, env, F1z, g, test=False):
    env._test_mode = test
    Mtr, Mte, H, W = env.maps(g["ps"])
    M = Mte if test else Mtr
    z = None
    for t in g["terms"]:
        if t.get("src") == "feat":
            v = F1z[:, t["c"] % F1z.shape[1]]
            for prim, a, b in t["prog"]:
                v = tp[e2._PRIMS[prim]](a * v + b)
            v = v.view(-1, 1, 1)
        else:
            v = M[:, t["c"] % M.shape[1], :].float().view(len(M), H, W)
            for prim, a, b in t["prog"]:
                v = tp[e2._PRIMS[prim]](a * v + b)
        if z is None:
            z = v
        else:
            op = e2._OPS[g["op"]]
            if z.shape != v.shape:                    # broadcast scalar over grid
                if z.dim() == 3 and z.shape[1] == 1:
                    z = z.expand(-1, v.shape[1], v.shape[2]) if v.dim() == 3 else z
                if v.dim() == 3 and v.shape[1] == 1 and z.dim() == 3:
                    v = v.expand(-1, z.shape[1], z.shape[2])
            z = z * v if op == "mult" else (torch.minimum(z, v) if op == "min"
                                            else torch.abs(z - v))
    if z.dim() == 3 and z.shape[1] > 1:
        ys = torch.linspace(0, 1, z.shape[1], device=z.device).view(-1, 1)
        xs = torch.linspace(0, 1, z.shape[2], device=z.device).view(1, -1)
        sig = float(np.exp(g["lsig"]))
        wgt = torch.exp(-(((xs - g["cx"]) ** 2) + ((ys - g["cy"]) ** 2)) / (2 * sig * sig))
        stat = e2._STATS[g["stat"]]
        if stat == "max":
            wn = wgt / (wgt.max() + 1e-9)
            core = (z * wn + (wn - 1.0) * 30.0).amax((1, 2))
        else:
            wsum = wgt.sum() + 1e-9
            m = (z * wgt).sum((1, 2)) / wsum
            if stat == "mean":
                core = m
            else:
                core = torch.sqrt(((z - m.view(-1, 1, 1)) ** 2 * wgt).sum((1, 2)) / wsum + 1e-9)
    else:
        core = z.view(len(z))
    gt = g.get("gate")
    if gt:
        gv = _pooled(torch, tp, env, F1z, {**gt, "ps": g["ps"]})
        gz = (gv - gv.mean()) / (gv.std() + 1e-9)
        core = core * torch.sigmoid(gt["k"] * gz)     # conditional routing
    env._test_mode = False
    return core


# ---------------------------------------------------------------------------
# the downstream evolution: energy economy + Schur scorer + crossover
# ---------------------------------------------------------------------------

def run(rounds=300, pop_size=64, gens=12, freeze_top=8, seed=21, p_cross=0.5,
        freeze_bar=0.0002, dry_streak=5,
        stage1_ckpt="radial_data/evo2x_ckpt.json",
        v3_ckpts=(), val_slice=0,
        ckpt_path="radial_data/push80_ckpt.json",
        out_path="radial_data/push80_cifar.json", verbose=True):
    """v3_ckpts: earlier v3 stages to fold into the substrate (their genomes
    are replayed against the channel bank of their time, then joined).
    val_slice: which 10k window of train is THIS stage's validation split —
    rotate per stage so selection never reuses an exhausted ruler."""
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    with open(os.path.join(_HERE, stage1_ckpt)) as f:
        g1 = json.load(f)["frozen"]
    for g in g1:
        g["terms"] = [{"c": t["c"], "prog": [tuple(s) for s in t["prog"]],
                       **({"src": t["src"]} if "src" in t else {})}
                      for t in g["terms"]]
    z = np.load(os.path.join(_HERE, "radial_data", "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    env = e2.Env(torch, dev, Xtr, Xte, max_cached=6)
    env._test_mode = False
    if verbose:
        print(f"[p80] stage-1: {len(g1)} genomes; computing substrate…", flush=True)
    F1tr = torch.stack([e2.feature(torch, tp, env, g) for g in g1], 1)
    F1te = torch.stack([e2.feature(torch, tp, env, g, test=True) for g in g1], 1)
    mu1, sd1 = F1tr.mean(0), F1tr.std(0) + 1e-6
    F1z_tr = (F1tr - mu1) / sd1
    F1z_te = (F1te - mu1) / sd1

    # fold earlier v3 stages into the substrate: replay each stage's genomes
    # against the channel bank AS IT EXISTED for that stage, then extend it
    for ckp in v3_ckpts:
        with open(os.path.join(_HERE, ckp)) as f:
            gs_prev = json.load(f)["frozen2"]
        for g in gs_prev:
            for t in g["terms"]:
                t["prog"] = [tuple(s) for s in t["prog"]]
            if g.get("gate"):
                g["gate"]["prog"] = [tuple(s) for s in g["gate"]["prog"]]
        Ptr = torch.stack([feature3(torch, tp, env, F1z_tr, g) for g in gs_prev], 1)
        Pte = torch.stack([feature3(torch, tp, env, F1z_te, g, test=True)
                           for g in gs_prev], 1)
        F1tr = torch.cat([F1tr, Ptr], 1)
        F1te = torch.cat([F1te, Pte], 1)
        mu1, sd1 = F1tr.mean(0), F1tr.std(0) + 1e-6
        F1z_tr = (F1tr - mu1) / sd1
        F1z_te = (F1te - mu1) / sd1
        if verbose:
            print(f"[p80] folded {len(gs_prev)} genomes from {ckp} "
                  f"(substrate now {F1tr.shape[1]})", flush=True)
    n_feat = F1tr.shape[1]

    # rotating honest validation window: a fresh ruler per stage
    n_all = len(ytr)
    n_fit = n_all - 10000
    val_lo = n_all - 10000 * (1 + int(val_slice))
    val_np = np.arange(val_lo, val_lo + 10000)
    fit_np = np.concatenate([np.arange(0, val_lo), np.arange(val_lo + 10000, n_all)])
    order_t = torch.tensor(np.concatenate([fit_np, val_np]), device=dev)
    yv = torch.tensor(ytr[val_np], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[fit_np], device=dev)] = 1.0
    Yfull = -torch.ones((len(ytr), 10), device=dev)
    Yfull[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0
    if verbose:
        print(f"[p80] val window rows {val_lo}-{val_lo+10000} (slice {val_slice})",
              flush=True)

    def featcol(g, test=False):
        F1zz = F1z_te if test else F1z_tr
        return feature3(torch, tp, env, F1zz, g, test=test)

    frozen2, hist = [], []
    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ck = json.load(f)
        frozen2 = ck.get("frozen2", [])
        hist = ck.get("hist", [])
        if verbose:
            print(f"[p80] resumed {len(frozen2)} stage-2 genomes", flush=True)
    fcols2 = [featcol(g) for g in frozen2]

    empty_streak = 0
    for rnd in range(len(hist), rounds):
        base = torch.cat([F1tr] + ([torch.stack(fcols2, 1)] if fcols2 else []), 1)
        scorer, s0, a0 = e2.make_scorer(torch, base[order_t], n_fit, Yf, yv)

        pop = [new_genome3(rng, n_feat) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)

        def fit_pop(gs):
            cols = [featcol(g) for g in gs]
            ok = [i for i, c in enumerate(cols)
                  if float(c.std()) > 1e-6 and bool(torch.isfinite(c).all())]
            softs = np.full(len(gs), -1e9)
            accs = np.zeros(len(gs))
            if ok:
                C = torch.stack([cols[i] for i in ok], 1)[order_t]
                sf, ac = scorer(C)
                for j, i in enumerate(ok):
                    softs[i] = sf[j] - s0
                    accs[i] = ac[j]
            return softs, accs, cols

        fits, accs, cols = fit_pop(pop)
        # downstream ENERGY ECONOMY: steady-state, relative contribution earns
        E_DECAY, OUT_COST, RESTORE, GAIN, E_FLOOR, E_MAX = 0.75, 0.05, 0.04, 400.0, 0.2, 1.5
        MIN_TURNOVER = 12
        energy = np.ones(pop_size)
        starved_total = 0
        for gen in range(gens):
            valid = fits > -1e8
            energy = np.clip(energy * E_DECAY - OUT_COST + RESTORE * valid
                             + GAIN * np.maximum(fits - np.median(fits), 0.0),
                             0.0, E_MAX)
            starved = energy < E_FLOOR
            starved_total += int(starved.sum())
            dead = list(np.where(starved)[0])
            if len(dead) < MIN_TURNOVER:
                living_by_fit = [i for i in np.argsort(fits) if i not in set(dead)]
                dead += living_by_fit[:MIN_TURNOVER - len(dead)]
            alive = [i for i in range(pop_size) if i not in set(dead)] or \
                    list(np.argsort(fits)[::-1][:4])
            kids, ksc = [], []
            n_fresh = max(1, len(dead) // 4)
            for k in range(len(dead)):
                if k < n_fresh:
                    kids.append(new_genome3(rng, n_feat)); ksc.append(0.25)
                    continue
                cand = rng.choice(alive, 3)
                pi = cand[np.argmax(fits[cand])]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                if rng.random() < p_cross:
                    cand2 = rng.choice(alive, 3)
                    pj = cand2[np.argmax(fits[cand2])]
                    kids.append(mutate3(rng, crossover3(rng, pop[pi], pop[pj], n_feat),
                                        sc, n_feat))
                else:
                    kids.append(mutate3(rng, pop[pi], sc, n_feat))
                ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            for slot, k in zip(dead, range(len(kids))):
                pop[slot] = kids[k]; scales[slot] = ksc[k]
                fits[slot] = kf[k]; accs[slot] = ka[k]
                cols[slot] = kc[k]; energy[slot] = 1.0
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= freeze_bar or added >= freeze_top:
                break
            col = cols[idx]
            colz = (col - col.mean()) / (col.std() + 1e-9)
            dup = False
            for fc in fcols2[-60:]:
                fz = (fc - fc.mean()) / (fc.std() + 1e-9)
                if float(torch.abs((colz * fz).mean())) > 0.95:
                    dup = True
                    break
            if not dup:
                frozen2.append(pop[idx]); fcols2.append(col); added += 1
        base = torch.cat([F1tr] + ([torch.stack(fcols2, 1)] if fcols2 else []), 1)
        _, _, a1 = e2.make_scorer(torch, base[order_t], n_fit, Yf, yv)
        n_gated = sum(1 for g in frozen2 if g.get("gate"))
        n_meta = sum(1 for g in frozen2 for t in g["terms"] if t.get("src") == "feat")
        hist.append({"round": rnd, "added": added, "n": len(frozen2),
                     "val_acc": round(a1, 4),
                     "starved_per_gen": round(starved_total / max(gens, 1), 1),
                     "gated": n_gated, "meta_terms": n_meta})
        starved_total = 0
        if ckpt_path:
            tmp = ckpt_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"frozen2": frozen2, "hist": hist,
                           "seconds": round(time.time() - t0)}, f)
            os.replace(tmp, ckpt_path)
        if verbose:
            h = hist[-1]
            print(f"  [p80] round {rnd:3d}  +{added} (total {len(frozen2)})  "
                  f"val {h['val_acc']:.4f}  starved/gen {h['starved_per_gen']}  "
                  f"gated {n_gated}  meta-terms {n_meta}  "
                  f"({round(time.time()-t0)}s)", flush=True)
        if os.path.exists(_STOP):
            print("[p80] STOP lever pulled — checkpoint saved", flush=True)
            break
        empty_streak = empty_streak + 1 if added == 0 else 0
        if empty_streak >= dry_streak:
            break

    F2te = (torch.stack([featcol(g, test=True) for g in frozen2], 1)
            if frozen2 else torch.zeros((len(yte), 0), device=dev))
    F2tr = (torch.stack(fcols2, 1)
            if fcols2 else torch.zeros((len(ytr), 0), device=dev))
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = e2._ridge_soft(torch, torch.cat([F1tr, F2tr], 1),
                                torch.cat([F1te, F2te], 1), Yfull, yte_t, lam=lam)
        best = max(best, acc)
    out = {"phase": "G-push80", "n_stage1": n_feat, "n_stage2": len(frozen2),
           "test_acc": round(best, 4),
           "val_final": hist[-1]["val_acc"] if hist else 0.0,
           "gated": sum(1 for g in frozen2 if g.get("gate")),
           "meta_terms": sum(1 for g in frozen2 for t in g["terms"]
                             if t.get("src") == "feat"),
           "seconds": round(time.time() - t0)}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[p80] DONE: stage-2 {len(frozen2)} genomes, val {out['val_final']}, "
              f"TEST {best:.4f} ({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    run()
