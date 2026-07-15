"""radial_stack.py — the CIFAR grammar line, restructured to match resnet_evo.

The ResNet line (`resnet_evo.py`) was copied from this project and then grew
new properties (documentation/stacking.txt: "Don't set the cap. Make it a
pressure. Let the energy economy decide."). This module ports that structure
back so the CIFAR run is shaped identically:

  EMERGENT-CAP STACKED SPACES — no freeze_top hard cap: every above-threshold,
    decorrelated contributor is frozen; a space is FULL when a round's val
    gain drops below the cap threshold (value-based saturation, not count).
    Overflow opens the NEXT space, which reads the previous space's frozen
    outputs as its data. Depth emerges from scarcity, not design.
  ENERGY ECONOMY IN EVERY SPACE — steady-state population, existence + output
    costs, only above-median contribution earns; the starved die and their
    slots go to children (validated Phase-D constants).
  LIVE-TUNABLE CAP — cap.txt in the artifact dir overrides the saturation
    threshold at every round, no restart needed.
  ARTIFACTS OFF C: — default F:\\Radial (env GENREG_RADIAL_DIR), falling back
    to radial_data/ where F: is absent (e.g. the pod).
  RUNS INTEGRATION — every completed run writes the standard runs/ file trio
    so it appears on /runs (environment "radial_stack").
  SMOKE MODE — end-to-end pipeline validation on a small subset.

The grammar itself is this line's own: space 0 evolves grammar-v2 spatial
feature genomes (scale/terms/depth/window all genes, radial_evo2); deeper
spaces evolve VECTOR grammar genomes over the previous space's outputs —
terms bend and fold other genomes' outputs (the meta-genome) with an optional
sigmoid gate (conditional routing), per radial_push80. No gradients anywhere.
"""
import datetime
import hashlib
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import (Env, make_scorer, new_genome, mutate, feature,
                         _PRIMS, _OPS)

_HERE = os.path.dirname(os.path.abspath(__file__))

# Artifacts live off the C: drive by default (mirrors resnet_evo). On
# non-Windows hosts (the pod) "F:\..." would silently create a junk-named
# local dir, so fall back to radial_data/ there unless the env var is set.
OUT_DIR = os.environ.get("GENREG_RADIAL_DIR") or (
    r"F:\Radial" if os.name == "nt" else os.path.join(_HERE, "radial_data"))
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except OSError:
    OUT_DIR = os.path.join(_HERE, "radial_data")
    os.makedirs(OUT_DIR, exist_ok=True)

_RUNS = os.path.join(_HERE, "runs", "radial_stack")

# Phase-D energy economy constants (identical to resnet_evo).
E_DECAY, OUT_COST, RESTORE, E_GAIN, E_FLOOR, E_MAX = 0.75, 0.05, 0.04, 400.0, 0.2, 1.5
MIN_TURNOVER = 12
SATURATE_ROUNDS = 3
MIN_WINDOW_GAIN = 0.0014        # a round earning less than this => space FULL
MIN_SPACE_GAIN = 0.003          # a whole space earning less => stop stacking
MAX_SPACES = 6
FREEZE_THRESH = 0.0005

CAP_FILE = os.path.join(OUT_DIR, "cap.txt")


def _cap_thresh():
    try:
        with open(CAP_FILE) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return MIN_WINDOW_GAIN


# ---------------------------------------------------------------------------
# vector grammar genome: terms over a previous space's outputs (+ gate)
# ---------------------------------------------------------------------------

def new_vec_genome(rng, F_prev):
    order = 2 if rng.random() < 0.7 else 3
    g = {"terms": [{"c": int(rng.integers(F_prev)),
                    "prog": [(int(rng.integers(len(_PRIMS))),
                              float(rng.uniform(0.5, 2.5)),
                              float(rng.uniform(-1, 1)))
                             for _ in range(1 if rng.random() < 0.7 else 2)]}
                   for _ in range(order)],
         "op": int(rng.integers(len(_OPS)))}
    if rng.random() < 0.3:
        g["gate"] = {"c": int(rng.integers(F_prev)),
                     "prog": [(int(rng.integers(len(_PRIMS))),
                               float(rng.uniform(0.5, 2.5)),
                               float(rng.uniform(-1, 1)))],
                     "k": float(rng.uniform(0.5, 3.0))}
    else:
        g["gate"] = None
    return g


def mutate_vec(rng, g, sc, F_prev):
    c = json.loads(json.dumps(g))
    for t in c["terms"]:
        if rng.random() < 0.10:
            t["c"] = int(rng.integers(F_prev))
        prog = [list(st) for st in t["prog"]]
        for st in prog:
            if rng.random() < 0.10:
                st[0] = int(rng.integers(len(_PRIMS)))
            st[1] = float(np.clip(st[1] + rng.normal(0, sc), 0.1, 4.0))
            st[2] = float(np.clip(st[2] + rng.normal(0, sc), -2.0, 2.0))
        if rng.random() < 0.10:
            if len(prog) == 1:
                prog.append([int(rng.integers(len(_PRIMS))),
                             float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1, 1))])
            else:
                prog.pop(int(rng.integers(len(prog))))
        t["prog"] = [tuple(st) for st in prog]
    if rng.random() < 0.08:
        if len(c["terms"]) == 2:
            c["terms"].append({"c": int(rng.integers(F_prev)),
                               "prog": [(int(rng.integers(len(_PRIMS))),
                                         float(rng.uniform(0.5, 2.5)),
                                         float(rng.uniform(-1, 1)))]})
        else:
            c["terms"].pop(int(rng.integers(len(c["terms"]))))
    if rng.random() < 0.08:
        c["op"] = int(rng.integers(len(_OPS)))
    if rng.random() < 0.06:
        c["gate"] = None if c.get("gate") else \
            {"c": int(rng.integers(F_prev)),
             "prog": [(int(rng.integers(len(_PRIMS))),
                       float(rng.uniform(0.5, 2.5)), float(rng.uniform(-1, 1)))],
             "k": float(rng.uniform(0.5, 3.0))}
    gt = c.get("gate")
    if gt:
        if rng.random() < 0.10:
            gt["c"] = int(rng.integers(F_prev))
        p0 = list(gt["prog"][0])
        if rng.random() < 0.10:
            p0[0] = int(rng.integers(len(_PRIMS)))
        p0[1] = float(np.clip(p0[1] + rng.normal(0, sc), 0.1, 4.0))
        p0[2] = float(np.clip(p0[2] + rng.normal(0, sc), -2.0, 2.0))
        gt["prog"] = [tuple(p0)]
        gt["k"] = float(np.clip(gt["k"] * np.exp(rng.normal(0, sc)), 0.1, 10.0))
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


def feature_vec(torch, tp, prevF, g):
    """Vector grammar genome over prevF (N, F_prev) z-scored -> (N,)."""
    z = None
    for t in g["terms"]:
        v = prevF[:, t["c"] % prevF.shape[1]]
        for prim, a, b in t["prog"]:
            v = tp[_PRIMS[prim]](a * v + b)
        if z is None:
            z = v
        else:
            op = _OPS[g["op"]]
            z = z * v if op == "mult" else (torch.minimum(z, v) if op == "min"
                                            else torch.abs(z - v))
    gt = g.get("gate")
    if gt:
        gv = prevF[:, gt["c"] % prevF.shape[1]]
        for prim, a, b in gt["prog"]:
            gv = tp[_PRIMS[prim]](a * gv + b)
        gz = (gv - gv.mean()) / (gv.std() + 1e-9)
        z = z * torch.sigmoid(gt["k"] * gz)
    return z


# ---------------------------------------------------------------------------
# runs/ integration (identical file trio to resnet_evo)
# ---------------------------------------------------------------------------

def _record_run(cfg, hist, stats, log_lines, tags):
    try:
        ts = datetime.datetime.now()
        h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str)
                         .encode()).hexdigest()[:6]
        rid = f"{ts.strftime('%Y%m%d-%H%M%S')}-radial_stack-{h}"
        base = os.path.join(_RUNS, rid)
        os.makedirs(base, exist_ok=True)
        created = ts.isoformat(timespec="seconds")
        with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "environment": "radial_stack", "created": created,
                       "config": cfg, "status": "finished"}, f, indent=2)
        with open(os.path.join(base, "history.jsonl"), "w", encoding="utf-8") as f:
            for hrec in hist:
                f.write(json.dumps({"gen": hrec.get("round"),
                                    "fitness": hrec.get("val_acc"),
                                    "added": hrec.get("added"),
                                    "n": hrec.get("n")}) + "\n")
        with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "environment": "radial_stack", "status": "finished",
                       "finished": created, "best": stats, "checkpoint": None},
                      f, indent=2)
        with open(os.path.join(base, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"label": f"radial-stack {stats.get('n_frozen_total')} genomes "
                                f"test {stats.get('test_acc')}",
                       "favorite": False, "group": "", "tags": tags}, f, indent=2)
        with open(os.path.join(base, "report.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "kind": "radial_stack", "created": created,
                       "params": cfg, "stats": stats,
                       "log": [str(x)[:300] for x in log_lines][-200:]},
                      f, indent=2)
        return rid
    except Exception as exc:
        print(f"[radial-stack] run record failed (non-fatal): {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# one space under the energy economy, emergent cap (mirrors resnet_evo)
# ---------------------------------------------------------------------------

def _evolve_space(torch, rng, pop_size, gens, max_rounds, n_fit, Yf, yv,
                  base_prev, new_fn, mut_fn, feat_tr, log, verbose):
    dev = Yf.device
    frozen, fcols = [], []
    vals = []
    for rnd in range(max_rounds):
        cols_here = torch.stack(fcols, 1) if fcols else \
            torch.zeros((base_prev.shape[0], 0), device=dev)
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
                    sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]),
                                       0.03, 0.6))
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
        cols_here = torch.stack(fcols, 1) if fcols else \
            torch.zeros((base_prev.shape[0], 0), device=dev)
        full = torch.cat([base_prev, cols_here], 1)
        _, a1 = _ridge_soft(torch, full[:n_fit], full[n_fit:], Yf, yv)
        vals.append(float(a1))
        spg = round(starved_total / max(gens, 1), 1)
        wgain = (vals[-1] - vals[-2]) if len(vals) >= 2 else None
        thresh = _cap_thresh()
        log(f"    round {rnd:3d}  +{added} (space {len(frozen)})  val {a1:.4f}  "
            f"starved/gen {spg}"
            + (f"  d-val +{wgain:.4f} (cap {thresh:.4f})" if wgain is not None else ""),
            verbose)
        if wgain is not None and wgain < thresh:
            log(f"    space FULL at {len(frozen)} genomes — first round below cap "
                f"(+{wgain:.4f} < {thresh:.4f}); easy gains over, stacking next",
                verbose)
            break
        if os.path.exists(_STOP):
            break
    return frozen, fcols


# ---------------------------------------------------------------------------
# the stacked run
# ---------------------------------------------------------------------------

def _r0_cache_path(seed, pop_size, gens, max_rounds, n_train, n_test, smoke):
    """Space 0 is deterministic given its config, so it is trained once and
    cached. The key covers everything that shapes its evolution EXCEPT the
    live cap threshold — reusing an R0 across cap settings is exactly the
    point (identical substrate for A/B arms)."""
    key = hashlib.sha1(json.dumps(
        [int(seed), int(pop_size), int(gens), int(max_rounds),
         n_train, n_test, bool(smoke)]).encode()).hexdigest()[:10]
    return os.path.join(OUT_DIR, f"radial_stack_r0_{key}.json")


def run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5, smoke=False,
                n_train=None, n_test=None, out_path=None, record=True, verbose=True,
                rot_deg=1.0, r0_cache=True):
    """Emergent-cap stacked CIFAR grammar evolution. Space 0 = spatial
    grammar-v2 genomes over the patch-PCA environment; deeper spaces = vector
    grammar genomes (terms + gates) over the previous space's outputs. Depth
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

    spaces = []
    all_tr, all_te = [], []
    prev_tr = prev_te = None
    val_prev = 0.0

    for si in range(max_spaces):
        base_prev = (torch.stack(all_tr, 1) if all_tr
                     else torch.zeros((len(Xtr), 0), device=dev))
        if si == 0:
            new_fn = new_genome
            mut_fn = lambda r, g, sc: mutate(r, g, sc)
            feat_tr = lambda g: feature(torch, tp, env, g)
            feat_te = lambda g: feature(torch, tp, env, g, test=True)
            src = "patch-PCA maps (grammar v2)"
        else:
            mu = prev_tr.mean(0); sd = prev_tr.std(0) + 1e-6
            prevF_tr = (prev_tr - mu) / sd
            prevF_te = (prev_te - mu) / sd
            rot = si * rot_deg          # R_k rotated by k degrees (R0=0, R1=1, ...)
            rot_note = ""
            if rot:
                prevF_tr = _rotate_features(torch, prevF_tr, rot)
                prevF_te = _rotate_features(torch, prevF_te, rot)
                rot_note = f", rotated {rot}°"
            F_prev = prevF_tr.shape[1]
            new_fn = lambda r: new_vec_genome(r, F_prev)
            mut_fn = lambda r, g, sc: mutate_vec(r, g, sc, F_prev)
            feat_tr = lambda g: feature_vec(torch, tp, prevF_tr, g)
            feat_te = lambda g: feature_vec(torch, tp, prevF_te, g)
            src = f"space {si-1} outputs ({F_prev} feats{rot_note}, vector grammar + gates)"

        log(f"  [space {si}] opening — reads {src}", verbose)
        frozen = None
        r0_path = _r0_cache_path(seed, pop_size, gens, max_rounds,
                                 n_train, n_test, smoke)
        if si == 0 and r0_cache and os.path.exists(r0_path):
            with open(r0_path) as f:
                frozen = json.load(f)["genomes"]
            for g in frozen:
                g["terms"] = [{"c": t["c"], "prog": [tuple(s) for s in t["prog"]]}
                              for t in g["terms"]]
            fcols = [feat_tr(g) for g in frozen]
            log(f"  [space 0] CACHED — reused {len(frozen)} genomes from "
                f"{os.path.basename(r0_path)} (deterministic R0)", verbose)
        if frozen is None:
            frozen, fcols = _evolve_space(torch, rng, pop_size, gens, max_rounds,
                                          n_fit, Yf, yv, base_prev, new_fn, mut_fn,
                                          feat_tr, log, verbose)
            if si == 0 and r0_cache and frozen:
                tmp = r0_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"genomes": frozen,
                               "config": {"seed": seed, "pop_size": pop_size,
                                          "gens": gens, "max_rounds": max_rounds,
                                          "n_train": n_train, "n_test": n_test,
                                          "smoke": bool(smoke)}}, f)
                os.replace(tmp, r0_path)
                log(f"  [space 0] cached -> {os.path.basename(r0_path)}", verbose)
        if not frozen:
            log(f"  [space {si}] produced nothing — stop stacking", verbose)
            break
        fte = [feat_te(g) for g in frozen]
        all_tr.extend(fcols); all_te.extend(fte)
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:], Yf, yv)
        gain = val_now - val_prev
        gated = sum(1 for g in frozen if g.get("gate"))
        spaces.append({"space": si, "source": src, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4),
                       "gated": gated})
        log(f"  [space {si}] FULL: {len(frozen)} genomes ({gated} gated), "
            f"val {val_now:.4f} (+{gain:.4f}) ({round(time.time()-t0)}s)", verbose)
        spaces[-1]["genomes"] = frozen              # genomes ride in the ckpt
        tmp = os.path.join(OUT_DIR, "radial_stack_ckpt.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"spaces": spaces, "seconds": round(time.time() - t0)}, f)
        os.replace(tmp, os.path.join(OUT_DIR, "radial_stack_ckpt.json"))
        prev_tr = torch.stack(fcols, 1)
        prev_te = torch.stack(fte, 1)
        val_prev = val_now
        if si > 0 and gain < MIN_SPACE_GAIN:
            log(f"  [space {si}] gain {gain:.4f} < {MIN_SPACE_GAIN} — the stack "
                f"is done; deeper spaces can't earn their keep", verbose)
            break
        if os.path.exists(_STOP):
            log("[radial-stack] STOP lever pulled", verbose)
            break

    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
        best = max(best, acc)
    total = sum(s["n_frozen"] for s in spaces)
    out = {"phase": "radial-stack (emergent-cap stacked grammar)", "smoke": bool(smoke),
           "n_train": len(Xtr), "n_test": len(Xte),
           "n_spaces": len(spaces), "n_frozen_total": total,
           "test_acc": round(best, 4),
           "val_final": spaces[-1]["val_after"] if spaces else 0.0,
           "space_caps": [s["n_frozen"] for s in spaces],
           "rot_deg_per_space": rot_deg,
           "spaces": spaces,
           "references": {"grammar_v2_record": 0.7035,
                          "v3_freshval_tower": 0.7144,
                          "seven_seed_union": 0.7702,
                          "resnet_single_space": 0.6593},
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(OUT_DIR,
                                  "radial_stack_smoke.json" if smoke
                                  else "radial_stack_cifar.json")
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
                          [f"space {s['space']}: {s['n_frozen']} genomes "
                           f"({s['gated']} gated), val {s['val_after']} "
                           f"(+{s['val_gain']})" for s in spaces],
                          tags=(["smoke"] if smoke else []) +
                               ["radial", "gradient-free", "stacked", "emergent-cap"])
        out["run_id"] = rid
    log(f"[radial-stack] DONE: {len(spaces)} spaces {out['space_caps']} "
        f"= {total} genomes, val {out['val_final']}, TEST {best:.4f} "
        f"(records -> {op}) ({round(time.time()-t0)}s)", verbose)
    return out


if __name__ == "__main__":
    import sys
    run_stacked(smoke=("--smoke" in sys.argv))
