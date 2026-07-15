"""radial_anim.py — TEMPORAL radial stack for the Animation tab.

The task: 6-frame animations of a shape that GROWS (rate is a class) and
either drifts or holds still (motion is a class). Start size, position, and
shape identity are randomized, so no single frame answers — the value is
carried BETWEEN frames. Label = growth-rate(4) x moving(2) = 8 classes.

The wiring (current best radial setup, made temporal):
  R0  — grammar-v2 spatial genomes evolve on ALL frames at once (the env
        holds every frame as a row); a genome's fitness column is its
        per-frame scalar averaged over the sequence. R0 becomes per-frame
        perception (size/shape/position detectors).
  hand-off — each R0 genome emits its GRID map for EVERY frame; the R1
        channel bank is laid out (genome, frame) so the existing universal
        grammar (shift genes, folds, gates, residual arch, moment stats)
        composes ACROSS TIME exactly the way it composes across space:
        |com(genome g @ frame 5) - com(genome g @ frame 0)| IS growth.
  R1+ — the standard emergent-cap stacked spaces over that temporal bank.

No gradients anywhere. Test touched once. Exports radial_data/anim_radial.json
for the Animation page.
"""
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, make_scorer, new_genome, mutate
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))
T = 6                                   # frames per sequence
RATES = [0.35, 0.85, 1.35, 1.85]        # growth px/frame — the temporal signal


def make_anim_data(n_train=7500, n_test=1875, seed=0, noise=0.07,
                   path=os.path.join(_HERE, "radial_data", "anim_seq.npz")):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    rate_c = rng.integers(0, 4, n)
    moving = rng.integers(0, 2, n)
    y = rate_c * 2 + moving             # 8 classes, all temporal
    X = np.zeros((n, T, 32, 32), np.float32)
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    for i in range(n):
        r = rng.uniform(3.0, 6.0)
        cy, cx = rng.uniform(10, 22), rng.uniform(10, 22)
        ring = rng.random() < 0.5       # shape identity: distractor
        if moving[i]:
            ang = rng.uniform(0, 2 * np.pi)
            dy, dx = 2.0 * np.sin(ang), 2.0 * np.cos(ang)
        else:
            dy = dx = 0.0
        for f in range(T):
            rad = r + RATES[rate_c[i]] * f
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            if ring:
                X[i, f] = ((d > rad - 1.6) & (d < rad + 1.6)).astype(np.float32)
            else:
                X[i, f] = (d < rad).astype(np.float32)
            cy += dy; cx += dx
            cy = float(np.clip(cy, 6, 26)); cx = float(np.clip(cx, 6, 26))
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    X8 = (np.repeat(X[..., None], 3, axis=4) * 255).astype(np.uint8)
    np.savez(path,
             Xtr=X8[:n_train], ytr=y[:n_train].astype(np.int64),
             Xte=X8[n_train:], yte=y[n_train:].astype(np.int64))
    print(f"anim_seq: {n_train}/{n_test} x {T} frames -> {path} "
          f"(class balance {np.bincount(y, minlength=8).tolist()})", flush=True)
    return path


def run(pop_size=64, gens=12, max_rounds=200, seed=5, max_spaces=16,
        grid_size=8, out_path=None, verbose=True):
    import torch
    rk.GRID = int(grid_size)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    z = np.load(os.path.join(_HERE, "radial_data", "anim_seq.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0        # (N, T, 32, 32, 3)
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    Ntr, Nte = len(ytr), len(yte)
    # every frame is an environment row; sequence n frame f = row n*T + f
    env = Env(torch, dev, Xtr.reshape(Ntr * T, 32, 32, 3),
              Xte.reshape(Nte * T, 32, 32, 3), max_cached=6)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, 10), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    def log(m, v=True):
        if v:
            print(m, flush=True)

    G = rk.GRID
    spaces, all_tr, all_te = [], [], []
    prev_grid_tr = prev_grid_te = None
    val_prev = 0.0

    for si in range(max_spaces):
        base_prev = (torch.stack(all_tr, 1) if all_tr
                     else torch.zeros((Ntr, 0), device=dev))
        if si == 0:
            # R0: spatial genome on every frame; column = per-sequence mean
            new_fn = new_genome
            mut_fn = lambda r, g, sc: mutate(r, g, sc)
            feat_tr = lambda g: rk.feature_r0(torch, tp, env, g).view(Ntr, T).mean(1)
            feat_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True).view(Nte, T).mean(1)
            grid_tr_of = lambda g: rk.feature_r0(torch, tp, env, g,
                                                 want_grid=True).view(Ntr, T, G, G)
            grid_te_of = lambda g: rk.feature_r0(torch, tp, env, g, test=True,
                                                 want_grid=True).view(Nte, T, G, G)
            src = f"patch-PCA maps of ALL {T} frames (per-frame perception)"
        else:
            C_prev = prev_grid_tr.shape[1]
            new_fn = lambda r: rk.new_grid_genome(r, C_prev)
            mut_fn = lambda r, g, sc: rk.mutate_grid_g(r, g, sc, C_prev)
            feat_tr = lambda g: rk.feature_grid_g(torch, tp, prev_grid_tr, g)
            feat_te = lambda g: rk.feature_grid_g(torch, tp, prev_grid_te, g)
            grid_tr_of = lambda g: rk.feature_grid_g(torch, tp, prev_grid_tr, g,
                                                     want_grid=True)
            grid_te_of = lambda g: rk.feature_grid_g(torch, tp, prev_grid_te, g,
                                                     want_grid=True)
            src = (f"space {si-1}: {C_prev} channels (genome x FRAME layout — "
                   f"folds compose across time)")

        log(f"  [space {si}] opening — reads {src}")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens, max_rounds,
                                         n_fit, Yf, yv, base_prev, new_fn, mut_fn,
                                         feat_tr, log, verbose)
        if not frozen:
            log(f"  [space {si}] produced nothing — stop")
            break
        fte = [feat_te(g) for g in frozen]
        all_tr.extend(fcols); all_te.extend(fte)
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:], Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "source": src, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4)})
        log(f"  [space {si}] FULL: {len(frozen)} genomes, val {val_now:.4f} "
            f"(+{gain:.4f}) ({round(time.time()-t0)}s)")
        if si == 0:
            # temporal hand-off: every genome's map for EVERY frame is a channel
            g_tr = [grid_tr_of(g) for g in frozen]         # each (Ntr, T, G, G)
            g_te = [grid_te_of(g) for g in frozen]
            prev_grid_tr = torch.cat(g_tr, 1).half()       # (Ntr, n*T, G, G)
            prev_grid_te = torch.cat(g_te, 1).half()
            log(f"  [hand-off] temporal bank: {prev_grid_tr.shape[1]} channels "
                f"({len(frozen)} genomes x {T} frames)")
        else:
            prev_grid_tr = torch.stack([grid_tr_of(g) for g in frozen], 1).half()
            prev_grid_te = torch.stack([grid_te_of(g) for g in frozen], 1).half()
        val_prev = val_now
        if si > 0 and gain < rk.MIN_SPACE_GAIN:
            log(f"  [space {si}] gain {gain:.4f} < {rk.MIN_SPACE_GAIN} — done")
            break
        if os.path.exists(_STOP):
            log("[radial-anim] STOP lever pulled")
            break

    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
        best = max(best, acc)
    out = {"phase": "radial-anim (temporal stack)", "frames": T,
           "n_train": Ntr, "n_test": Nte, "n_classes": 8, "chance": 0.125,
           "n_spaces": len(spaces), "space_caps": [s["n_frozen"] for s in spaces],
           "test_acc": round(best, 4),
           "val_final": spaces[-1]["val_after"] if spaces else 0.0,
           "spaces": spaces, "grid": G,
           "task": "growth-rate(4) x moving(2); start size/position/shape random"
                   " — value carried between frames",
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(_HERE, "radial_data", "anim_radial.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[radial-anim] DONE: {len(spaces)} spaces {out['space_caps']}, "
          f"val {out['val_final']}, TEST {best:.4f} -> {op} "
          f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if not os.path.exists(os.path.join(_HERE, "radial_data", "anim_seq.npz")) \
            or "--regen" in sys.argv:
        make_anim_data()
    run()
