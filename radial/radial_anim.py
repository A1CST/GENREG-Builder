"""radial_anim.py — TEMPORAL radial stack for the Animation tab.

The task is the ANIMATION DATASET's motion paths (genreg_train/
animation_data.py: line, diagonal, swoop, loop, figure8, zigzag, wave,
spiral, bounce, scurve). A sequence is a T-frame window of one of those
paths — but the SHAPE riding the path is random (any of the ten shapes),
the window starts anywhere in the clip, and the whole path is shifted by a
random offset. So no single frame answers: shape is a decoy and position
is decorrelated — only the MOTION between frames names the animation.
Label = which path, 10 classes, chance 0.10.

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
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, make_scorer, new_genome, mutate
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 6                                   # frames per sequence window

try:
    from genreg_train import animation_data as ad
except ImportError:                     # pod: flat copy of the module
    import animation_data as ad

PATHS = [(name, path) for name, path, _shape in ad.ANIMATIONS]
SHAPES = [shape for _name, _path, shape in ad.ANIMATIONS]
PATH_NAMES = [n for n, _ in PATHS]
SHAPE_NAMES = [s.__name__ for s in SHAPES]
N_CLASSES = len(PATHS)                  # 10 — one per animation / per shape


def _downscale(X, res):
    """Downscale composited frames (n, T, S, S, 3) to (n, T, res, res, 3) by
    area averaging — same scene, sampled at a different resolution."""
    S = X.shape[2]
    if res == S:
        return X
    import torch
    import torch.nn.functional as Fn
    n, Tn, _, _, C = X.shape
    Xt = torch.from_numpy(np.ascontiguousarray(X)).reshape(n * Tn, S, S, C).permute(0, 3, 1, 2)
    Xt = Fn.interpolate(Xt, size=(res, res), mode="area")
    return Xt.permute(0, 2, 3, 1).reshape(n, Tn, res, res, C).contiguous().numpy()


def make_anim_data(n_train=7500, n_test=1875, seed=0, noise=0.05,
                   path=os.path.join(_HERE, "radial_data", "anim_seq.npz"),
                   bg="black", res=32):
    """Sequences from the REAL animation dataset's motion paths.

    Per sequence: path class = the label; shape = random (decoy); window =
    T consecutive frames starting anywhere in the 24-frame clip; the whole
    path is shifted by a random per-sequence offset (position decorrelated
    from path identity). 64x64 renders are 2x2 mean-pooled to 32x32.

    bg="black": the original solid-black background.
    bg="randcolor": each FRAME gets an independent random solid RGB color as
        its background (no order, uncorrelated with the label); the white shape
        is alpha-composited over it. A pure distractor the model must ignore."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = rng.integers(0, N_CLASSES, n)
    y_shape = np.zeros(n, np.int64)
    S = ad.SIZE                                     # composite at native 64
    X = np.zeros((n, T, S, S, 3), np.float32)       # RGB
    F = ad.FRAMES
    for i in range(n):
        pfn = PATHS[y[i]][1]
        y_shape[i] = rng.integers(0, len(SHAPES))   # independent of the path
        sfn = SHAPES[y_shape[i]]
        s = rng.integers(0, F - T + 1)              # window start
        oy, ox = rng.uniform(-6, 6), rng.uniform(-6, 6)
        for f in range(T):
            t = (s + f) / (F - 1)
            cx, cy = pfn(t)
            cx = float(np.clip(cx + ox, 4, ad.SIZE - 4))
            cy = float(np.clip(cy + oy, 4, ad.SIZE - 4))
            a = sfn(cx, cy)                          # (S, S) shape alpha
            col = (rng.uniform(0, 1, 3).astype(np.float32) if bg == "randcolor"
                   else np.zeros(3, np.float32))                 # per-frame bg color
            # white shape composited over the background color
            X[i, f] = col[None, None, :] * (1.0 - a[..., None]) + a[..., None]
    X = _downscale(X, res)                          # -> (n, T, res, res, 3)
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    X8 = (X * 255).astype(np.uint8)
    np.savez(path,
             Xtr=X8[:n_train], ytr=y[:n_train].astype(np.int64),
             Xte=X8[n_train:], yte=y[n_train:].astype(np.int64),
             ytr_shape=y_shape[:n_train], yte_shape=y_shape[n_train:])
    print(f"anim_seq[bg={bg},res={res}]: {n_train}/{n_test} x {T}-frame windows, both labels "
          f"(path + shape, {N_CLASSES} classes each) -> {path} "
          f"(path balance {np.bincount(y, minlength=N_CLASSES).tolist()})",
          flush=True)
    return path


_SIZE_RANGE = {"fixed": (1.0, 1.0), "rand": (0.5, 1.8),
               "small": (0.4, 0.7), "large": (1.4, 2.0)}


def sample_seqs(n=6, bg="randcolor", seed=7, noise=0.05, size="fixed", res=32):
    """A handful of sample sequences (RGB uint8, (n,T,32,32,3)) for page
    previews — the SAME renderer as make_anim_data, but nothing is written to
    disk. `size` scales the shape radius per sequence. Returns (X8, y_path,
    y_shape)."""
    rng = np.random.default_rng(seed)
    lo, hi = _SIZE_RANGE.get(size, (1.0, 1.0))
    y = rng.integers(0, N_CLASSES, n)
    y_shape = rng.integers(0, len(SHAPES), n)
    S = ad.SIZE                                     # composite at native 64
    X = np.zeros((n, T, S, S, 3), np.float32)
    F = ad.FRAMES
    for i in range(n):
        pfn = PATHS[y[i]][1]
        sfn = SHAPES[y_shape[i]]
        s = rng.integers(0, F - T + 1)
        oy, ox = rng.uniform(-6, 6), rng.uniform(-6, 6)
        r = ad.RADIUS * float(rng.uniform(lo, hi))
        for f in range(T):
            t = (s + f) / (F - 1)
            cx, cy = pfn(t)
            cx = float(np.clip(cx + ox, 4, ad.SIZE - 4))
            cy = float(np.clip(cy + oy, 4, ad.SIZE - 4))
            a = sfn(cx, cy, r=r)
            if bg == "inv":                               # white bg, black shape
                X[i, f] = np.repeat((1.0 - a)[..., None], 3, axis=2)
            else:                                         # white shape over bg
                col = (rng.uniform(0, 1, 3).astype(np.float32) if bg == "randcolor"
                       else np.zeros(3, np.float32))
                X[i, f] = col[None, None, :] * (1.0 - a[..., None]) + a[..., None]
    X = _downscale(X, res)                          # -> (n, T, res, res, 3)
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    return (X * 255).astype(np.uint8), y, y_shape


def run(pop_size=64, gens=12, max_rounds=200, seed=5, max_spaces=16,
        grid_size=8, out_path=None, verbose=True, task="path",
        data_path=None, out_tag=""):
    """task="path": which animation (motion is the answer, shape the decoy).
    task="shape": which shape (shape is the answer, motion the decoy).
    Same sequences either way — only the label column flips."""
    import torch
    rk.GRID = int(grid_size)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    suffix = "" if task == "path" else "_shape"
    class_names = PATH_NAMES if task == "path" else SHAPE_NAMES
    z = np.load(data_path or os.path.join(_HERE, "radial_data", "anim_seq.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0        # (N, T, 32, 32, 3)
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr" + suffix], z["yte" + suffix]
    Ntr, Nte = len(ytr), len(yte)
    RES = Xtr.shape[2]                                # resolution inferred from data
    # every frame is an environment row; sequence n frame f = row n*T + f
    env = Env(torch, dev, Xtr.reshape(Ntr * T, RES, RES, 3),
              Xte.reshape(Nte * T, RES, RES, 3), max_cached=6)

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
    space_genomes = []
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
        space_genomes.append(frozen)
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
    out = {"phase": "radial-anim (temporal stack)", "frames": T, "res": RES,
           "n_train": Ntr, "n_test": Nte, "n_classes": N_CLASSES,
           "chance": round(1.0 / N_CLASSES, 4),
           "n_spaces": len(spaces), "space_caps": [s["n_frozen"] for s in spaces],
           "test_acc": round(best, 4),
           "val_final": spaces[-1]["val_after"] if spaces else 0.0,
           "spaces": spaces, "grid": G, "classes": class_names,
           "label": task,
           "task": ("which motion path (the animation dataset's 10 clips); "
                    "shape random, window random, position offset random — "
                    "only the motion between frames answers") if task == "path"
                   else ("which SHAPE rides the path; motion path, window and "
                         "offset random — the decoys and the answer swap"),
           "seconds": round(time.time() - t0)}
    out["bg"] = "randcolor" if (data_path and "bgcolor" in str(data_path)) else "black"
    op = out_path or os.path.join(_HERE, "radial_data",
                                  f"anim_radial{suffix}{out_tag}.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    # full model checkpoint: every space's genomes (structure only — heads are
    # refit wherever the model is loaded, so PCA sign conventions can't break it)
    model = {"frames": T, "grid": G, "res": RES, "spaces": space_genomes, "label": task}
    with open(os.path.join(_HERE, "radial_data",
                           f"anim_model{suffix}{out_tag}.json"), "w") as f:
        json.dump(model, f)
    print(f"[radial-anim] DONE: {len(spaces)} spaces {out['space_caps']}, "
          f"val {out['val_final']}, TEST {best:.4f} -> {op} "
          f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if not os.path.exists(os.path.join(_HERE, "radial_data", "anim_seq.npz")) \
            or "--regen" in sys.argv:
        make_anim_data()
    run(task="shape" if "shape" in sys.argv else "path")
