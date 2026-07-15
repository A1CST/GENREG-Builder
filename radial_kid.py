"""radial_kid.py - the ENGLISH CURRICULUM: models learn language like a
kid (user's design). Each stage is its own model, trained on top of the
previous stage's frozen genomes (the continue-train warm start):

  Stage A  letters  - identify single rendered letters (26 classes).
                      Pure perception; the gate is near-perfect accuracy.
  Stage B  words    - a word is a STRIP of letter tiles; identify the
                      word (V classes) with stage A frozen as the eye.
  Stage C  cloze    - short phrase with a blank; name the missing word.
  Stage D  autoregression - next word, on top of C.

No stage advances until the one below earns its gate. Pixels are the
whole environment; no embeddings, tables, or one-hots anywhere in the
curriculum. Stage A here.

    python radial_kid.py A
"""
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, new_genome, mutate
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(_HERE, "radial_data")
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _font(size, cache={}):
    from PIL import ImageFont
    if size not in cache:
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     "arial.ttf"):
            try:
                cache[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            cache[size] = ImageFont.load_default()
    return cache[size]


def render_letter(ch, rng, size=None):
    """One 32x32 letter tile: random size, position jitter, noise later."""
    from PIL import Image, ImageDraw
    im = Image.new("L", (32, 32), 0)
    d = ImageDraw.Draw(im)
    f = _font(int(size or rng.integers(14, 27)))
    x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=f)
    dx = int(rng.integers(-4, 5))
    dy = int(rng.integers(-4, 5))
    d.text(((32 - (x1 - x0)) / 2 - x0 + dx, (32 - (y1 - y0)) / 2 - y0 + dy),
           ch, fill=255, font=f)
    return np.asarray(im, np.float32) / 255.0


def make_letters(n_train=20000, n_test=5000, seed=0, noise=0.05,
                 path=os.path.join(RD, "kid_letters.npz")):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = rng.integers(0, 26, n)
    X = np.zeros((n, 32, 32), np.float32)
    for i in range(n):
        X[i] = render_letter(LETTERS[y[i]], rng)
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    X8 = (np.repeat(X[..., None], 3, axis=3) * 255).astype(np.uint8)
    np.savez(path, Xtr=X8[:n_train], ytr=y[:n_train].astype(np.int64),
             Xte=X8[n_train:], yte=y[n_train:].astype(np.int64))
    print(f"kid_letters: {n_train}/{n_test} single-letter tiles "
          f"(size 14-26px, jitter +-4, noise {noise}) -> {path}", flush=True)
    return path


def stage_a(pop_size=64, gens=12, max_rounds=200, seed=5, max_spaces=6,
            grid_size=8, verbose=True):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    rk.GRID = int(grid_size)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    z = np.load(os.path.join(RD, "kid_letters.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    Ntr, Nte = len(ytr), len(yte)
    env = Env(torch, dev, Xtr, Xte, max_cached=6)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 26), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, 26), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    log_lines = []

    def log(m, v=True):
        log_lines.append(m)
        if v:
            print(m, flush=True)

    G = rk.GRID
    spaces, all_tr, all_te, space_genomes = [], [], [], []
    prev_tr = prev_te = None
    val_prev = 0.0
    for si in range(max_spaces):
        base_prev = (torch.stack(all_tr, 1) if all_tr
                     else torch.zeros((Ntr, 0), device=dev))
        if si == 0:
            new_fn = new_genome
            mut_fn = lambda r, g, sc: mutate(r, g, sc)
            feat_tr = lambda g: rk.feature_r0(torch, tp, env, g)
            feat_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True)
            g_tr = lambda g: rk.feature_r0(torch, tp, env, g, want_grid=True)
            g_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True,
                                           want_grid=True)
        else:
            C = prev_tr.shape[1]
            new_fn = lambda r: rk.new_grid_genome(r, C)
            mut_fn = lambda r, g, sc: rk.mutate_grid_g(r, g, sc, C)
            feat_tr = lambda g: rk.feature_grid_g(torch, tp, prev_tr, g)
            feat_te = lambda g: rk.feature_grid_g(torch, tp, prev_te, g)
            g_tr = lambda g: rk.feature_grid_g(torch, tp, prev_tr, g,
                                               want_grid=True)
            g_te = lambda g: rk.feature_grid_g(torch, tp, prev_te, g,
                                               want_grid=True)
        log(f"  [A space {si}] opening")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_tr, log, verbose)
        if not frozen:
            break
        space_genomes.append(frozen)
        all_tr.extend(fcols)
        all_te.extend(feat_te(g) for g in frozen)
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                                 Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4)})
        log(f"  [A space {si}] FULL: {len(frozen)} genomes, val "
            f"{val_now:.4f} (+{gain:.4f}) ({round(time.time()-t0)}s)")
        if si == 0:
            prev_tr = torch.cat([g_tr(g).unsqueeze(1) if g_tr(g).dim() == 3
                                 else g_tr(g) for g in []], 1) \
                if False else torch.stack([g_tr(g) for g in frozen], 1).half()
            prev_te = torch.stack([g_te(g) for g in frozen], 1).half()
        else:
            prev_tr = torch.stack([g_tr(g) for g in frozen], 1).half()
            prev_te = torch.stack([g_te(g) for g in frozen], 1).half()
        val_prev = val_now
        if si > 0 and gain < rk.MIN_SPACE_GAIN:
            break
        if os.path.exists(_STOP):
            break

    if not all_tr:
        print("[kid A] nothing earned - stop", flush=True)
        return None
    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best1 = 0.0
    for lam in (1.0, 3.0, 10.0):
        _, a = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
        best1 = max(best1, a)

    from anim_infer import count_params
    gp = sum(count_params(g) for sp in space_genomes for g in sp)
    n_genomes = sum(len(sp) for sp in space_genomes)
    out = {"phase": "curriculum stage A: letters", "n_classes": 26,
           "chance": round(1 / 26, 4), "n_train": Ntr, "n_test": Nte,
           "test_acc": round(best1, 4),
           "val_final": spaces[-1]["val_after"] if spaces else 0,
           "n_spaces": len(spaces),
           "space_caps": [s["n_frozen"] for s in spaces],
           "spaces": spaces, "n_genomes": n_genomes, "genome_params": gp,
           "head_params": (n_genomes + 1) * 26,
           "total_params": gp + (n_genomes + 1) * 26,
           "task": "identify single rendered letters (26 classes; size "
                   "14-26px, position jitter, brightness + noise). The "
                   "curriculum's first gate: the eye must work before "
                   "words exist.",
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "kid_stageA.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, "kid_modelA.json"), "w") as f:
        json.dump({"grid": G, "spaces": space_genomes,
                   "label": "letter", "stage": "A"}, f)
    rk._record_run(
        {"env": "kid-stageA", "n_classes": 26, "n_train": Ntr,
         "pop_size": pop_size, "gens": gens, "seed": seed},
        [{"round": s["space"], "added": s["n_frozen"],
          "val_acc": s["val_after"], "n": s["n_frozen"]} for s in spaces],
        {"test_acc": round(best1, 4), "n_frozen_total": n_genomes,
         "total_params": out["total_params"]},
        log_lines, ["lm", "kid", "curriculum", "radial"])
    print(f"[kid A] DONE: {len(spaces)} spaces {out['space_caps']}, "
          f"TEST {best1:.4f} ({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if not os.path.exists(os.path.join(RD, "kid_letters.npz")) \
            or "--regen" in sys.argv:
        make_letters()
    stage_a()
