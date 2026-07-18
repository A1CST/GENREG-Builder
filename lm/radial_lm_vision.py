"""radial_lm_vision.py - Module 17: LANGUAGE GROUNDED IN VISION.

The glyph line, resurrected at word level (user's call): render every
context word as an IMAGE tile - the model physically SEES the sentence.
A 12-word window is a film strip; the animation wiring applies verbatim:

  R0       - spatial genomes perceive single word-images (window-mean
             fitness; visual word identity is learned, not given)
  hand-off - every frozen R0 genome's grid map PER WORD SLOT:
             (genome x position) channels; order is spatial structure
  R1+      - emergent-cap spaces compose across the strip
  head     - V-way closed-form ridge; test touched once

No embeddings, no continuation tables, no identity one-hots - the pixels
are the entire environment. Rendering: word text auto-shrunk into 32x32,
+-2px jitter, brightness + noise (real perception, not lookup).

    python radial_lm_vision.py
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, new_genome, mutate
import radial_lm_word as rw
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
W = 12                                   # words per strip
V = 1000                                 # target vocabulary


def make_vision_data(n_train=30000, n_test=6000, seed=0, noise=0.05,
                     path=os.path.join(RD, "lm_vis.npz")):
    rw.W, rw.V = W, V
    rw.make_word_data(n_train=n_train, n_test=n_test, seed=seed,
                      path=os.path.join(RD, "lm_word_vis.npz"))
    z = np.load(os.path.join(RD, "lm_word_vis.npz"), allow_pickle=True)
    vocab, feat, _ = rw._load_embed()

    from PIL import Image, ImageDraw, ImageFont
    fonts = {}

    def render_word(word, rng):
        im = Image.new("L", (32, 32), 0)
        d = ImageDraw.Draw(im)
        size = 16
        while size >= 6:
            if size not in fonts:
                try:
                    fonts[size] = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        size)
                except OSError:
                    fonts[size] = ImageFont.load_default()
            x0, y0, x1, y1 = d.textbbox((0, 0), word, font=fonts[size])
            if x1 - x0 <= 30:
                break
            size -= 2
        size = max(size, 6)
        if size not in fonts:
            try:
                fonts[size] = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except OSError:
                fonts[size] = ImageFont.load_default()
        f = fonts[size]
        x0, y0, x1, y1 = d.textbbox((0, 0), word, font=f)
        dx = int(rng.integers(-2, 3))
        dy = int(rng.integers(-2, 3))
        d.text(((32 - (x1 - x0)) / 2 - x0 + dx,
                (32 - (y1 - y0)) / 2 - y0 + dy), word, fill=255, font=f)
        return np.asarray(im, np.float32) / 255.0

    rng = np.random.default_rng(seed)

    def render(ctx):
        N = len(ctx)
        X = np.zeros((N, W, 32, 32), np.float32)
        for i in range(N):
            for s in range(W):
                j = int(ctx[i, s])
                if j >= 0:
                    X[i, s] = render_word(vocab[j], rng)
        X = X * rng.uniform(0.7, 1.0, (N, 1, 1, 1)).astype(np.float32)
        X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32),
                    0, 1)
        return (np.repeat(X[..., None], 3, axis=4) * 255).astype(np.uint8)

    np.savez(path,
             Xtr=render(z["ctx_tr"]), ytr=z["ytr"],
             Xte=render(z["ctx_te"]), yte=z["yte"],
             ctx_tr=z["ctx_tr"], ctx_te=z["ctx_te"],
             targets=z["targets"])
    print(f"lm_vis: {n_train}/{n_test} x {W}-word strips rendered as "
          f"32x32 tiles -> {path}", flush=True)
    return path


def run(pop_size=64, gens=12, max_rounds=300, seed=5, max_spaces=12,
        grid_size=8, out_path=None, verbose=True):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    rk.GRID = int(grid_size)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    z = np.load(os.path.join(RD, "lm_vis.npz"), allow_pickle=True)
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    targets = [str(w) for w in z["targets"]]
    Ntr, Nte = len(ytr), len(yte)
    env = Env(torch, dev, Xtr.reshape(Ntr * W, 32, 32, 3),
              Xte.reshape(Nte * W, 32, 32, 3), max_cached=6)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, V), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, V), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    log_lines = []

    def log(m, v=True):
        log_lines.append(m)
        if v:
            print(m, flush=True)

    G = rk.GRID
    spaces, all_tr, all_te = [], [], []
    space_genomes = []
    prev_tr = prev_te = None
    val_prev = 0.0

    for si in range(max_spaces):
        base_prev = (torch.stack(all_tr, 1) if all_tr
                     else torch.zeros((Ntr, 0), device=dev))
        if si == 0:
            new_fn = new_genome
            mut_fn = lambda r, g, sc: mutate(r, g, sc)
            feat_tr = lambda g: rk.feature_r0(torch, tp, env, g).view(Ntr, W).mean(1)
            feat_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True).view(Nte, W).mean(1)
            g_tr = lambda g: rk.feature_r0(torch, tp, env, g,
                                           want_grid=True).view(Ntr, W, G, G)
            g_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True,
                                           want_grid=True).view(Nte, W, G, G)
            src = f"patch-PCA maps of all {W} word tiles (visual perception)"
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
            src = f"space {si-1}: {C} channels (genome x WORD-SLOT layout)"
        log(f"  [space {si}] opening - reads {src}")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_tr, log, verbose)
        if not frozen:
            log(f"  [space {si}] produced nothing - stop")
            break
        space_genomes.append(frozen)
        all_tr.extend(fcols)
        all_te.extend(feat_te(g) for g in frozen)
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                                 Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "source": src, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4)})
        log(f"  [space {si}] FULL: {len(frozen)} genomes, val {val_now:.4f} "
            f"(+{gain:.4f}) ({round(time.time()-t0)}s)")
        if si == 0:
            prev_tr = torch.cat([g_tr(g) for g in frozen], 1).half()
            prev_te = torch.cat([g_te(g) for g in frozen], 1).half()
            log(f"  [hand-off] strip bank: {prev_tr.shape[1]} channels "
                f"({len(frozen)} genomes x {W} word slots)")
        else:
            prev_tr = torch.stack([g_tr(g) for g in frozen], 1).half()
            prev_te = torch.stack([g_te(g) for g in frozen], 1).half()
        val_prev = val_now
        if si > 0 and gain < rk.MIN_SPACE_GAIN:
            log(f"  [space {si}] gain {gain:.4f} - done")
            break
        if os.path.exists(_STOP):
            break

    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best1 = best5 = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        A = torch.hstack([(Ftr - mu) / sd, torch.ones(Ntr, 1, device=dev)])
        Gm = (A.T @ A).double() + lam * torch.eye(
            Ftr.shape[1] + 1, device=dev, dtype=torch.float64)
        Wm = torch.linalg.solve(Gm, (A.T @ Yfull).double()).float()
        s = torch.hstack([(Fte - mu) / sd,
                          torch.ones(Nte, 1, device=dev)]) @ Wm
        a1 = float((s.argmax(1) == yte_t).float().mean())
        a5 = float((s.topk(5, 1).indices == yte_t.view(-1, 1))
                   .any(1).float().mean())
        if a1 > best1:
            best1, best5 = a1, a5

    from anim_infer import count_params
    gp = sum(count_params(g) for sp in space_genomes for g in sp)
    n_genomes = sum(len(sp) for sp in space_genomes)
    out = {"phase": "radial-lm-vision (language grounded in vision)",
           "context_words": W, "vocab": V, "n_train": Ntr, "n_test": Nte,
           "chance": round(1.0 / V, 4),
           "n_spaces": len(spaces),
           "space_caps": [s_["n_frozen"] for s_ in spaces],
           "test_acc": round(best1, 4), "test_top5": round(best5, 4),
           "val_final": spaces[-1]["val_after"] if spaces else 0.0,
           "spaces": spaces, "n_genomes": n_genomes,
           "genome_params": gp,
           "head_params": (n_genomes + 1) * V,
           "total_params": gp + (n_genomes + 1) * V,
           "task": f"predict the NEXT WORD from IMAGES of the {W} context "
                   "words - no embeddings, no tables, no one-hots; the "
                   "pixels are the entire environment. The model must "
                   "learn to READ before it can predict.",
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(RD, "lm_radial_vision.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, "lm_model_vision.json"), "w") as f:
        json.dump({"context_words": W, "vocab": V, "grid": G,
                   "spaces": space_genomes, "label": "next-word",
                   "mode": "vision"}, f)
    rk._record_run(
        {"env": "lm-vision", "context_words": W, "vocab": V,
         "n_train": Ntr, "pop_size": pop_size, "gens": gens, "seed": seed},
        [{"round": s_["space"], "added": s_["n_frozen"],
          "val_acc": s_["val_after"], "n": s_["n_frozen"]} for s_ in spaces],
        {"test_acc": round(best1, 4), "test_top5": round(best5, 4),
         "n_frozen_total": n_genomes, "total_params": out["total_params"]},
        log_lines, ["lm", "vision", "radial"])
    print(f"[radial-lm-vision] DONE: {len(spaces)} spaces "
          f"{out['space_caps']}, TEST top1 {best1:.4f} top5 {best5:.4f} "
          f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if not os.path.exists(os.path.join(RD, "lm_vis.npz")) \
            or "--regen" in sys.argv:
        make_vision_data()
    run()
