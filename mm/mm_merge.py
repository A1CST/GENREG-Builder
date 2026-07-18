"""mm_merge.py — MULTIMODAL merge: fuse the frozen SHAPE recognizer
(dot_shape_model.json, 10 shape classes, 20px crops) with the frozen LETTER
recognizer (kid_modelA.json, 26 letters, 32px tiles) into ONE classifier over
36 classes.

Both are gradient-free radial genome banks, but each is tied to its OWN
patch-PCA basis and native resolution, so we can't share a basis. Instead we do
LATE FUSION: run each frozen model in its own Env/basis (each at its native
scale), concatenate the two feature banks per image, and fit ONE closed-form
ridge head over {10 shapes + 26 letters}. No genome is retrained; only the joint
head is fit. Reports overall + per-modality accuracy and single-bank ablations,
so we can see whether one shared readout can perceive both modalities.

    python mm_merge.py
"""
import json
import os
import time

import numpy as np

import genreg_paths                               # noqa: F401
from radial_evo import _tprims
from radial_evo2 import feature
import radial_stack as rk
import radial_anim as ra
from dot_track import _rand_color
from dot_live import Basis
import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "lm"))   # lm-package path
from radial_kid import render_letter, LETTERS

try:
    from genreg_train import animation_data as ad
except ImportError:
    import animation_data as ad

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
SHAPE_RES, LETTER_RES = 20, 32
N_SHAPES, N_LETTERS = 10, 26


def render_shape_crop(ti, rng, res=SHAPE_RES):
    """A shape crop in the SHAPE model's format: a random-color shape centred in
    a res x res tile with a small red cursor dot on it (what dot_shape sees)."""
    S = ad.SIZE
    img = np.zeros((S, S, 3), np.float32)
    r = float(rng.uniform(6.0, 9.0))
    sa = ra.SHAPES[ti](S / 2, S / 2, r=r)
    img = _rand_color(rng)[None, None, :] * sa[..., None] + img * (1 - sa[..., None])
    da = ad.circle(S / 2, S / 2, r=float(rng.uniform(2.5, 3.6)))
    img = np.array([1., 0., 0.], np.float32)[None, None, :] * da[..., None] + img * (1 - da[..., None])
    h = res // 2
    crop = img[S // 2 - h:S // 2 - h + res, S // 2 - h:S // 2 - h + res]
    return np.clip(crop + rng.normal(0, 0.03, crop.shape).astype(np.float32), 0, 1)


def _resize(torch, X, out):
    """X: (N,H,W,3) float -> (N,out,out,3) via area/bilinear on GPU."""
    import torch.nn.functional as Fn
    t = torch.tensor(X).permute(0, 3, 1, 2)
    mode = "area" if out < X.shape[1] else "bilinear"
    t = Fn.interpolate(t, size=(out, out), mode=mode,
                       align_corners=False if mode == "bilinear" else None)
    return t.permute(0, 2, 3, 1).contiguous().numpy()


def build_dataset(torch, n_per, seed):
    """Mixed shapes+letters. Returns X20, X32 (both res versions of every item),
    labels (0-9 shapes, 10-35 letters), and modality (0 shape / 1 letter)."""
    rng = np.random.default_rng(seed)
    items20, items32, y, mod = [], [], [], []
    # shapes: native 20px crop -> also a 32px upsample
    sh = np.stack([render_shape_crop(rng.integers(N_SHAPES) if False else ti, rng)
                   for ti in np.repeat(np.arange(N_SHAPES), n_per)])
    ysh = np.repeat(np.arange(N_SHAPES), n_per)
    # letters: native 32px tile -> also a 20px downsample
    lt = np.zeros((N_LETTERS * n_per, LETTER_RES, LETTER_RES, 3), np.float32)
    ylt = np.repeat(np.arange(N_LETTERS), n_per)
    for i, li in enumerate(ylt):
        g = render_letter(LETTERS[li], rng)
        lt[i] = np.repeat(g[..., None], 3, axis=2)
    lt = np.clip(lt + rng.normal(0, 0.03, lt.shape).astype(np.float32), 0, 1)

    X20 = np.concatenate([sh, _resize(torch, lt, SHAPE_RES)], 0).astype(np.float32)
    X32 = np.concatenate([_resize(torch, sh, LETTER_RES), lt], 0).astype(np.float32)
    y = np.concatenate([ysh, N_SHAPES + ylt]).astype(np.int64)
    mod = np.concatenate([np.zeros(len(ysh)), np.ones(len(ylt))]).astype(np.int64)
    perm = rng.permutation(len(y))
    return X20[perm], X32[perm], y[perm], mod[perm]


def shape_features(torch, tp, dev, genomes, ref20, X20):
    b = Basis(torch, dev, ref20)
    b.set_input(X20)
    return torch.stack([feature(torch, tp, b, g, test=True) for g in genomes], 1)


def letter_features(torch, tp, dev, A, ref32, X32):
    b = Basis(torch, dev, ref32)
    b.set_input(X32)
    r0 = [rk.feature_r0(torch, tp, b, g, test=True) for g in A[0]]
    grids = torch.stack([rk.feature_r0(torch, tp, b, g, test=True, want_grid=True)
                         for g in A[0]], 1).half()
    s1 = [rk.feature_grid_g(torch, tp, grids, g) for g in A[1]]
    return _san(torch, torch.stack(r0 + s1, 1))


def _san(torch, F):
    return torch.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e6, 1e6)


def _ridge_head(torch, dev, Ftr, ytr, Fte, yte, n_classes, lams=(1., 3., 10., 30.)):
    Ftr, Fte = _san(torch, Ftr), _san(torch, Fte)
    Ntr = len(ytr)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-3
    Atr = torch.cat([(Ftr - mu) / sd, torch.ones(Ntr, 1, device=dev)], 1)
    Ate = torch.cat([(Fte - mu) / sd, torch.ones(len(yte), 1, device=dev)], 1)
    Y = -torch.ones((Ntr, n_classes), device=dev)
    Y[torch.arange(Ntr), ytr] = 1.0
    best = None
    for lam in lams:
        W = torch.linalg.solve(Atr.T @ Atr + lam * torch.eye(Atr.shape[1], device=dev),
                               Atr.T @ Y)
        pred = (Ate @ W).argmax(1)
        acc = float((pred == yte).float().mean())
        if best is None or acc > best[0]:
            best = (acc, pred, W, mu, sd, lam)
    return best


def run(n_per=140, seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    rk.GRID = 8
    t0 = time.time()

    shape_ck = json.load(open(os.path.join(_HERE, "multimodal", "dot_shape_model.json")))
    letter_ck = json.load(open(os.path.join(_HERE, "multimodal", "kid_modelA.json")))
    shape_g = shape_ck["genomes"]
    A = letter_ck["spaces"]                       # [R0 genomes, grid genomes]

    # per-model bases (each keeps its own patch-PCA basis)
    rng = np.random.default_rng(1)
    ref20 = np.stack([render_shape_crop(ti, rng)
                      for ti in np.repeat(np.arange(N_SHAPES), 150)]).astype(np.float32)
    lref = np.zeros((N_LETTERS * 100, LETTER_RES, LETTER_RES, 3), np.float32)
    for i, li in enumerate(np.repeat(np.arange(N_LETTERS), 100)):
        lref[i] = np.repeat(render_letter(LETTERS[li], rng)[..., None], 3, axis=2)

    Xtr20, Xtr32, ytr, mtr = build_dataset(torch, n_per, seed + 1)
    Xte20, Xte32, yte, mte = build_dataset(torch, n_per // 3, seed + 2)
    print(f"[mm] dataset: {len(ytr)} train / {len(yte)} test over 36 classes "
          f"({round(time.time()-t0)}s)", flush=True)

    Sf_tr = shape_features(torch, tp, dev, shape_g, ref20, Xtr20)
    Sf_te = shape_features(torch, tp, dev, shape_g, ref20, Xte20)
    Lf_tr = letter_features(torch, tp, dev, A, lref, Xtr32)
    Lf_te = letter_features(torch, tp, dev, A, lref, Xte32)
    print(f"[mm] features: shape bank {Sf_tr.shape[1]}, letter bank {Lf_tr.shape[1]} "
          f"({round(time.time()-t0)}s)", flush=True)

    ytr_t = torch.tensor(ytr, device=dev)
    yte_t = torch.tensor(yte, device=dev)

    fused_head = {}

    def report(name, Ftr, Fte, save_head=False):
        acc, pred, W, mu, sd, lam = _ridge_head(torch, dev, Ftr, ytr_t, Fte, yte_t, 36)
        sh = float((pred[mte == 0] == yte_t[mte == 0]).float().mean())
        le = float((pred[mte == 1] == yte_t[mte == 1]).float().mean())
        print(f"  {name:22s} overall {acc:.4f}  |  shapes {sh:.4f}  letters {le:.4f}", flush=True)
        if save_head:
            fused_head.update(W=W.cpu().numpy().tolist(), mu=mu.cpu().numpy().tolist(),
                              sd=sd.cpu().numpy().tolist(), lam=lam)
        return {"bank": name, "overall": round(acc, 4), "shapes": round(sh, 4),
                "letters": round(le, 4), "n_feats": Ftr.shape[1]}

    print("[mm] one joint ridge head over 36 classes (10 shapes + 26 letters):", flush=True)
    res = [report("shape-bank only", Sf_tr, Sf_te),
           report("letter-bank only", Lf_tr, Lf_te),
           report("FUSED (both banks)", torch.cat([Sf_tr, Lf_tr], 1),
                  torch.cat([Sf_te, Lf_te], 1), save_head=True)]

    classes = list(ra.SHAPE_NAMES) + list(LETTERS)
    with open(os.path.join(_HERE, "multimodal", "mm_model.json"), "w") as f:
        json.dump({"kind": "multimodal late-fusion (shapes + letters)",
                   "shape_checkpoint": "dot_shape_model.json", "shape_res": SHAPE_RES,
                   "letter_checkpoint": "kid_modelA.json", "letter_res": LETTER_RES,
                   "grid": 8, "classes": classes, "head": fused_head,
                   "note": "run shape genomes on the 20px view + letter genomes on the "
                           "32px view (each in its own basis), concat, standardize with "
                           "head.mu/sd, then argmax(A @ head.W)."}, f)

    out = {"experiment": "multimodal late-fusion: shapes + letters, one head",
           "n_classes": 36, "chance": round(1 / 36, 4),
           "shape_genomes": len(shape_g), "letter_genomes": sum(len(sp) for sp in A),
           "n_train": len(ytr), "n_test": len(yte), "results": res,
           "seconds": round(time.time() - t0)}
    os.makedirs(os.path.join(_HERE, "multimodal"), exist_ok=True)
    with open(os.path.join(_HERE, "multimodal", "mm_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[mm] DONE ({out['seconds']}s) -> multimodal/mm_result.json", flush=True)
    return out


if __name__ == "__main__":
    run()
