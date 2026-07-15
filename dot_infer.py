"""dot_infer.py — run the trained cursor tracker (dot_model.json) on MOVING
cursor sequences (a red cursor pinned to a shape gliding along a path) and export
a demo the page can animate: per-frame RGB frames + the true cursor position +
the model's PREDICTED position. Shows Model 1 following the cursor.
Exports radial_data/dot_cursor_demo.json.
"""
import base64
import json
import os

import numpy as np

from radial_evo import _tprims
from radial_evo2 import Env, feature
import radial_anim as ra
from dot_track import gen_dot, _DCOL

try:
    from genreg_train import animation_data as ad
except ImportError:
    import animation_data as ad

_HERE = os.path.dirname(os.path.abspath(__file__))
T = 8


def gen_cursor_seq(n, res, seed=0, noise=0.03, distractors=0):
    """n sequences (n,T,res,res,3 uint8): a shape glides along a random path with
    the red cursor centred on it, over `distractors` STATIC non-red distractor
    shapes; pos (n,T,2) normalised cursor positions."""
    rng = np.random.default_rng(seed)
    S = ad.SIZE
    F = ad.FRAMES
    X = np.zeros((n, T, S, S, 3), np.float32)
    pos = np.zeros((n, T, 2), np.float32)
    red = np.array([1.0, 0.0, 0.0], np.float32)
    for i in range(n):
        pfn = ra.PATHS[int(rng.integers(len(ra.PATHS)))][1]
        sfn = ra.SHAPES[int(rng.integers(len(ra.SHAPES)))]
        s = int(rng.integers(0, F - T + 1))
        oy, ox = rng.uniform(-6, 6), rng.uniform(-6, 6)
        sr = float(rng.uniform(5.0, 7.0))
        dr = float(rng.uniform(2.5, 3.8))
        # static distractors (same every frame of this sequence)
        base = np.zeros((S, S, 3), np.float32)
        cidx = rng.permutation(len(_DCOL))[:distractors]
        for k in range(distractors):
            dsfn = ra.SHAPES[int(rng.integers(len(ra.SHAPES)))]
            da = dsfn(float(rng.uniform(8, S - 8)), float(rng.uniform(8, S - 8)),
                      r=float(rng.uniform(3.0, 9.0)))
            base = _DCOL[cidx[k]][None, None, :] * da[..., None] + base * (1.0 - da[..., None])
        for f in range(T):
            t = (s + f) / (F - 1)
            cx, cy = pfn(t)
            cx = float(np.clip(cx + ox, 10, S - 10))
            cy = float(np.clip(cy + oy, 10, S - 10))
            frame = base.copy()
            sa = sfn(cx, cy, r=sr)                          # white target shape
            frame = sa[..., None] + frame * (1.0 - sa[..., None])
            da = ad.circle(cx, cy, r=dr)                   # red cursor on top
            frame = red[None, None, :] * da[..., None] + frame * (1.0 - da[..., None])
            X[i, f] = frame
            pos[i, f] = [cx / S, cy / S]
    X = ra._downscale(X, res)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    return (X * 255).astype(np.uint8), pos


def main(n=6):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    with open(os.path.join(_HERE, "radial_data", "dot_model.json")) as f:
        model = json.load(f)
    res = model["res"]
    dist = model.get("distractors", 0)
    genomes = model["genomes"]
    W = torch.tensor(model["W"], device=dev)
    mu = torch.tensor(model["mu"], device=dev)
    sd = torch.tensor(model["sd"], device=dev)

    Xb, _ = gen_dot(2000, res, seed=1, distractors=dist)     # env basis = training regime
    X8, pos = gen_cursor_seq(n, res, seed=42, distractors=dist)
    Xf = (X8.astype(np.float32) / 255.0).reshape(n * T, res, res, 3)
    env = Env(torch, dev, Xb, Xf, max_cached=6)
    F = torch.stack([feature(torch, tp, env, g, test=True) for g in genomes], 1)
    A = torch.cat([(F - mu) / sd, torch.ones(len(F), 1, device=dev)], 1)
    pred = (A @ W).cpu().numpy().reshape(n, T, 2)

    err = np.sqrt((((pred - pos) * ad.SIZE) ** 2).sum(-1))   # (n, T) px error
    seqs = []
    for i in range(n):
        seqs.append({"frames": [base64.b64encode(X8[i, f].tobytes()).decode() for f in range(T)],
                     "size": res, "true": pos[i].tolist(), "pred": pred[i].tolist(),
                     "err_px": [round(float(e), 2) for e in err[i]]})
    out = {"model": "cursor tracker (Model 1)", "frame_px": ad.SIZE, "res": res, "T": T,
           "n_feats": len(genomes), "mean_err_px": round(float(err.mean()), 3),
           "sequences": seqs}
    op = os.path.join(_HERE, "radial_data", "dot_cursor_demo.json")
    with open(op, "w") as f:
        json.dump(out, f)
    print(f"[dot-infer] DONE: {n} sequences, mean follow error {err.mean():.2f}px -> {op}",
          flush=True)


if __name__ == "__main__":
    main()
