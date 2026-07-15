"""dot_live.py — interactive Model-1b inference for the /animation page.

Generates a random canvas of colored shapes, then for a GRID of candidate cursor
positions runs the frozen cursor tracker + the shape classifier (Model 1b) in one
batched pass. The browser draws the scene, moves a red cursor with the mouse, and
reads the model's answer (where it thinks the cursor is + what shape sits under
it) straight out of the precomputed field — no per-move round trip. "Shuffle"
just asks for a new random scene.

All gradient-free. Reuses the frozen tracker (dot_model.json) and the
overlap-trained shape classifier (dot_shape_model.json), each with ITS OWN
patch-PCA basis rebuilt from its training regime (the recurring basis rule).
"""
import base64
import json
import os

import numpy as np

from radial_evo import _tprims
from radial_evo2 import Env, feature
from dot_track import gen_dot, _rand_color
import dot_shape as ds
import radial_anim as ra

try:
    from genreg_train import animation_data as ad
except ImportError:
    import animation_data as ad

_HERE = os.path.dirname(os.path.abspath(__file__))
_S = {}                                     # lazily loaded model state


def _load():
    if _S:
        return _S
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    with open(os.path.join(_HERE, "radial_data", "dot_model.json")) as f:
        trk = json.load(f)
    with open(os.path.join(_HERE, "radial_data", "dot_shape_model.json")) as f:
        ck = json.load(f)
    res, dist = trk["res"], trk["distractors"]
    Xb, _ = gen_dot(3000, res, seed=1, distractors=dist)                 # tracker basis
    _S.update(dict(torch=torch, dev=dev, tp=tp, trk=trk, ck=ck, res=res,
                   dist=dist, Xb=Xb, crop=ck["crop"], classes=ck["classes"]))
    # classifier crop basis: the tracker's crops of the classifier's own training
    # scenes (seed 1, overlap 0.6). Env fits its patch-PCA on Xtr[:2000], and
    # seed-1 generation is deterministic, so the first 2000 crops give the SAME
    # basis the classifier trained under — no need to rebuild all 7000.
    Xtr, _p, _y = ds.gen_labeled(2000, res, 1, dist, overlap=0.6)
    _S["Ctr"] = _crop(Xtr, _track(Xtr))
    return _S


def _track(X):
    """Frozen tracker -> predicted cursor pixel position (N, 2)."""
    torch, dev, tp = _S["torch"], _S["dev"], _S["tp"]
    trk, res = _S["trk"], _S["res"]
    W = torch.tensor(trk["W"], device=dev)
    mu = torch.tensor(trk["mu"], device=dev)
    sd = torch.tensor(trk["sd"], device=dev)
    env = Env(torch, dev, _S["Xb"], X, max_cached=6)
    F = torch.stack([feature(torch, tp, env, g, test=True) for g in trk["genomes"]], 1)
    A = torch.cat([(F - mu) / sd, torch.ones(len(F), 1, device=dev)], 1)
    return (A @ W).cpu().numpy() * res


def _crop(X, px):
    return ds._crop_at(X, px, _S["crop"])


def _classify(crops):
    """Shape classifier on the attended crops -> class idx (N,)."""
    torch, dev, tp = _S["torch"], _S["dev"], _S["tp"]
    ck = _S["ck"]
    W = torch.tensor(ck["W"], device=dev)
    mu = torch.tensor(ck["mu"], device=dev)
    sd = torch.tensor(ck["sd"], device=dev)
    env = Env(torch, dev, _S["Ctr"], crops, max_cached=6)
    F = torch.stack([feature(torch, tp, env, g, test=True) for g in ck["genomes"]], 1)
    A = torch.cat([(F - mu) / sd, torch.ones(len(F), 1, device=dev)], 1)
    return (A @ W).argmax(1).cpu().numpy()


def random_scene(seed, n_shapes=5):
    """A random canvas of colored shapes (no cursor). Returns the base RGB frame
    (res, res, 3) float and a list of {name, x, y, r} (front-to-back z order is
    the list order: later = drawn on top)."""
    rng = np.random.default_rng(seed)
    res = _S["res"]
    S = ad.SIZE
    SH = ra.SHAPES
    frame = np.zeros((S, S, 3), np.float32)
    shapes = []
    for _ in range(n_shapes):
        ti = int(rng.integers(len(SH)))
        x = float(rng.uniform(10, S - 10))
        y = float(rng.uniform(10, S - 10))
        r = float(rng.uniform(4.5, 8.5))
        sa = SH[ti](x, y, r=r)
        frame = _rand_color(rng)[None, None, :] * sa[..., None] + frame * (1.0 - sa[..., None])
        shapes.append({"name": ra.SHAPE_NAMES[ti], "x": round(x, 1), "y": round(y, 1),
                       "r": round(r, 1)})
    if res != S:
        frame = ra._downscale(frame[None, None], res)[0, 0]
    return frame.astype(np.float32), shapes


def compute(seed, stride=2, n_shapes=5):
    """Render a random scene and a full cursor-position field of model outputs."""
    _load()
    res = _S["res"]
    base, shapes = random_scene(seed, n_shapes)
    red = np.array([1.0, 0.0, 0.0], np.float32)
    gs = list(range(0, res, stride))
    frames, cells = [], []
    for gy in gs:
        for gx in gs:
            da = ad.circle(gx, gy, r=3.2)                # the red cursor at this cell
            if res != ad.SIZE:                           # cursor drawn in res-space directly
                da = _circle_res(gx, gy, res)
            f = red[None, None, :] * da[..., None] + base * (1.0 - da[..., None])
            frames.append(f)
            cells.append((gx, gy))
    X = np.stack(frames).astype(np.float32)
    trkpos = _track(X)                                   # (N,2) px
    crops = _crop(X, trkpos)
    pred = _classify(crops)                              # (N,) class idx

    G = len(gs)
    shape_field = pred.reshape(G, G).astype(int).tolist()
    trk_field = (trkpos.reshape(G, G, 2) / res).round(4).tolist()
    canvas = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    return {"res": res, "stride": stride, "grid": gs,
            "canvas": base64.b64encode(canvas.tobytes()).decode(),
            "shape_field": shape_field, "track_field": trk_field,
            "classes": _S["classes"], "shapes": shapes}


def _circle_res(cx, cy, res):
    yy, xx = np.mgrid[0:res, 0:res]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(3.2 - d + 0.5, 0, 1).astype(np.float32)


if __name__ == "__main__":
    import time
    _load()
    t = time.time()
    out = compute(7, stride=2)
    print(f"field {len(out['grid'])}x{len(out['grid'])} in {time.time()-t:.2f}s, "
          f"scene shapes: {[s['name'] for s in out['shapes']]}")
