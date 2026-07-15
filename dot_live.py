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
from radial_evo2 import Env, feature, SCALES, C_PER_SCALE
from dot_track import gen_dot, _rand_color
import dot_shape as ds
import radial_anim as ra

try:
    from genreg_train import animation_data as ad
except ImportError:
    import animation_data as ad

_HERE = os.path.dirname(os.path.abspath(__file__))
_S = {}                                     # lazily loaded model state


class Basis:
    """A patch-PCA basis fit ONCE from a reference set, then used to project any
    new batch. Mirrors radial_evo2.Env.maps exactly (same seeds, cap, comps) so
    frozen genomes read the same features — but the SVD runs once at load, not
    per request, so live inference is fast and low-peak-memory. Test maps are
    normalised by the REFERENCE std (the training distribution), which is also
    the correct normalisation for a narrow live batch."""

    def __init__(self, torch, dev, ref):
        import torch.nn.functional as Fn
        self.torch, self.dev, self.Fn = torch, dev, Fn
        self.ref = ref
        self.S = ref.shape[1]
        self.b = {}                          # ps -> (comps, mu, sd, H, W, stride)
        self._cur = None                     # (X, {ps: Mte}) cache for one call

    def _basis(self, ps):
        torch, Fn = self.torch, self.Fn
        if ps in self.b:
            return self.b[ps]
        stride = max(2, ps // 2)
        d = ps * ps * 3
        i2k = torch.tensor(self.ref[:2000], device=self.dev).permute(0, 3, 1, 2).contiguous()
        P = Fn.unfold(i2k, ps, stride=stride)
        cols = P.permute(0, 2, 1).reshape(-1, d)
        gg = torch.Generator(device="cpu").manual_seed(ps)
        cols = cols[torch.randperm(len(cols), generator=gg)[:100000].to(self.dev)]
        mu = cols.mean(0)
        _, _, V = torch.linalg.svd(cols - mu, full_matrices=False)
        comps = V[:min(C_PER_SCALE, d)]
        H = W = (self.S - ps) // stride + 1
        sd = self._project(self.ref, ps, comps, mu, stride).std((0, 2), keepdim=True) + 1e-6
        self.b[ps] = (comps, mu, sd, H, W, stride)
        return self.b[ps]

    def _project(self, X, ps, comps, mu, stride, bs=400):
        torch, Fn = self.torch, self.Fn
        out = None
        for b in range(0, len(X), bs):
            imgs = torch.tensor(X[b:b + bs], device=self.dev).permute(0, 3, 1, 2).contiguous()
            U = Fn.unfold(imgs, ps, stride=stride)
            M = torch.einsum("cd,bdl->bcl", comps, U - mu.view(1, -1, 1))
            if out is None:
                out = torch.zeros((len(X), M.shape[1], M.shape[2]), device=self.dev)
            out[b:b + len(imgs)] = M
        return out

    def set_input(self, X):
        self._cur = (id(X), {})
        self._X = X

    def maps(self, ps):                      # feature() calls this; only Mte is used
        comps, mu, sd, H, W, stride = self._basis(ps)
        if ps not in self._cur[1]:
            M = (self._project(self._X, ps, comps, mu, stride) / sd).half()
            self._cur[1][ps] = M
        Mte = self._cur[1][ps]
        return Mte, Mte, H, W


def _load():
    if _S:
        return _S
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    with open(os.path.join(_HERE, "radial_data", "dot_model.json")) as f:
        trk = json.load(f)
    # prefer a distinct-shape subset classifier for a clean interactive read;
    # fall back to the full 10-class model.
    sub = os.path.join(_HERE, "radial_data", "dot_shape_sub_model.json")
    ckp = sub if os.path.exists(sub) else os.path.join(_HERE, "radial_data",
                                                        "dot_shape_model.json")
    with open(ckp) as f:
        ck = json.load(f)
    res, dist = trk["res"], trk["distractors"]
    Xb, _ = gen_dot(2000, res, seed=1, distractors=dist)                 # tracker basis
    # the shape indices (into ra.SHAPES) this classifier knows, in class order
    allow = [ra.SHAPE_NAMES.index(n) for n in ck["classes"]]
    _S.update(dict(torch=torch, dev=dev, tp=tp, trk=trk, ck=ck, res=res,
                   dist=dist, crop=ck["crop"], classes=ck["classes"], allow=allow))
    _S["trk_basis"] = Basis(torch, dev, Xb)          # fit ONCE, reused per request
    # classifier crop basis: the tracker's crops of the classifier's own training
    # scenes (seed 1, overlap 0.6) — the first 2000 give the same basis the
    # classifier trained under (deterministic seed-1 generation).
    subset = allow if os.path.exists(sub) else None
    Xtr, _p, _y = ds.gen_labeled(2000, res, 1, dist, overlap=0.6, shapes=subset)
    _S["cls_basis"] = Basis(torch, dev, _crop(Xtr, _track(Xtr)))
    return _S


def _track(X):
    """Frozen tracker -> predicted cursor pixel position (N, 2)."""
    torch, dev, tp = _S["torch"], _S["dev"], _S["tp"]
    trk, res = _S["trk"], _S["res"]
    W = torch.tensor(trk["W"], device=dev)
    mu = torch.tensor(trk["mu"], device=dev)
    sd = torch.tensor(trk["sd"], device=dev)
    if "trk_basis" in _S:
        env = _S["trk_basis"]; env.set_input(X)
    else:                                            # first call (building the basis itself)
        env = Env(torch, dev, X, X, max_cached=6)
    F = torch.stack([feature(torch, tp, env, g, test=True) for g in trk["genomes"]], 1)
    A = torch.cat([(F - mu) / sd, torch.ones(len(F), 1, device=dev)], 1)
    out = (A @ W).cpu().numpy() * res
    torch.cuda.empty_cache()
    return out


def _crop(X, px):
    return ds._crop_at(X, px, _S["crop"])


def _classify(crops):
    """Shape classifier on the attended crops -> class idx (N,). Uses the cached
    crop basis, so the test maps are normalised by the training distribution."""
    torch, dev, tp = _S["torch"], _S["dev"], _S["tp"]
    ck = _S["ck"]
    W = torch.tensor(ck["W"], device=dev)
    mu = torch.tensor(ck["mu"], device=dev)
    sd = torch.tensor(ck["sd"], device=dev)
    env = _S["cls_basis"]; env.set_input(crops)
    F = torch.stack([feature(torch, tp, env, g, test=True) for g in ck["genomes"]], 1)
    A = torch.cat([(F - mu) / sd, torch.ones(len(F), 1, device=dev)], 1)
    out = (A @ W).argmax(1).cpu().numpy()
    torch.cuda.empty_cache()
    return out


def random_scene(seed, n_shapes=5):
    """A random canvas of WELL-SEPARATED colored shapes (no cursor). Shapes are
    rejection-sampled to keep their centers apart, so each has a clean body to
    hover — the model was trained with the cursor at a shape's CENTER, so the
    readable signal is strongest there. Returns the base RGB frame (res,res,3)
    float and a list of {name, x, y, r}."""
    rng = np.random.default_rng(seed)
    res = _S["res"]
    S = ad.SIZE
    SH = ra.SHAPES
    frame = np.zeros((S, S, 3), np.float32)
    shapes = []
    allow = _S.get("allow") or list(range(len(SH)))
    tries = 0
    while len(shapes) < n_shapes and tries < 800:
        tries += 1
        ti = int(allow[int(rng.integers(len(allow)))])
        x = float(rng.uniform(12, S - 12))
        y = float(rng.uniform(12, S - 12))
        r = float(rng.uniform(7.0, 9.5))
        if any((x - s["x"]) ** 2 + (y - s["y"]) ** 2 < (r + s["r"] + 7) ** 2 for s in shapes):
            continue                                 # keep shapes from overlapping
        sa = SH[ti](x, y, r=r)
        frame = _rand_color(rng)[None, None, :] * sa[..., None] + frame * (1.0 - sa[..., None])
        shapes.append({"name": ra.SHAPE_NAMES[ti], "x": round(x, 1), "y": round(y, 1),
                       "r": round(r, 1), "ti": ti})
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
    grid_pred = pred.reshape(G, G).astype(int)
    # spatial-majority smoothing: within a shape body the plurality read is the
    # right one, so a 3x3 mode vote cleans up sporadic single-cell errors.
    sm = grid_pred.copy()
    for iy in range(G):
        for ix in range(G):
            y0, x0 = max(0, iy - 1), max(0, ix - 1)
            win = grid_pred[y0:iy + 2, x0:ix + 2].ravel()
            vals, cnts = np.unique(win, return_counts=True)
            sm[iy, ix] = int(vals[cnts.argmax()])
    shape_field = sm.tolist()
    trk_field = (trkpos.reshape(G, G, 2) / res).round(4).tolist()
    # gate: is the cursor inside any shape's extent? (geometric — handles hollow
    # shapes like ring/frame whose centers are empty). Off-shape cells read
    # "nothing" instead of a random class.
    sc = ad.SIZE / res
    has = np.zeros((G, G), bool)
    for iy, gy in enumerate(gs):
        for ix, gx in enumerate(gs):
            gx_s, gy_s = gx * sc, gy * sc            # grid cell in native (shape) coords
            has[iy, ix] = any((gx_s - s["x"]) ** 2 + (gy_s - s["y"]) ** 2
                              <= (s["r"] + 1.5) ** 2 for s in shapes)
    # per-shape read: the model's MAJORITY vote over every cursor cell inside the
    # shape body — a robust, stable answer that reads the shape from many points,
    # not one fragile pixel. Stored on each shape as class index `read`.
    for s in shapes:
        votes = []
        core = max(3.0, 0.5 * s["r"])               # inner region — where the read is reliable
        for iy, gy in enumerate(gs):
            for ix, gx in enumerate(gs):
                if (gx * sc - s["x"]) ** 2 + (gy * sc - s["y"]) ** 2 <= core ** 2:
                    votes.append(int(grid_pred[iy, ix]))
        if votes:
            v, c = np.unique(votes, return_counts=True)
            s["read"] = int(v[c.argmax()])
            s["conf"] = round(float(c.max()) / len(votes), 2)
        else:
            s["read"] = int(grid_pred[G // 2, G // 2])
            s["conf"] = 0.0
    canvas = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    return {"res": res, "stride": stride, "grid": gs,
            "canvas": base64.b64encode(canvas.tobytes()).decode(),
            "shape_field": shape_field, "track_field": trk_field,
            "has_field": has.astype(int).tolist(),
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
