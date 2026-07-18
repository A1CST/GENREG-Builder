"""dot_ood.py — out-of-distribution stress tests for the attention line.

The tracker (dot_model.json) and the 10-class shape classifier
(dot_shape_model.json) were trained in one narrow regime: a red cursor on a
random-color shape over a BLACK background with 3 colored distractors, shapes
r~6-9.5, cursor r~2.5-3.8, light noise. Here we push each axis OUT of that
regime and measure how the frozen, gradient-free models hold up:

  clutter density (6 / 10 distractors), backgrounds (solid random color / pixel
  noise), near-RED decoy distractors (the tracker keys on red), cursor scale
  (big / tiny), shape scale (big / tiny), and heavy noise.

For each condition: the tracker's mean/median pixel error at localizing the red
cursor, and the classifier's accuracy reading the shape under it (chance 0.10).
Uses the cached-basis projector (dot_live.Basis) so features are normalized by
the training distribution. Exports radial_data/dot_ood.json (+ sample frames).
"""
import base64
import json
import os
import time

import numpy as np

import genreg_paths                               # noqa: F401
from radial_evo import _tprims
from radial_evo2 import feature
from dot_track import gen_dot, _rand_color
import dot_shape as ds
import radial_anim as ra
from dot_live import Basis

try:
    from genreg_train import animation_data as ad
except ImportError:
    import animation_data as ad

_HERE = os.path.dirname(os.path.abspath(__file__))

# each condition: (key, label, kwargs for gen_scene) — the FIRST is the in-dist ref
CONDITIONS = [
    ("indist", "in-distribution", {}),
    ("dist6", "6 distractors", {"distractors": 6}),
    ("dist10", "10 distractors", {"distractors": 10}),
    ("bgcolor", "solid color background", {"bg": "color"}),
    ("bgnoise", "noise background", {"bg": "noise"}),
    ("nearred", "near-red decoys", {"nearred": True}),
    ("bigcur", "big cursor", {"cur": (5.5, 7.0)}),
    ("tinycur", "tiny cursor", {"cur": (1.4, 2.0)}),
    ("bigshape", "big shapes", {"shape": (11.0, 15.0)}),
    ("smallshape", "small shapes", {"shape": (3.0, 4.5)}),
    ("heavynoise", "heavy noise", {"noise": 0.16}),
]


def gen_scene(n, res, seed, distractors=3, bg="black", nearred=False,
              cur=(2.5, 3.8), shape=(6.0, 9.5), noise=0.03):
    """Composite exactly like training but with one OOD axis pushed. Returns
    frames (n,res,res,3), cursor pixel positions (n,2), target shape idx (n,)."""
    rng = np.random.default_rng(seed)
    S = ad.SIZE
    SH = ra.SHAPES
    X = np.zeros((n, S, S, 3), np.float32)
    if bg == "color":
        for i in range(n):
            X[i] = _rand_color(rng)[None, None, :]
    elif bg == "noise":
        X = rng.uniform(0, 1, (n, S, S, 3)).astype(np.float32) * 0.6
    pos = np.zeros((n, 2), np.float32)
    ysh = np.zeros(n, np.int64)
    red = np.array([1.0, 0.0, 0.0], np.float32)
    for i in range(n):
        x = float(rng.uniform(10, S - 10))
        y = float(rng.uniform(10, S - 10))
        for _k in range(distractors):
            dsfn = SH[int(rng.integers(len(SH)))]
            da = dsfn(float(rng.uniform(8, S - 8)), float(rng.uniform(8, S - 8)),
                      r=float(rng.uniform(3.0, 9.0)))
            if nearred:                                  # reddish decoy colors
                col = np.array([rng.uniform(0.7, 1.0), rng.uniform(0.1, 0.45),
                                rng.uniform(0.1, 0.45)], np.float32)
            else:
                col = _rand_color(rng)
            X[i] = col[None, None, :] * da[..., None] + X[i] * (1.0 - da[..., None])
        ti = int(rng.integers(len(SH)))
        ysh[i] = ti
        sa = SH[ti](x, y, r=float(rng.uniform(*shape)))
        X[i] = _rand_color(rng)[None, None, :] * sa[..., None] + X[i] * (1.0 - sa[..., None])
        da = ad.circle(x, y, r=float(rng.uniform(*cur)))
        X[i] = red[None, None, :] * da[..., None] + X[i] * (1.0 - da[..., None])
        pos[i] = [x, y]
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    return X.astype(np.float32), pos, ysh


class _Rig:
    def __init__(self):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tp = _tprims(torch)
        with open(os.path.join(_HERE, "radial_data", "dot_model.json")) as f:
            self.trk = json.load(f)
        with open(os.path.join(_HERE, "radial_data", "dot_shape_model.json")) as f:
            self.ck = json.load(f)
        self.res = self.trk["res"]
        self.dist = self.trk["distractors"]
        self.crop = self.ck["crop"]
        Xb, _ = gen_dot(2000, self.res, seed=1, distractors=self.dist)
        self.tb = Basis(torch, self.dev, Xb)
        Xtr, _p, _y = ds.gen_labeled(2000, self.res, 1, self.dist, overlap=0.6)
        Ctr = ds._crop_at(Xtr, self.track(Xtr), self.crop)
        self.cb = Basis(torch, self.dev, Ctr)

    def track(self, X):
        torch, dev = self.torch, self.dev
        W = torch.tensor(self.trk["W"], device=dev)
        mu = torch.tensor(self.trk["mu"], device=dev)
        sd = torch.tensor(self.trk["sd"], device=dev)
        if hasattr(self, "tb"):
            self.tb.set_input(X); env = self.tb
        else:
            from radial_evo2 import Env
            env = Env(torch, dev, X, X, max_cached=6)
        F = torch.stack([feature(torch, self.tp, env, g, test=True) for g in self.trk["genomes"]], 1)
        A = torch.cat([(F - mu) / sd, torch.ones(len(F), 1, device=dev)], 1)
        return (A @ W).cpu().numpy() * self.res

    def classify(self, crops):
        torch, dev = self.torch, self.dev
        W = torch.tensor(self.ck["W"], device=dev)
        mu = torch.tensor(self.ck["mu"], device=dev)
        sd = torch.tensor(self.ck["sd"], device=dev)
        self.cb.set_input(crops)
        F = torch.stack([feature(torch, self.tp, self.cb, g, test=True) for g in self.ck["genomes"]], 1)
        A = torch.cat([(F - mu) / sd, torch.ones(len(F), 1, device=dev)], 1)
        return (A @ W).argmax(1).cpu().numpy()


def run(n=1500, verbose=True):
    t0 = time.time()
    rig = _Rig()
    res = rig.res
    results, samples = [], {}
    for j, (key, label, kw) in enumerate(CONDITIONS):
        X, pos, ysh = gen_scene(n, res, seed=100 + j, **kw)
        pred = rig.track(X)
        err = np.sqrt(((pred - pos) ** 2).sum(1))
        crops = ds._crop_at(X, pred, rig.crop)
        psh = rig.classify(crops)
        acc = float((psh == ysh).mean())
        results.append({"key": key, "label": label,
                        "track_err_px": round(float(err.mean()), 2),
                        "track_med_px": round(float(np.median(err)), 2),
                        "shape_acc": round(acc, 3)})
        # a few sample frames for the visual (the first 3)
        samples[key] = [{
            "frame": base64.b64encode((np.clip(X[i], 0, 1) * 255).astype(np.uint8).tobytes()).decode(),
            "true": [round(float(pos[i, 0]), 1), round(float(pos[i, 1]), 1)],
            "pred": [round(float(pred[i, 0]), 1), round(float(pred[i, 1]), 1)],
            "true_shape": ra.SHAPE_NAMES[int(ysh[i])], "pred_shape": ra.SHAPE_NAMES[int(psh[i])]}
            for i in range(3)]
        if verbose:
            print(f"  {label:24s} err {results[-1]['track_err_px']:5.2f}px "
                  f"(med {results[-1]['track_med_px']:.2f})  shape {acc:.3f}", flush=True)
    out = {"res": res, "n": n, "crop": rig.crop, "chance": round(1.0 / len(ra.SHAPES), 3),
           "results": results, "samples": samples, "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "radial_data", "dot_ood.json"), "w") as f:
        json.dump(out, f)
    print(f"[dot-ood] DONE: {len(results)} conditions ({out['seconds']}s)", flush=True)
    return out


if __name__ == "__main__":
    run()
