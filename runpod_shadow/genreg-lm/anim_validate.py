"""anim_validate.py — adversarial validation suite for the animation models.

Addresses the skeptic's checklist (documentation/valid_animation.txt): the
"unseen" families must be provably unseen, and the fitness must be shown to
have rewarded the claimed concept, not a proxy. Every test uses the FROZEN
checkpoints — genomes fixed, heads fit once on the original training set.

  time-shuffle    frames of each test sequence randomly reordered.
                  PREDICTION: the motion model CRASHES (its answer lives in
                  frame order), the shape model DOESN'T CARE (its answer is
                  per-frame). If both hold, each fitness measured what we
                  claimed and not a shared proxy.
  time-reverse    sequences played backwards — a milder order probe.
  novel-shape     motion task with decoy shapes that DO NOT EXIST in
                  training (star / ellipse / bar). If motion acc holds, the
                  motion genomes never keyed on shape identity.
  label-shuffle   heads refit on permuted labels, same frozen features.
                  PREDICTION: chance. Kills any train/test leakage through
                  the features or protocol.
  raw-ridge       closed-form ridge on raw pixels (6,145 params/class head,
                  11x the shape model) — the "is the task trivial?" anchor.
  leakage-stat    translation-invariant distance from every unseen-family
                  trajectory (bezier / random-walk) to its NEAREST training
                  path window; control sequences anchor the scale at ~0.
  param-audit     per-space genome and parameter counts for both models.

Exports radial_data/anim_validation.json for the Animation page.

    python anim_validate.py          # ~2-3 min on GPU
"""
import json
import math
import os
import time

import numpy as np

from anim_ablate import _features, gen
from anim_infer import count_params
from radial_anim import T, PATHS, SHAPE_NAMES
from radial_evo import _tprims
from radial_evo2 import Env
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))
F = 24
N_AB = 1500

# ── novel decoy shapes: NOT in the training generator ────────────────
_YY64, _XX64 = np.mgrid[0:64, 0:64].astype(np.float32)


def _soft(d):
    return np.clip(0.5 - d, 0.0, 1.0)


def star5(x, y, r=7.0):
    """Five-pointed star (angular radius modulation)."""
    dx, dy = _XX64 - x, _YY64 - y
    ang = np.arctan2(dy, dx)
    rad = np.hypot(dx, dy)
    rr = r * (0.55 + 0.45 * np.cos(5.0 * ang))
    return _soft(rad - rr)


def ellipse(x, y, rx=8.0, ry=4.0):
    d = np.sqrt(((_XX64 - x) / rx) ** 2 + ((_YY64 - y) / ry) ** 2) - 1.0
    return _soft(d * min(rx, ry))


def bar(x, y, w=12.0, h=2.5):
    d = np.maximum(np.abs(_XX64 - x) - w / 2, np.abs(_YY64 - y) - h / 2)
    return _soft(d)


NOVEL_SHAPES = [star5, ellipse, bar]


def gen_novel_shape_motion(n=N_AB, seed=77):
    """Motion task (label = path) with decoy shapes unseen in training."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, len(PATHS), n)
    X = np.zeros((n, T, 32, 32), np.float32)
    for i in range(n):
        pfn = PATHS[y[i]][1]
        sfn = NOVEL_SHAPES[rng.integers(0, len(NOVEL_SHAPES))]
        s = rng.integers(0, F - T + 1)
        oy, ox = rng.uniform(-6, 6), rng.uniform(-6, 6)
        for f in range(T):
            cx, cy = pfn((s + f) / (F - 1))
            cx = float(np.clip(cx + ox, 4, 60))
            cy = float(np.clip(cy + oy, 4, 60))
            X[i, f] = sfn(cx, cy).reshape(32, 2, 32, 2).mean((1, 3))
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, 0.05, X.shape).astype(np.float32), 0, 1)
    return (np.repeat(X[..., None], 3, axis=4) * 255).astype(np.uint8), y


# ── leakage statistic ────────────────────────────────────────────────
def _traj(kind, n=400, seed=5):
    """Trajectories only (n, T, 2) for a regime, mirroring anim_ablate.gen."""
    rng = np.random.default_rng(seed + sum(ord(c) for c in kind))
    out = np.zeros((n, T, 2), np.float32)
    for i in range(n):
        if kind == "control":
            pfn = PATHS[rng.integers(0, len(PATHS))][1]
            s = rng.integers(0, F - T + 1)
            oy, ox = rng.uniform(-6, 6), rng.uniform(-6, 6)
            cs = [pfn((s + f) / (F - 1)) for f in range(T)]
            cs = [(cx + ox, cy + oy) for cx, cy in cs]
        elif kind == "random-walk":
            cx, cy = rng.uniform(14, 50), rng.uniform(14, 50)
            cs = []
            for _f in range(T):
                cs.append((cx, cy))
                cx = float(np.clip(cx + rng.normal(0, 2.5), 6, 58))
                cy = float(np.clip(cy + rng.normal(0, 2.5), 6, 58))
        elif kind == "bezier":
            P = rng.uniform(10, 54, (4, 2))
            s = rng.integers(0, F - T + 1)
            cs = []
            for f in range(T):
                t = (s + f) / (F - 1)
                u = 1.0 - t
                p = (u**3 * P[0] + 3 * u**2 * t * P[1]
                     + 3 * u * t**2 * P[2] + t**3 * P[3])
                cs.append((float(p[0]), float(p[1])))
        out[i] = cs
    return out


def leakage_stat():
    """Translation-invariant distance from each probe trajectory to its
    nearest training-path window. Control anchors the scale near zero."""
    wins = []                                   # all training path windows
    for _nm, pfn in PATHS:
        for s in range(F - T + 1):
            w = np.array([pfn((s + f) / (F - 1)) for f in range(T)], np.float32)
            wins.append(w - w.mean(0))
    wins = np.stack(wins)                       # (W, T, 2), centred
    out = {}
    for kind in ("control", "random-walk", "bezier"):
        tr = _traj(kind)
        tr = tr - tr.mean(1, keepdims=True)     # centre each trajectory
        # mean per-frame distance to every window; keep the nearest
        d = np.sqrt(((tr[:, None] - wins[None]) ** 2).sum(-1)).mean(-1)
        nearest = d.min(1)
        out[kind] = {"mean_px": round(float(nearest.mean()), 3),
                     "p5_px": round(float(np.percentile(nearest, 5)), 3),
                     "min_px": round(float(nearest.min()), 3)}
    return out


# ── main ─────────────────────────────────────────────────────────────
def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    z = np.load(os.path.join(_HERE, "radial_data", "anim_seq.npz"))
    XtrF = (z["Xtr"].astype(np.float32) / 255.0).reshape(-1, 32, 32, 3)
    XteF = (z["Xte"].astype(np.float32) / 255.0).reshape(-1, 32, 32, 3)
    Nte = len(z["yte"])
    rng = np.random.default_rng(9)

    def _fit_head(Ftr, ytr, lam_pick=True):
        Ntr = len(ytr)
        ytr_t = torch.tensor(ytr, device=dev)
        Y = -torch.ones((Ntr, 10), device=dev)
        Y[torch.arange(Ntr), ytr_t] = 1.0

        def _fit(Xf, Yf, lam):
            n, d = Xf.shape
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
            A = torch.hstack([(Xf - mu) / sd, torch.ones(n, 1, device=dev)])
            W = torch.linalg.solve(
                A.T @ A + lam * torch.eye(d + 1, device=dev), A.T @ Yf)
            return mu, sd, W

        if not lam_pick:
            return _fit(Ftr, Y, 1.0)
        n_fit = int(Ntr * 0.8)
        best = (1.0, -1.0)
        for lam in (1.0, 3.0, 10.0, 30.0):
            mu, sd, W = _fit(Ftr[:n_fit], Y[:n_fit], lam)
            s = torch.hstack([(Ftr[n_fit:] - mu) / sd,
                              torch.ones(Ntr - n_fit, 1, device=dev)]) @ W
            a = float((s.argmax(1) == ytr_t[n_fit:]).float().mean())
            if a > best[1]:
                best = (lam, a)
        return _fit(Ftr, Y, best[0])

    def _acc(Fte, head, y):
        mu, sd, W = head
        s = torch.hstack([(Fte - mu) / sd,
                          torch.ones(len(y), 1, device=dev)]) @ W
        return float((s.argmax(1).cpu().numpy() == y).mean())

    out = {"chance": 0.1, "tests": [], "seconds": None}

    # ---- both frozen models: features on train + original test ------
    models = {}
    for task, suffix, ylab in (("path", "", "ytr"), ("shape", "_shape", "ytr_shape")):
        with open(os.path.join(_HERE, "radial_data",
                               f"anim_model{suffix}.json")) as fh:
            ckpt = json.load(fh)
        rk.GRID = int(ckpt["grid"])
        env0 = Env(torch, dev, XtrF, XteF, max_cached=6)
        Ftr = _features(torch, tp, env0, ckpt, len(z[ylab]), test=False)
        Fte = _features(torch, tp, env0, ckpt, Nte, test=True)
        head = _fit_head(Ftr, z[ylab])
        yte = z["yte" + ("" if task == "path" else "_shape")]
        models[task] = dict(ckpt=ckpt, Ftr=Ftr, Fte=Fte, head=head, yte=yte,
                            base_acc=_acc(Fte, head, yte))
        print(f"{task}: baseline test {models[task]['base_acc']:.4f} "
              f"({round(time.time()-t0)}s)", flush=True)

    def _variant_acc(task, Xte_u8, y):
        m = models[task]
        rk.GRID = int(m["ckpt"]["grid"])
        envA = Env(torch, dev, XtrF,
                   (Xte_u8.astype(np.float32) / 255.0).reshape(-1, 32, 32, 3),
                   max_cached=6)
        Fte = _features(torch, tp, envA, m["ckpt"], len(y), test=True)
        return _acc(Fte, m["head"], y)

    # ---- 1. time-shuffle: motion must crash, shape must not care ----
    Xte_shuf = z["Xte"].copy()
    for i in range(Nte):
        Xte_shuf[i] = Xte_shuf[i][rng.permutation(T)]
    a_m = _variant_acc("path", Xte_shuf, z["yte"])
    a_s = _variant_acc("shape", Xte_shuf, z["yte_shape"])
    out["tests"].append({
        "name": "time-shuffle", "claim": "fitness measured the claimed concept",
        "desc": "frames of every test sequence randomly reordered",
        "motion_acc": round(a_m, 4), "shape_acc": round(a_s, 4),
        "prediction": "motion crashes toward its R0-only level (~0.58); "
                      "shape unaffected",
        "passed": bool(a_m < 0.7 and a_s > 0.95)})
    print(f"time-shuffle: motion {a_m:.4f} shape {a_s:.4f}", flush=True)

    # ---- 2. time-reverse ---------------------------------------------
    a_m = _variant_acc("path", z["Xte"][:, ::-1].copy(), z["yte"])
    a_s = _variant_acc("shape", z["Xte"][:, ::-1].copy(), z["yte_shape"])
    out["tests"].append({
        "name": "time-reverse", "claim": "order sensitivity (milder probe)",
        "desc": "every test sequence played backwards",
        "motion_acc": round(a_m, 4), "shape_acc": round(a_s, 4),
        "prediction": "shape unaffected; motion degrades (reversed paths are "
                      "near-novel motions)", "passed": bool(a_s > 0.95)})
    print(f"time-reverse: motion {a_m:.4f} shape {a_s:.4f}", flush=True)

    # ---- 3. novel decoy shapes for the MOTION model ------------------
    Xn, yn = gen_novel_shape_motion()
    a_m = _variant_acc("path", Xn, yn)
    out["tests"].append({
        "name": "novel-shape", "claim": "OOD generalization (motion model)",
        "desc": "motion task with star / ellipse / bar decoys — shapes that "
                "do not exist anywhere in training",
        "motion_acc": round(a_m, 4), "shape_acc": None,
        "prediction": "motion holds near its 0.88 baseline",
        "passed": bool(a_m > 0.8)})
    print(f"novel-shape: motion {a_m:.4f}", flush=True)

    # ---- 4. label-shuffle: leakage control ---------------------------
    res = {}
    for task in ("path", "shape"):
        m = models[task]
        yperm = rng.permutation(z["ytr" if task == "path" else "ytr_shape"])
        head_p = _fit_head(m["Ftr"], yperm, lam_pick=False)
        res[task] = _acc(m["Fte"], head_p, m["yte"])
    out["tests"].append({
        "name": "label-shuffle", "claim": "no train/test leakage",
        "desc": "heads refit on permuted labels, same frozen features",
        "motion_acc": round(res["path"], 4), "shape_acc": round(res["shape"], 4),
        "prediction": "both fall to chance (0.10)",
        "passed": bool(res["path"] < 0.15 and res["shape"] < 0.15)})
    print(f"label-shuffle: motion {res['path']:.4f} shape {res['shape']:.4f}",
          flush=True)

    # ---- 5. raw-pixel ridge baseline ---------------------------------
    raw_tr = torch.tensor(z["Xtr"][..., 0].reshape(len(z["ytr"]), -1)
                          .astype(np.float32) / 255.0, device=dev)
    raw_te = torch.tensor(z["Xte"][..., 0].reshape(Nte, -1)
                          .astype(np.float32) / 255.0, device=dev)
    raw = {}
    for task, ylab in (("path", "ytr"), ("shape", "ytr_shape")):
        head_r = _fit_head(raw_tr, z[ylab])
        raw[task] = _acc(raw_te, head_r,
                         z["yte" if task == "path" else "yte_shape"])
    out["tests"].append({
        "name": "raw-ridge", "claim": "task is not linearly trivial",
        "desc": f"closed-form ridge on raw pixels — {raw_tr.shape[1] + 1:,} "
                "params per class, 11x the whole shape model",
        "motion_acc": round(raw["path"], 4), "shape_acc": round(raw["shape"], 4),
        "prediction": "far below the evolved models despite more parameters",
        "passed": bool(raw["path"] < models["path"]["base_acc"] - 0.1)})
    print(f"raw-ridge: motion {raw['path']:.4f} shape {raw['shape']:.4f}",
          flush=True)

    # ---- 6. leakage statistic ----------------------------------------
    out["leakage"] = leakage_stat()
    print("leakage:", out["leakage"], flush=True)

    # ---- 7. param audit -----------------------------------------------
    audit = {}
    for task in ("path", "shape"):
        ck = models[task]["ckpt"]
        spaces = [{"space": i, "genomes": len(sp),
                   "params": sum(count_params(g) for g in sp)}
                  for i, sp in enumerate(ck["spaces"])]
        gp = sum(s["params"] for s in spaces)
        hp = (sum(s["genomes"] for s in spaces) + 1) * 10
        audit[task] = {"spaces": spaces, "genome_params": gp,
                       "head_params": hp, "total_params": gp + hp,
                       "kb_fp32": round((gp + hp) * 4 / 1024, 1),
                       "baseline_test_acc": round(models[task]["base_acc"], 4)}
    out["param_audit"] = audit
    out["raw_baseline_params"] = int(raw_tr.shape[1] + 1) * 10

    out["seconds"] = round(time.time() - t0)
    op = os.path.join(_HERE, "radial_data", "anim_validation.json")
    with open(op, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[anim-validate] DONE ({out['seconds']}s) -> {op}", flush=True)


if __name__ == "__main__":
    main()
