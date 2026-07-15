"""dot_shape.py — the attention payoff: identify the SHAPE the cursor is over,
building on the frozen tracker.

The tracker (dot_model.json) already knows WHERE the red cursor is. So we use ITS
OWN predicted position as an attention spotlight: crop the small window around the
prediction and evolve a shape classifier on that crop. "Where" is the tracker's
(frozen); only "what" — the shape — is learned. Distractors elsewhere fall
outside the attention crop. 10-class shape ID, gradient-free. Exports
radial_data/dot_shape.json.
"""
import base64
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, make_scorer, new_genome, mutate, feature
import radial_anim as ra
from dot_track import _rand_color, gen_dot

try:
    from genreg_train import animation_data as ad
except ImportError:
    import animation_data as ad

_HERE = os.path.dirname(os.path.abspath(__file__))
CROP = 20                                   # attended-window size (native px)
NSH = 10


def gen_labeled(n, res, seed, distractors, noise=0.03, shapes=None, overlap=0.6,
                jitter=0.45):
    """Frames + the TARGET shape label + the true cursor pixel position (res-space).
    `shapes` restricts the target to a subset of shape indices (label = index in
    the subset). `overlap`: fraction of examples where a distractor is deliberately
    drawn LARGE and CENTERED on the target so it ENCOMPASSES it (behind it) — the
    occlusion / figure-ground case. `jitter`: the cursor is placed at a random
    point WITHIN the target (up to jitter*r off center), not dead-center, so the
    model can read the shape from anywhere on its body (free hovering), not only
    when perfectly centered."""
    rng = np.random.default_rng(seed)
    S = ad.SIZE
    SH = ra.SHAPES
    subset = shapes if shapes is not None else list(range(NSH))
    X = np.zeros((n, 1, S, S, 3), np.float32)
    pos = np.zeros((n, 2), np.float32)
    ysh = np.zeros(n, np.int64)
    for i in range(n):
        x = float(rng.uniform(10, S - 10))
        y = float(rng.uniform(10, S - 10))
        big = rng.random() < overlap                 # an encompassing distractor?
        for k in range(distractors):
            dsfn = SH[int(rng.integers(len(SH)))]
            if big and k == 0:                       # large, centered on target, BEHIND it
                dx = float(np.clip(x + rng.uniform(-4, 4), 8, S - 8))
                dy = float(np.clip(y + rng.uniform(-4, 4), 8, S - 8))
                da = dsfn(dx, dy, r=float(rng.uniform(11.0, 15.0)))
            else:
                da = dsfn(float(rng.uniform(8, S - 8)), float(rng.uniform(8, S - 8)),
                          r=float(rng.uniform(3.0, 9.0)))
            X[i, 0] = _rand_color(rng)[None, None, :] * da[..., None] + X[i, 0] * (1.0 - da[..., None])
        li = int(rng.integers(len(subset)))
        ti = subset[li]
        ysh[i] = li
        tr = float(rng.uniform(6.0, 9.5))
        sa = SH[ti](x, y, r=tr)                               # target — random color, like distractors
        X[i, 0] = _rand_color(rng)[None, None, :] * sa[..., None] + X[i, 0] * (1.0 - sa[..., None])
        jm = float(rng.uniform(0, jitter)) * tr              # cursor within the shape, off-center
        ja = float(rng.uniform(0, 2 * np.pi))
        cx = float(np.clip(x + jm * np.cos(ja), 6, S - 6))
        cy = float(np.clip(y + jm * np.sin(ja), 6, S - 6))
        da = ad.circle(cx, cy, r=float(rng.uniform(2.5, 3.8)))
        X[i, 0] = np.array([1., 0., 0.], np.float32)[None, None, :] * da[..., None] \
            + X[i, 0] * (1.0 - da[..., None])
        pos[i] = [cx, cy]                                     # tracker target = the cursor
    X = ra._downscale(X, res)[:, 0]
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    return X.astype(np.float32), pos * (res / S), ysh


def _crop_at(X, px, crop):
    """Crop crop x crop around pixel position px=(x,y) per image (clipped)."""
    n, res = X.shape[0], X.shape[1]
    h = crop // 2
    out = np.zeros((n, crop, crop, 3), np.float32)
    for i in range(n):
        cx = int(round(np.clip(px[i, 0], h, res - h)))
        cy = int(round(np.clip(px[i, 1], h, res - h)))
        out[i] = X[i, cy - h:cy - h + crop, cx - h:cx - h + crop]
    return out


def _export_demo(Xte, ap_te, pred_lbl, yte, shapes, res, use_true, n=8):
    """Save a few test frames with the model's attended point + its shape call,
    so the page can show the attention payoff visually."""
    names = ra.SHAPE_NAMES
    subset = shapes if shapes is not None else list(range(NSH))
    # pick a spread: prefer some correct + a couple wrong if any
    correct = [i for i in range(len(yte)) if pred_lbl[i] == yte[i]]
    wrong = [i for i in range(len(yte)) if pred_lbl[i] != yte[i]]
    pick = (wrong[:2] + correct[:n - min(2, len(wrong))])[:n]
    if len(pick) < n:
        pick = list(range(min(n, len(Xte))))
    samples = []
    for i in pick:
        frame = (np.clip(Xte[i], 0, 1) * 255).astype(np.uint8)
        crop = _crop_at(Xte[i:i + 1], ap_te[i:i + 1], CROP)[0]
        crop = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        samples.append({
            "frame": base64.b64encode(frame.tobytes()).decode(),
            "crop": base64.b64encode(crop.tobytes()).decode(),
            "attend": [round(float(ap_te[i, 0]), 2), round(float(ap_te[i, 1]), 2)],
            "true_shape": names[subset[int(yte[i])]],
            "pred_shape": names[subset[int(pred_lbl[i])]],
            "correct": bool(pred_lbl[i] == yte[i])})
    demo = {"res": res, "crop": CROP, "attention": "true" if use_true else "tracker",
            "n_classes": len(subset), "samples": samples}
    with open(os.path.join(_HERE, "radial_data", "dot_shape_demo.json"), "w") as f:
        json.dump(demo, f)


def main(rounds=80, pop=64, gens=10, freeze_top=8, seed=0, use_true=False, shapes=None,
         overlap=0.6, save=True, test_overlaps=None):
    nsh = len(shapes) if shapes is not None else NSH
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    with open(os.path.join(_HERE, "radial_data", "dot_model.json")) as f:
        trk = json.load(f)
    res, dist = trk["res"], trk.get("distractors", 0)
    tg = trk["genomes"]
    Wt = torch.tensor(trk["W"], device=dev)
    mut = torch.tensor(trk["mu"], device=dev)
    sdt = torch.tensor(trk["sd"], device=dev)

    Xtr, ptr, ytr = gen_labeled(7000, res, 1, dist, shapes=shapes, overlap=overlap)
    Xte, pte, yte = gen_labeled(2000, res, 2, dist, shapes=shapes, overlap=overlap)
    # the tracker's genomes reference ITS OWN training basis — rebuild that basis
    # from the tracker's regime (gen_dot), not the shape data.
    Xbasis, _ = gen_dot(3000, res, seed=1, distractors=dist)

    def track(X):                           # the frozen tracker's predicted pixel pos
        env = Env(torch, dev, Xbasis, X, max_cached=6)
        F = torch.stack([feature(torch, tp, env, g, test=True) for g in tg], 1)
        A = torch.cat([(F - mut) / sdt, torch.ones(len(F), 1, device=dev)], 1)
        return (A @ Wt).cpu().numpy() * res     # normalised -> res px

    if use_true:
        ap_tr, ap_te = ptr, pte
    else:
        ap_tr, ap_te = track(Xtr), track(Xte)   # attend where the MODEL thinks the cursor is
    print(f"[dot-shape] attention crops from {'TRUE' if use_true else 'the tracker'}'s "
          f"position ({round(time.time()-t0)}s)", flush=True)

    Ctr = _crop_at(Xtr, ap_tr, CROP)
    Cte = _crop_at(Xte, ap_te, CROP)
    env = Env(torch, dev, Ctr, Cte, max_cached=6)   # evolve shape features on the crop
    ytr_t = torch.tensor(ytr, device=dev)
    yte_t = torch.tensor(yte, device=dev)
    n_fit = int(len(ytr) * 0.8)
    yv = ytr_t[n_fit:]
    Yf = -torch.ones((n_fit, nsh), device=dev)
    Yf[torch.arange(n_fit), ytr_t[:n_fit]] = 1.0
    Yfull = -torch.ones((len(ytr), nsh), device=dev)
    Yfull[torch.arange(len(ytr)), ytr_t] = 1.0
    rng = np.random.default_rng(seed)

    frozen, fcols = [], []
    hist = []
    for rnd in range(rounds):
        base = torch.stack(fcols, 1) if fcols else torch.zeros((len(Ctr), 0), device=dev)
        scorer, s0, a0 = make_scorer(torch, base, n_fit, Yf, yv)

        def fit_pop(gs):
            cols = [feature(torch, tp, env, g) for g in gs]
            ok = [i for i, c in enumerate(cols)
                  if float(c.std()) > 1e-6 and bool(torch.isfinite(c).all())]
            softs = np.full(len(gs), -1e9)
            accs = np.zeros(len(gs))
            if ok:
                sf, ac = scorer(torch.stack([cols[i] for i in ok], 1))
                for j, i in enumerate(ok):
                    softs[i] = sf[j] - s0
                    accs[i] = ac[j]
            return softs, accs, cols

        pg = [new_genome(rng) for _ in range(pop)]
        scales = np.full(pop, 0.25)
        fits, accs, cols = fit_pop(pg)
        for _ in range(gens):
            order = np.argsort(fits)[::-1]
            keep = list(order[:6])
            kids, ksc = [], []
            while len(kids) < pop - 6:
                cand = rng.choice(pop, 3)
                pi = cand[int(np.argmax(fits[cand]))]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                kids.append(mutate(rng, pg[pi], sc)); ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            pg = [pg[i] for i in keep] + kids
            scales = np.concatenate([scales[keep], ksc])
            fits = np.concatenate([fits[keep], kf])
            cols = [cols[i] for i in keep] + kc
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= 0.0005 or added >= freeze_top:
                break
            c = cols[idx]
            cz = (c - c.mean()) / (c.std() + 1e-9)
            if not any(float(torch.abs((cz * ((fc - fc.mean()) / (fc.std() + 1e-9))).mean())) > 0.95
                       for fc in fcols[-60:]):
                frozen.append(pg[idx]); fcols.append(c); added += 1
        base = torch.stack(fcols, 1)
        _, a1 = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv)
        hist.append({"round": rnd, "fitness": round(float(a1), 4), "added": added, "n": len(frozen)})
        print(f"  round {rnd:3d}  +{added} (feats {len(frozen)})  val acc {a1:.4f}  "
              f"({round(time.time()-t0)}s)", flush=True)
        if os.path.exists(_STOP) or (added == 0 and rnd > 3):
            break

    Fte = torch.stack([feature(torch, tp, env, g, test=True) for g in frozen], 1)
    Ftr = torch.stack(fcols, 1)
    best, best_lam = -1.0, 3.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        a = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)[1]
        if a > best:
            best, best_lam = a, lam
    # predicted class per test example at the best lambda, for the page demo
    muf, sdf = Ftr.mean(0), Ftr.std(0) + 1e-6
    Af = torch.cat([(Ftr - muf) / sdf, torch.ones(len(Ftr), 1, device=dev)], 1)
    Bf = torch.cat([(Fte - muf) / sdf, torch.ones(len(Fte), 1, device=dev)], 1)
    Wf = torch.linalg.solve(Af.T @ Af + best_lam * torch.eye(Af.shape[1], device=dev),
                            Af.T @ Yfull)
    pred_lbl = (Bf @ Wf).argmax(1).cpu().numpy()
    if shapes is None:                          # the page demo/result are the full 10-class run
        _export_demo(Xte, ap_te, pred_lbl, yte, shapes, res, use_true)
    # save a checkpoint so live/interactive inference can reload the classifier
    subset = shapes if shapes is not None else list(range(NSH))
    ckpt = {"genomes": frozen, "W": Wf.cpu().numpy().tolist(),
            "mu": muf.cpu().numpy().tolist(), "sd": sdf.cpu().numpy().tolist(),
            "crop": CROP, "res": res, "distractors": dist, "best_lam": best_lam,
            "classes": [ra.SHAPE_NAMES[i] for i in subset]}
    if save:
        ck_name = "dot_shape_sub_model.json" if shapes is not None else "dot_shape_model.json"
        with open(os.path.join(_HERE, "radial_data", ck_name), "w") as f:
            json.dump(ckpt, f)
    out = {"experiment": "shape-at-cursor (attention-gated classify on the tracker)",
           "task": "identify the shape under the red cursor (10 classes)", "chance": 0.1,
           "res": res, "distractors": dist, "crop": CROP,
           "attention": "true" if use_true else "tracker-predicted",
           "n_feats": len(frozen), "test_acc": round(best, 4),
           "seconds": round(time.time() - t0)}
    if shapes is None:
        with open(os.path.join(_HERE, "radial_data", "dot_shape.json"), "w") as f:
            json.dump(out, f, indent=1)
    print(f"[dot-shape] DONE: shape acc {best:.4f} (chance 0.10) on the "
          f"{'tracker-attended' if not use_true else 'true'} crop, "
          f"{len(frozen)} feats ({out['seconds']}s)", flush=True)

    # controlled cross-eval on chosen overlap regimes (correct crop basis = Ctr)
    if test_overlaps:
        depth = {}
        for j, ov in enumerate(test_overlaps):
            Xo, _po, yo = gen_labeled(2000, res, 50 + j, dist, shapes=shapes, overlap=ov)
            apo = track(Xo)
            Co = _crop_at(Xo, apo, CROP)
            eenv = Env(torch, dev, Ctr, Co, max_cached=6)
            Fo = torch.stack([feature(torch, tp, eenv, g, test=True) for g in frozen], 1)
            Ao = torch.cat([(Fo - muf) / sdf, torch.ones(len(Fo), 1, device=dev)], 1)
            acc = float(((Ao @ Wf).argmax(1).cpu().numpy() == yo).mean())
            depth[f"{ov:.2f}"] = round(acc, 4)
            print(f"[dot-shape] eval overlap={ov:.2f}: acc {acc:.4f}", flush=True)
        out["depth"] = depth
        if save and shapes is None:
            with open(os.path.join(_HERE, "radial_data", "dot_shape.json"), "w") as f:
                json.dump(out, f, indent=1)
    try:
        import dot_runs
        cfg = {"rounds": rounds, "pop": pop, "gens": gens, "freeze_top": freeze_top,
               "seed": seed, "shapes": shapes, "overlap": overlap, "n_classes": nsh,
               "crop": CROP, "attention": "true" if use_true else "tracker"}
        kind = "circle-vs-square" if nsh == 2 else (f"{nsh}-shape subset" if shapes else "10-shape")
        dot_runs.record("animation", cfg, hist, out,
                        label=f"shape-at-cursor ({kind}) acc {best:.3f}, {len(frozen)} feats",
                        tags=["attention", "classifier", "model1b"])
    except Exception as exc:
        print(f"[dot-shape] run record skipped: {exc}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    sh = [0, 1] if "--cs" in sys.argv else None    # circle vs square
    main(use_true=("--true" in sys.argv), shapes=sh)
