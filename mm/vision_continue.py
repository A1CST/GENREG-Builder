"""vision_continue.py — CONTINUED TRAINING (staple 2 of the vision demo).

Take the frozen SHAPE recognizer (multimodal/dot_shape_model.json, 634 genomes,
10 shape classes, 20px) and KEEP EVOLVING it so it also learns the 26 letters —
ending as ONE 36-class model, WITHOUT ever training a separate letter model. This
is the contrast to the union (mm_merge.py, which fuses two already-trained models):
here we GROW one model.

Method (gradient-free, the anim/dot_shape.py evolve loop, warm-started):
  - Load the 634 shape genomes; run them at 20px in the shape model's patch-PCA
    Basis over the 36-class shapes+letters data -> the (N,634) FROZEN residual base.
  - The base-only 36-class ridge head is the "before": the shape features already
    transfer to letters cross-modally (~0.92), so we start well above chance.
  - Evolve NEW radial_evo2 genomes scored ONLY on what they add over the frozen
    base (make_scorer soft-ridge gain), admit decorrelated contributors, freeze,
    extend the base. Every round we fit one 36-class head and measure TEST
    overall / shapes / letters -> the climb (the "after").

Exports radial_data/vision_demo_continue.json and multimodal/vision_continue_model.json.

    python mm/vision_continue.py            # full run
    python mm/vision_continue.py --smoke    # quick pipeline check
"""
import json
import os
import time

import numpy as np

import os as _os, sys as _sys                     # repo-root shim (run-anywhere)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # mm/ for mm_merge
import genreg_paths                               # noqa: F401  (repo-root shim)
import mm_merge as mmg                            # renderers, build_dataset, helpers
from radial_evo import _tprims, _ridge_soft
from radial_evo2 import make_scorer, new_genome, mutate, feature
from dot_live import Basis
import radial_anim as ra

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
MM = os.path.join(_HERE, "multimodal")
N_SHAPES, N_LETTERS = mmg.N_SHAPES, mmg.N_LETTERS
N_CLASSES = N_SHAPES + N_LETTERS               # 36


def _shape_cols(torch, tp, basis, genomes, X):
    """Run the frozen shape genomes on X (set as the basis input) -> (N, G)."""
    basis.set_input(X)
    return torch.stack([feature(torch, tp, basis, g, test=True) for g in genomes], 1)


def _head_eval(torch, dev, Ftr, ytr, Fte, yte, mte):
    """One 36-class ridge head; returns overall + per-modality test accuracy."""
    acc, pred, W, mu, sd, lam = mmg._ridge_head(torch, dev, Ftr, ytr, Fte, yte, N_CLASSES)
    sh = float((pred[mte == 0] == yte[mte == 0]).float().mean())
    le = float((pred[mte == 1] == yte[mte == 1]).float().mean())
    return {"overall": round(acc, 4), "shapes": round(sh, 4), "letters": round(le, 4)}, \
        (W, mu, sd, lam)


def run(n_per=140, rounds=50, pop=64, gens=10, freeze_top=8, seed=0, smoke=False,
        warm=True, save=True):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False          # guide rail 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if smoke:
        n_per, rounds = 40, 8

    # --- data: 36-class shapes+letters at 20px (the shape model's native scale) ---
    Xtr20, _Xtr32, ytr, mtr = mmg.build_dataset(torch, n_per, seed + 1)
    Xte20, _Xte32, yte, mte = mmg.build_dataset(torch, max(1, n_per // 3), seed + 2)
    ytr_t = torch.tensor(ytr, device=dev)
    yte_t = torch.tensor(yte, device=dev)
    mte_t = torch.tensor(mte, device=dev)
    Ntr = len(ytr)
    print(f"[continue] data: {Ntr} train / {len(yte)} test over {N_CLASSES} classes "
          f"({round(time.time()-t0)}s)", flush=True)

    # --- patch-PCA basis over shape crops (the shape model's native "eye") ---
    rng0 = np.random.default_rng(1)
    ref20 = np.stack([mmg.render_shape_crop(ti, rng0)
                      for ti in np.repeat(np.arange(N_SHAPES), 150)]).astype(np.float32)
    basis = Basis(torch, dev, ref20)

    n_fit = int(Ntr * 0.8)
    yv = ytr_t[n_fit:]
    Yf = -torch.ones((n_fit, N_CLASSES), device=dev)
    Yf[torch.arange(n_fit), ytr_t[:n_fit]] = 1.0
    rng = np.random.default_rng(seed)

    if warm:
        # WARM: seed the frozen base with the 634 shape genomes -> continued training.
        # New genomes are scored ONLY on what they add over this frozen base, so the
        # shape capability is preserved (never re-optimized) while letters are learned.
        shape_g = json.load(open(os.path.join(MM, "dot_shape_model.json")))["genomes"]
        Sf_tr = mmg._san(torch, _shape_cols(torch, tp, basis, shape_g, Xtr20))
        Sf_te = mmg._san(torch, _shape_cols(torch, tp, basis, shape_g, Xte20))
        frozen = list(shape_g)
        fcols = [Sf_tr[:, i].contiguous() for i in range(Sf_tr.shape[1])]
        ftecols = [Sf_te[:, i].contiguous() for i in range(Sf_te.shape[1])]
        before, _ = _head_eval(torch, dev, Sf_tr, ytr_t, Sf_te, yte_t, mte_t)
        print(f"[continue] WARM: {len(frozen)} frozen shape genomes; BEFORE overall "
              f"{before['overall']} letters {before['letters']}", flush=True)
    else:
        # SCRATCH control: empty base, no shape knowledge reused (the efficiency A/B).
        frozen, fcols, ftecols = [], [], []
        ch = round(1.0 / N_CLASSES, 4)
        before = {"overall": ch, "shapes": ch, "letters": ch}
        print("[continue] SCRATCH: empty base — no shape genomes reused", flush=True)
    n_base = len(frozen)
    basis.set_input(Xtr20)                                  # evolve on the train view
    curve = [{"n_new": 0, "val": None, **before}]

    for rnd in range(rounds):
        base = torch.stack(fcols, 1) if fcols else torch.zeros((Ntr, 0), device=dev)
        scorer, s0, a0 = make_scorer(torch, base, n_fit, Yf, yv)

        def fit_pop(gs):
            cols = [feature(torch, tp, basis, g) for g in gs]
            ok = [i for i, c in enumerate(cols)
                  if float(c.std()) > 1e-6 and bool(torch.isfinite(c).all())]
            softs = np.full(len(gs), -1e9)
            if ok:
                sf, _ac = scorer(torch.stack([cols[i] for i in ok], 1))
                for j, i in enumerate(ok):
                    softs[i] = sf[j] - s0
            return softs, cols

        pg = [new_genome(rng) for _ in range(pop)]
        scales = np.full(pop, 0.25)
        fits, cols = fit_pop(pg)
        for _ in range(gens):
            order = np.argsort(fits)[::-1]
            keep = list(order[:6])
            kids, ksc = [], []
            while len(kids) < pop - 6:
                cand = rng.choice(pop, 3)
                pi = cand[int(np.argmax(fits[cand]))]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                kids.append(mutate(rng, pg[pi], sc)); ksc.append(sc)
            kf, kc = fit_pop(kids)
            pg = [pg[i] for i in keep] + kids
            scales = np.concatenate([scales[keep], ksc])
            fits = np.concatenate([fits[keep], kf])
            cols = [cols[i] for i in keep] + kc

        order = np.argsort(fits)[::-1]
        added_g, added_tr = [], []
        for idx in order:
            if fits[idx] <= 0.0005 or len(added_tr) >= freeze_top:
                break
            c = cols[idx]
            cz = (c - c.mean()) / (c.std() + 1e-9)
            if not any(float(torch.abs((cz * ((fc - fc.mean()) / (fc.std() + 1e-9))).mean())) > 0.95
                       for fc in fcols[-60:]):
                frozen.append(pg[idx]); fcols.append(c)
                added_g.append(pg[idx]); added_tr.append(c)

        # test columns for the genomes frozen THIS round, then measure the ladder
        if added_g:
            basis.set_input(Xte20)
            for g in added_g:
                ftecols.append(mmg._san(torch, feature(torch, tp, basis, g, test=True)))
            basis.set_input(Xtr20)                          # restore train view
        if fcols:
            base = torch.stack(fcols, 1)
            _, a1 = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv)   # train-split signal
            lad, _ = _head_eval(torch, dev, base, ytr_t, torch.stack(ftecols, 1), yte_t, mte_t)
        else:
            a1, lad = 0.0, before
        curve.append({"n_new": len(frozen) - n_base, "val": round(float(a1), 4), **lad})
        print(f"  round {rnd:3d}  +{len(added_g)} (new {len(frozen)-n_base})  "
              f"val {a1:.4f}  test {lad['overall']}  letters {lad['letters']}  "
              f"({round(time.time()-t0)}s)", flush=True)
        if len(added_g) == 0 and rnd > 3:
            break

    # --- final head + per-modality "after" ---
    Ftr = torch.stack(fcols, 1)
    Fte = torch.stack(ftecols, 1)
    after, (W, mu, sd, lam) = _head_eval(torch, dev, Ftr, ytr_t, Fte, yte_t, mte_t)
    n_new = len(frozen) - n_base
    after = {**after, "n_new_genomes": n_new}
    head_params = int(W.numel())
    seconds = round(time.time() - t0)
    print(f"[continue] AFTER (+{n_new} evolved genomes): overall {after['overall']}  "
          f"shapes {after['shapes']}  letters {after['letters']}  ({seconds}s)", flush=True)
    print(f"[continue] letters {before['letters']} -> {after['letters']}  "
          f"(+{round(after['letters']-before['letters'], 4)}); "
          f"overall {before['overall']} -> {after['overall']}", flush=True)

    classes = list(ra.SHAPE_NAMES) + list(mmg.LETTERS)
    if save and warm:
        os.makedirs(MM, exist_ok=True)
        with open(os.path.join(MM, "vision_continue_model.json"), "w") as f:
            json.dump({"kind": "continued-training (shape model grown to shapes+letters)",
                       "base_checkpoint": "dot_shape_model.json", "n_base_genomes": n_base,
                       "new_genomes": frozen[n_base:], "res": mmg.SHAPE_RES, "classes": classes,
                       "head": {"W": W.cpu().numpy().tolist(), "mu": mu.cpu().numpy().tolist(),
                                "sd": sd.cpu().numpy().tolist(), "lam": lam},
                       "note": "run the 634 base shape genomes + new_genomes on the 20px view "
                               "in the shape Basis, concat, standardize with head.mu/sd, "
                               "argmax(A @ head.W)."}, f)

    out = {"experiment": ("continued training: shape model -> shapes + letters (one model)"
                          if warm else "from scratch control: empty base -> shapes + letters"),
           "warm": warm, "n_classes": N_CLASSES, "chance": round(1 / N_CLASSES, 4),
           "n_base_genomes": n_base, "n_new_genomes": n_new,
           "before": before, "curve": curve, "after": after,
           "params": {"head": head_params, "base_genomes": n_base, "new_genomes": n_new},
           "n_train": Ntr, "n_test": len(yte), "seconds": seconds}
    if save and warm:
        os.makedirs(RD, exist_ok=True)
        with open(os.path.join(RD, "vision_demo_continue.json"), "w") as f:
            json.dump(out, f, indent=1)
    print(f"[continue] DONE ({seconds}s){'' if warm else ' [scratch]'}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    run(smoke=("--smoke" in sys.argv))
