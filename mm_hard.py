"""mm_hard.py — a HARD test of the multimodal shape+letter fusion.

The easy test had shapes at ceiling and clean renders. Here we crank difficulty:
 - small shapes/letters, heavy noise (0.12), brightness 0.4-1.0, +-20 deg rotation
 - a CROSS-MODAL confusion probe on visual look-alikes (circle/ring/hexagon vs
   'o', xcross vs 'x', crescent vs 'c') — the real multimodal challenge
 - an OOD arm: UPPERCASE letters at test time (the letter model only ever saw
   lowercase)

Frozen banks, late fusion, one closed-form head refit on the hard train set.
Reports shape-only / letter-only / fused, modality accuracy, cross-modal error,
and the look-alike breakdown. Exports multimodal/mm_hard_result.json.
"""
import json
import os
import time

import numpy as np
from PIL import Image

import genreg_paths                               # noqa: F401
import radial_anim as ra
from radial_evo import _tprims
from dot_track import _rand_color
import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "lm"))   # lm-package path
from radial_kid import render_letter, LETTERS
import mm_merge as mm

try:
    from genreg_train import animation_data as ad
except ImportError:
    import animation_data as ad

_HERE = os.path.dirname(os.path.abspath(__file__))
SHAPE_RES, LETTER_RES = mm.SHAPE_RES, mm.LETTER_RES
N_SHAPES, N_LETTERS = mm.N_SHAPES, mm.N_LETTERS
CLASSES = list(ra.SHAPE_NAMES) + list(LETTERS)
# shape class -> the letter it visually collides with
LOOKALIKE = {"circle": "o", "ring": "o", "hexagon": "o", "crescent": "c", "xcross": "x"}


def _rotate(img, deg):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    im = im.rotate(deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return np.asarray(im, np.float32) / 255.0


def hard_shape(ti, rng):
    S = ad.SIZE
    img = np.zeros((S, S, 3), np.float32)
    r = float(rng.uniform(4.5, 7.5))
    sa = ra.SHAPES[ti](S / 2, S / 2, r=r)
    col = _rand_color(rng) * float(rng.uniform(0.5, 1.0))
    img = col[None, None, :] * sa[..., None] + img * (1 - sa[..., None])
    da = ad.circle(S / 2, S / 2, r=float(rng.uniform(2.2, 3.4)))
    img = np.array([1., 0, 0], np.float32)[None, None, :] * da[..., None] + img * (1 - da[..., None])
    img = _rotate(img, float(rng.uniform(-20, 20)))
    h = SHAPE_RES // 2
    crop = img[S // 2 - h:S // 2 - h + SHAPE_RES, S // 2 - h:S // 2 - h + SHAPE_RES]
    return np.clip(crop + rng.normal(0, 0.12, crop.shape).astype(np.float32), 0, 1)


def hard_letter(li, rng, upper=False):
    ch = LETTERS[li].upper() if upper else LETTERS[li]
    g = render_letter(ch, rng, size=int(rng.integers(10, 19)))
    img = np.repeat(g[..., None], 3, axis=2) * float(rng.uniform(0.4, 1.0))
    img = _rotate(img, float(rng.uniform(-20, 20)))
    return np.clip(img + rng.normal(0, 0.12, img.shape).astype(np.float32), 0, 1)


def build_hard(torch, n_per, seed, upper=False):
    rng = np.random.default_rng(seed)
    sh = np.stack([hard_shape(ti, rng) for ti in np.repeat(np.arange(N_SHAPES), n_per)])
    ysh = np.repeat(np.arange(N_SHAPES), n_per)
    lt = np.stack([hard_letter(li, rng, upper) for li in np.repeat(np.arange(N_LETTERS), n_per)])
    ylt = np.repeat(np.arange(N_LETTERS), n_per)
    X20 = np.concatenate([sh, mm._resize(torch, lt, SHAPE_RES)], 0).astype(np.float32)
    X32 = np.concatenate([mm._resize(torch, sh, LETTER_RES), lt], 0).astype(np.float32)
    y = np.concatenate([ysh, N_SHAPES + ylt]).astype(np.int64)
    mod = np.concatenate([np.zeros(len(ysh)), np.ones(len(ylt))]).astype(np.int64)
    p = rng.permutation(len(y))
    return X20[p], X32[p], y[p], mod[p]


def run(n_per=140, seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    import radial_stack as rk
    rk.GRID = 8
    t0 = time.time()

    shape_g = json.load(open(os.path.join(_HERE, "multimodal", "dot_shape_model.json")))["genomes"]
    A = json.load(open(os.path.join(_HERE, "multimodal", "kid_modelA.json")))["spaces"]

    # frozen banks keep their CLEAN training distribution as basis
    rng = np.random.default_rng(1)
    ref20 = np.stack([mm.render_shape_crop(ti, rng)
                      for ti in np.repeat(np.arange(N_SHAPES), 150)]).astype(np.float32)
    lref = np.stack([np.repeat(render_letter(LETTERS[li], rng)[..., None], 3, axis=2)
                     for li in np.repeat(np.arange(N_LETTERS), 100)]).astype(np.float32)

    def feats(X20, X32):
        Sf = mm.shape_features(torch, tp, dev, shape_g, ref20, X20)
        Lf = mm.letter_features(torch, tp, dev, A, lref, X32)
        return Sf, Lf

    Xtr20, Xtr32, ytr, mtr = build_hard(torch, n_per, seed + 1)
    Xte20, Xte32, yte, mte = build_hard(torch, n_per // 3, seed + 2)
    Xou20, Xou32, you, mou = build_hard(torch, n_per // 3, seed + 3, upper=True)  # OOD uppercase
    print(f"[hard] train {len(ytr)} / test {len(yte)} / OOD-upper {len(you)} "
          f"({round(time.time()-t0)}s)", flush=True)

    Str, Ltr = feats(Xtr20, Xtr32)
    Ste, Lte = feats(Xte20, Xte32)
    Sou, Lou = feats(Xou20, Xou32)
    print(f"[hard] features extracted ({round(time.time()-t0)}s)", flush=True)

    ytr_t = torch.tensor(ytr, device=dev)

    def evalset(name, Ftr, Fte, yt, md):
        yt_t = torch.tensor(yt, device=dev)
        acc, pred, W, mu, sd, lam = mm._ridge_head(torch, dev, Ftr, ytr_t, Fte, yt_t, 36)
        pred_np = pred.cpu().numpy()
        sh = float((pred_np[md == 0] == yt[md == 0]).mean())
        le = float((pred_np[md == 1] == yt[md == 1]).mean())
        pred_mod = (pred_np >= N_SHAPES).astype(int)
        mod_acc = float((pred_mod == md).mean())
        wrong = pred_np != yt
        cross = float((pred_mod[wrong] != md[wrong]).mean()) if wrong.any() else 0.0
        return {"set": name, "overall": round(acc, 4), "shapes": round(sh, 4),
                "letters": round(le, 4), "modality_acc": round(mod_acc, 4),
                "cross_modal_of_errors": round(cross, 4)}, pred_np

    def bank(Ftr, Fte, Fou):
        r_te, pred_te = evalset("hard-test", Ftr, Fte, yte, mte)
        r_ou, _ = evalset("OOD-uppercase", Ftr, Fou, you, mou)
        return r_te, r_ou, pred_te

    print("[hard] joint 36-class head, hard renders (small/noisy/rotated):", flush=True)
    banks = {}
    for nm, tr, te, ou in [("shape-only", Str, Ste, Sou),
                           ("letter-only", Ltr, Lte, Lou),
                           ("FUSED", torch.cat([Str, Ltr], 1), torch.cat([Ste, Lte], 1),
                            torch.cat([Sou, Lou], 1))]:
        r_te, r_ou, pred_te = bank(tr, te, ou)
        banks[nm] = {"hard_test": r_te, "ood_upper": r_ou}
        print(f"  {nm:12s} HARD overall {r_te['overall']:.4f} (sh {r_te['shapes']:.4f} "
              f"le {r_te['letters']:.4f}, modality {r_te['modality_acc']:.4f}, "
              f"cross-modal-of-errors {r_te['cross_modal_of_errors']:.4f})  |  "
              f"OOD-upper {r_ou['overall']:.4f}", flush=True)
        if nm == "FUSED":
            fused_pred = pred_te

    # look-alike breakdown on the FUSED hard-test predictions
    print("[hard] cross-modal look-alikes (FUSED, hard-test):", flush=True)
    look = []
    for sname, lch in LOOKALIKE.items():
        sidx = ra.SHAPE_NAMES.index(sname)
        lidx = N_SHAPES + (ord(lch) - ord('a'))
        m = yte == sidx
        if not m.any():
            continue
        acc = float((fused_pred[m] == sidx).mean())
        as_letter = float((fused_pred[m] == lidx).mean())
        any_letter = float((fused_pred[m] >= N_SHAPES).mean())
        look.append({"shape": sname, "vs_letter": lch, "n": int(m.sum()),
                     "correct": round(acc, 3), "called_that_letter": round(as_letter, 3),
                     "called_any_letter": round(any_letter, 3)})
        print(f"    {sname:9s} vs '{lch}':  correct {acc:.3f}  called-'{lch}' {as_letter:.3f}"
              f"  called-any-letter {any_letter:.3f}", flush=True)

    out = {"experiment": "multimodal HARD test (small/noisy/rotated + OOD uppercase)",
           "n_classes": 36, "chance": round(1 / 36, 4), "n_train": len(ytr),
           "n_test": len(yte), "banks": banks, "lookalikes": look,
           "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "multimodal", "mm_hard_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[hard] DONE ({out['seconds']}s) -> multimodal/mm_hard_result.json", flush=True)
    return out


if __name__ == "__main__":
    run()
