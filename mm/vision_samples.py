"""vision_samples.py — build the animation payload for /vision_demo: render sample
images, run each frozen checkpoint on them, and export the model's predictions as
PNG data-URIs so the page can ANIMATE the checkpoints identifying shapes/letters.

Four panels:
  shape      — the shape checkpoint identifying the 10 shapes (its 10-class head)
  letter     — the letter checkpoint identifying the 26 letters (26-class head fit
               on its features; kid_modelA carries no head)
  union      — the fused model identifying both (mm_model.json, 36-class)
  continued  — the grown shape model identifying both (vision_continue_model.json)

Exports radial_data/vision_demo_samples.json.

    python mm/vision_samples.py
"""
import base64
import io
import json
import os

import numpy as np

import os as _os, sys as _sys                     # repo-root shim (run-anywhere)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import genreg_paths                               # noqa: F401

import mm_merge as mmg
from radial_evo import _tprims
from radial_evo2 import feature
from dot_live import Basis
import radial_anim as ra
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD, MM = os.path.join(_HERE, "radial_data"), os.path.join(_HERE, "multimodal")
N_SHAPES, N_LETTERS = mmg.N_SHAPES, mmg.N_LETTERS
SHAPE_NAMES = list(ra.SHAPE_NAMES)
CLASSES36 = SHAPE_NAMES + list(mmg.LETTERS)
K_SHOW = 14                                        # samples per panel (animation loop)


def _png_uri(img):
    """img: (H,W,3) float [0,1] -> a PNG data-URI (native res; page upscales)."""
    a = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a, "RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _render_shapes(n_per, seed):
    rng = np.random.default_rng(seed)
    imgs, y = [], []
    for ti in range(N_SHAPES):
        for _ in range(n_per):
            imgs.append(mmg.render_shape_crop(ti, rng)); y.append(ti)
    return np.array(imgs, np.float32), np.array(y, np.int64)


def _render_letters(n_per, seed):
    rng = np.random.default_rng(seed)
    imgs, y = [], []
    for li in range(N_LETTERS):
        for _ in range(n_per):
            g = mmg.render_letter(mmg.LETTERS[li], rng)
            imgs.append(np.repeat(g[..., None], 3, axis=2)); y.append(li)
    arr = np.clip(np.array(imgs, np.float32), 0, 1)
    return arr, np.array(y, np.int64)


def _apply_head(torch, F, W, mu, sd, dev):
    A = torch.cat([(F - torch.tensor(mu, device=dev)) / torch.tensor(sd, device=dev),
                   torch.ones(len(F), 1, device=dev)], 1)
    return (A @ torch.tensor(W, device=dev)).argmax(1).cpu().numpy()


def _pick(y, pred, k):
    """Choose a spread of indices: a couple of mistakes (if any) then correct."""
    wrong = [i for i in range(len(y)) if pred[i] != y[i]]
    correct = [i for i in range(len(y)) if pred[i] == y[i]]
    idx = (wrong[:2] + correct)[:k]
    return idx if len(idx) >= min(k, len(y)) else list(range(min(k, len(y))))


def _samples(imgs, y, pred, names, k, modality=None):
    out = []
    for i in _pick(y, pred, k):
        out.append({"uri": _png_uri(imgs[i]),
                    "true": names[int(y[i])], "pred": names[int(pred[i])],
                    "correct": bool(pred[i] == y[i]),
                    **({"modality": modality[i]} if modality is not None else {})})
    return out


def run():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    mmg.rk.GRID = 8

    shape_ck = json.load(open(os.path.join(MM, "dot_shape_model.json")))
    letter_ck = json.load(open(os.path.join(MM, "kid_modelA.json")))
    union_ck = json.load(open(os.path.join(MM, "mm_model.json")))
    cont_ck = json.load(open(os.path.join(MM, "vision_continue_model.json")))
    shape_g, A = shape_ck["genomes"], letter_ck["spaces"]

    # per-model bases (mirror mm_merge exactly)
    rng = np.random.default_rng(1)
    ref20 = np.stack([mmg.render_shape_crop(ti, rng)
                      for ti in np.repeat(np.arange(N_SHAPES), 150)]).astype(np.float32)
    lref = np.zeros((N_LETTERS * 100, mmg.LETTER_RES, mmg.LETTER_RES, 3), np.float32)
    for i, li in enumerate(np.repeat(np.arange(N_LETTERS), 100)):
        lref[i] = np.repeat(mmg.render_letter(mmg.LETTERS[li], rng)[..., None], 3, axis=2)

    def shape_feats(X20):
        b = Basis(torch, dev, ref20); b.set_input(X20)
        return mmg._san(torch, torch.stack([feature(torch, tp, b, g, test=True) for g in shape_g], 1))

    def cont_feats(X20):
        b = Basis(torch, dev, ref20); b.set_input(X20)
        allg = shape_g + cont_ck["new_genomes"]
        return mmg._san(torch, torch.stack([feature(torch, tp, b, g, test=True) for g in allg], 1))

    panels = {}

    # 1) shape checkpoint identifying shapes. The genomes ARE the model; refit the
    #    10-class ridge readout on the centered ref20 basis this demo uses (the
    #    saved head was fit on the tracker-attended training basis and doesn't
    #    transfer to centered crops — same closed-form readout, matches union/cont).
    Xs_tr, ys_tr = _render_shapes(24, 29)
    Xs, ys = _render_shapes(6, 30)
    acc_s, ps, _, _, _, _ = mmg._ridge_head(torch, dev, shape_feats(Xs_tr),
                                            torch.tensor(ys_tr, device=dev),
                                            shape_feats(Xs), torch.tensor(ys, device=dev), N_SHAPES)
    ps = ps.cpu().numpy()
    panels["shape"] = {"title": "Shape model → shapes", "classes": SHAPE_NAMES,
                       "acc": round(acc_s, 4),
                       "samples": _samples(Xs, ys, ps, SHAPE_NAMES, K_SHOW)}

    # 2) letter checkpoint identifying letters (fit a 26-class head on its features)
    Xl_tr, yl_tr = _render_letters(24, 31)
    Xl_te, yl_te = _render_letters(6, 32)
    Lf_tr = mmg.letter_features(torch, tp, dev, A, lref, Xl_tr)
    Lf_te = mmg.letter_features(torch, tp, dev, A, lref, Xl_te)
    acc, pl, W, mu, sd, lam = mmg._ridge_head(torch, dev, Lf_tr, torch.tensor(yl_tr, device=dev),
                                              Lf_te, torch.tensor(yl_te, device=dev), N_LETTERS)
    panels["letter"] = {"title": "Letter model → letters", "classes": list(mmg.LETTERS),
                        "acc": round(acc, 4),
                        "samples": _samples(Xl_te, yl_te, pl.cpu().numpy(), list(mmg.LETTERS), K_SHOW)}

    # 3) UNION model identifying both (fused 36-class head)
    X20, X32, ym, mod = mmg.build_dataset(torch, 6, 77)
    Sf = shape_feats(X20)
    Lf = mmg.letter_features(torch, tp, dev, A, lref, X32)
    uh = union_ck["head"]
    pu = _apply_head(torch, torch.cat([Sf, Lf], 1), uh["W"], uh["mu"], uh["sd"], dev)
    modn = ["shape" if m == 0 else "letter" for m in mod]
    panels["union"] = {"title": "Union model → both", "classes": CLASSES36,
                       "acc": round(float((pu == ym).mean()), 4),
                       "samples": _samples(X20, ym, pu, CLASSES36, K_SHOW, modality=modn)}

    # 4) CONTINUED model identifying both (grown shape model, 36-class head)
    ch = cont_ck["head"]
    pc = _apply_head(torch, cont_feats(X20), ch["W"], ch["mu"], ch["sd"], dev)
    panels["continued"] = {"title": "Continued model → both", "classes": CLASSES36,
                           "acc": round(float((pc == ym).mean()), 4),
                           "samples": _samples(X20, ym, pc, CLASSES36, K_SHOW, modality=modn)}

    os.makedirs(RD, exist_ok=True)
    with open(os.path.join(RD, "vision_demo_samples.json"), "w") as f:
        json.dump({"panels": panels, "disp_res": {"shape": mmg.SHAPE_RES,
                   "letter": mmg.LETTER_RES}}, f)
    for k, p in panels.items():
        print(f"[samples] {k:10s} acc {p['acc']:.4f}  ({len(p['samples'])} frames)", flush=True)
    print("[samples] DONE -> radial_data/vision_demo_samples.json", flush=True)


if __name__ == "__main__":
    run()
