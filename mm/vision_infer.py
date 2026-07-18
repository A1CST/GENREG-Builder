"""vision_infer.py — test the GENREG vision checkpoints yourself.

Loads a gradient-free vision checkpoint, renders random test samples of the right
kind, runs inference, and prints what the model sees vs. what it predicts (plus
accuracy). Optionally saves the sample images as PNGs so you can look at them.

    python vision_infer.py --model continued           # 8 mixed shapes+letters
    python vision_infer.py --model shape   --n 12       # 12 shapes
    python vision_infer.py --model letter  --n 12       # 12 letters
    python vision_infer.py --model union   --save out/  # + write the PNGs

Models
    shape      the shape recognizer (10 classes)          -> dot_shape_model.json
    letter     the letter recognizer (26 classes)         -> kid_modelA.json
    union      shape+letter fused into one head (36)       -> mm_model.json
    continued  the shape model GROWN to read letters (36)  -> vision_continue_model.json

Requirements: numpy, torch, pillow. Run it from inside the GENREG repo (it reuses
the repo's radial grammar so predictions are byte-identical to the page). The four
checkpoint JSONs live in ./multimodal/ (download them from the /vision_demo page).
Gradient-free throughout — there is no training here, only a closed-form readout.
"""
import argparse
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

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MM = os.path.join(_HERE, "multimodal")
SHAPE_NAMES = list(ra.SHAPE_NAMES)
CLASSES36 = SHAPE_NAMES + list(mmg.LETTERS)


def _apply_head(torch, F, W, mu, sd, dev):
    A = torch.cat([(F - torch.tensor(mu, device=dev)) / torch.tensor(sd, device=dev),
                   torch.ones(len(F), 1, device=dev)], 1)
    return (A @ torch.tensor(W, device=dev)).argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Test the GENREG vision checkpoints.")
    ap.add_argument("--model", choices=["shape", "letter", "union", "continued"],
                    default="continued")
    ap.add_argument("--n", type=int, default=8, help="samples per class-mix to render")
    ap.add_argument("--save", default=None, help="dir to write the sample PNGs into")
    args = ap.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    mmg.rk.GRID = 8
    rng = np.random.default_rng(0)

    # per-model patch-PCA bases (label-free; fit on reference renders)
    ref20 = np.stack([mmg.render_shape_crop(ti, rng)
                      for ti in np.repeat(np.arange(mmg.N_SHAPES), 150)]).astype(np.float32)
    lref = np.zeros((mmg.N_LETTERS * 100, mmg.LETTER_RES, mmg.LETTER_RES, 3), np.float32)
    for i, li in enumerate(np.repeat(np.arange(mmg.N_LETTERS), 100)):
        lref[i] = np.repeat(mmg.render_letter(mmg.LETTERS[li], rng)[..., None], 3, axis=2)

    def shape_feats(X20, genomes):
        b = Basis(torch, dev, ref20); b.set_input(X20)
        return mmg._san(torch, torch.stack([feature(torch, tp, b, g, test=True) for g in genomes], 1))

    # build the test batch + run the requested checkpoint
    if args.model in ("union", "continued"):
        X20, X32, y, mod = mmg.build_dataset(torch, args.n, 123)
        names, disp = CLASSES36, X20
        if args.model == "union":
            ck = json.load(open(os.path.join(MM, "mm_model.json")))
            Sf = shape_feats(X20, json.load(open(os.path.join(MM, "dot_shape_model.json")))["genomes"])
            Lf = mmg.letter_features(torch, tp, dev, json.load(
                open(os.path.join(MM, "kid_modelA.json")))["spaces"], lref, X32)
            F = torch.cat([Sf, Lf], 1)
        else:
            ck = json.load(open(os.path.join(MM, "vision_continue_model.json")))
            shape_g = json.load(open(os.path.join(MM, "dot_shape_model.json")))["genomes"]
            F = shape_feats(X20, shape_g + ck["new_genomes"])
        h = ck["head"]
        pred = _apply_head(torch, F, h["W"], h["mu"], h["sd"], dev)
    elif args.model == "shape":
        # genomes are the model; refit the closed-form 10-class readout on ref20
        Xtr = np.stack([mmg.render_shape_crop(ti, rng)
                        for ti in np.repeat(np.arange(mmg.N_SHAPES), 24)]).astype(np.float32)
        ytr = np.repeat(np.arange(mmg.N_SHAPES), 24)
        idx = rng.permutation(mmg.N_SHAPES * args.n)
        X20 = np.stack([mmg.render_shape_crop(ti, rng)
                        for ti in np.repeat(np.arange(mmg.N_SHAPES), args.n)]).astype(np.float32)[idx]
        y = np.repeat(np.arange(mmg.N_SHAPES), args.n)[idx]
        g = json.load(open(os.path.join(MM, "dot_shape_model.json")))["genomes"]
        _, pred, *_ = mmg._ridge_head(torch, dev, shape_feats(Xtr, g), torch.tensor(ytr, device=dev),
                                      shape_feats(X20, g), torch.tensor(y, device=dev), mmg.N_SHAPES)
        pred, names, disp = pred.cpu().numpy(), SHAPE_NAMES, X20
    else:  # letter
        def rl(npc, seed):
            r = np.random.default_rng(seed); ims, ys = [], []
            for li in range(mmg.N_LETTERS):
                for _ in range(npc):
                    ims.append(np.repeat(mmg.render_letter(mmg.LETTERS[li], r)[..., None], 3, 2)); ys.append(li)
            return np.clip(np.array(ims, np.float32), 0, 1), np.array(ys)
        A = json.load(open(os.path.join(MM, "kid_modelA.json")))["spaces"]
        Xtr, ytr = rl(24, 41); X20, y = rl(args.n, 42)
        _, pred, *_ = mmg._ridge_head(torch, dev, mmg.letter_features(torch, tp, dev, A, lref, Xtr),
                                      torch.tensor(ytr, device=dev),
                                      mmg.letter_features(torch, tp, dev, A, lref, X20),
                                      torch.tensor(y, device=dev), mmg.N_LETTERS)
        pred, names, disp = pred.cpu().numpy(), list(mmg.LETTERS), X20

    acc = float((pred == y).mean())
    print(f"\n  model = {args.model}   samples = {len(y)}   accuracy = {acc:.4f}\n")
    for i in range(len(y)):
        mark = "OK " if pred[i] == y[i] else "XX "
        print(f"  [{mark}] true {names[int(y[i])]:>8s}   ->  pred {names[int(pred[i])]:>8s}")

    if args.save:
        from PIL import Image
        os.makedirs(args.save, exist_ok=True)
        for i in range(len(y)):
            a = (np.clip(disp[i], 0, 1) * 255).astype(np.uint8)
            Image.fromarray(a, "RGB").resize((160, 160), Image.NEAREST).save(
                os.path.join(args.save, f"{i:02d}_{names[int(y[i])]}_as_{names[int(pred[i])]}.png"))
        print(f"\n  wrote {len(y)} PNGs to {args.save}")


if __name__ == "__main__":
    main()
