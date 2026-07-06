"""Sample text from a trained lm_char_v1 / composed checkpoint.

Usage (from repo root):
    python -m genreg_train.lm_sample                 # newest checkpoint, defaults
    python -m genreg_train.lm_sample --list          # list checkpoints + scores
    python -m genreg_train.lm_sample --ckpt runs/lm/<id>/checkpoint.npz \\
        --prompt "the old man " --temp 0.7 --len 200 --primed

Notes
-----
- Char-level model at ~31% top-1 produces word SHAPES and fragments, not
  sentences (see documentation/LLM__LM_RULES.md corpus ceilings).
- Low temperature collapses toward the marginal (spaces / 'e') — the
  documented exposure-gap attractor. temp 0.6-0.9 shows the model's real
  distribution. --primed seeds the recurrent state with real corpus text
  (closest to how held-out top-1 is measured).
"""
import argparse
import glob
import json
import os

import numpy as np


def _load(ck):
    from .genreg_lm import LMPopulation, DEFAULTS, load_char_corpus, sample_windows
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pop = LMPopulation({**DEFAULTS, "pop": 400}, dev)
    from .genreg_lm import load_population
    load_population(pop, ck)
    ids = load_char_corpus()
    # champion = best held-out fitness in the saved population
    W = sample_windows(ids, 256, 64, np.random.default_rng(31337))
    fit, top1, _ = pop.evaluate(W, 8)
    b = int(fit.argmax())
    return pop, b, ids, float(top1[b])


def list_checkpoints():
    rows = []
    for f in glob.glob("runs/lm/*/summary.json") + glob.glob("runs/enc/*/summary.json"):
        try:
            s = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        rid = os.path.basename(os.path.dirname(f))
        ro = s.get("rollout") or {}
        rows.append((os.path.getmtime(f), rid, s.get("best", {}).get("score"),
                     ro.get("top1")))
    for _mt, rid, t1, rt in sorted(rows, reverse=True):
        print(f"  {rid}  top1={t1}  rollout_top1={rt}")


def main():
    ap = argparse.ArgumentParser(description="Sample from an lm_char checkpoint")
    ap.add_argument("--ckpt", default=None, help="checkpoint.npz (default: newest under runs/lm)")
    ap.add_argument("--prompt", default="the ")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--len", type=int, default=160)
    ap.add_argument("--primed", action="store_true",
                    help="seed hidden state with real corpus text before the prompt")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        list_checkpoints()
        return

    ck = a.ckpt
    if ck is None:
        cks = sorted(glob.glob("runs/lm/*/checkpoint.npz"), key=os.path.getmtime)
        if not cks:
            print("no checkpoints under runs/lm/")
            return
        ck = cks[-1]
    from .genreg_lm import generate, CHARS
    pop, b, ids, t1 = _load(ck)
    print(f"checkpoint: {os.path.basename(os.path.dirname(ck))}  (champion top1 {t1*100:.2f}%)")

    prompt = a.prompt
    if a.primed:
        rng = np.random.default_rng(a.seed or 0)
        st = int(rng.integers(0, len(ids) - 200))
        real = "".join(CHARS[i] if i < len(CHARS) else "?" for i in ids[st:st + 48])
        prompt = real + prompt
        print(f"primed with real text: {real!r}")

    out = generate(pop, b, prompt, length=a.len, temperature=a.temp, seed=a.seed)
    print(f"\nprompt: {a.prompt!r}  temp={a.temp}")
    print(f"{a.prompt}{out}")


if __name__ == "__main__":
    main()
