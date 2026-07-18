"""lm_demo_trace.py - record a REAL generation trace for the /lm_demo
animation. Nothing is mocked: per step it captures the bank block
activations (down-sampled strips), the head dot-product DECOMPOSED per
block for each candidate (the actual linear algebra, split apart), the
genome structures with their live values, the topic steering bonus, the
grammar specialist's per-candidate margins, the merit pool, the softmax
and the sampled word.

  python lm/lm_demo_trace.py
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401

import json
import os
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_ROOT, "radial_data")
PROMPTS = ["he was born in the small town",
           "the chemical reaction in the water"]
N_STEPS = 14
TEMP, TOPK, LAM = 0.7, 3, 1.5


def strip(vals, n=24):
    """Down-sample a block to n cells (mean-pool) for the activation strip."""
    v = np.asarray(vals, np.float32)
    if len(v) <= n:
        return [round(float(x), 3) for x in v]
    k = len(v) // n
    return [round(float(v[i * k:(i + 1) * k].mean()), 3) for i in range(n)]


def main():
    import torch
    import lm_word_infer as li
    from radial_stack import _PRIMS
    t0 = time.time()
    li._build()
    M = li._M
    sa = li._steer_assets()
    ga = li._gram_assets()
    step_raw = M["step_raw"]
    hm, hs, Wm = M["head"]
    hm_c, hs_c = hm.cpu().numpy(), hs.cpu().numpy()
    Wm_c = Wm.cpu().numpy()
    layout = M["layout"]
    targets, w2i, tgt_i = M["targets"], M["w2i"], M["tgt_i"]
    W, s_cal = M["W"], M["s_cal"]
    print(f"[trace] model ready ({round(time.time() - t0)}s); "
          f"steer {sa.get('ready')} gram {ga.get('ready')}", flush=True)

    prims = list(_PRIMS)
    genome_defs = [{"terms": [{"c": t["c"], "prog": [[prims[p], a, b]
                                                    for p, a, b in t["prog"]]}
                              for t in g["terms"]],
                    "op": g.get("op"), "gate": bool(g.get("gate"))}
                   for g in M["genome_defs"]]

    traces = []
    for prompt in PROMPTS:
        rng = np.random.default_rng(7)
        words = prompt.lower().split()
        win = [w2i.get(w, -1) for w in words][-W:]
        while len(win) < W:
            win.insert(0, -1)
        ctx_words = list(words)
        p = sa["topic_probs"](words)
        t_star = int(p.argmax()) if p is not None and float(p.max()) >= 0.30 \
            else None
        steps, out = [], []
        for si in range(N_STEPS):
            F1, lg_t = step_raw(win)
            F1n = F1.numpy()
            Fz = (F1n - hm_c) / hs_c
            lg = lg_t.detach().cpu().numpy().astype(np.float64)
            lg = lg * s_cal / TEMP
            bonus = np.zeros_like(lg)
            if t_star is not None:
                bonus = LAM * sa["S"][:, t_star].copy()
                for wd in out:
                    kr = tgt_i.get(wd)
                    if kr is not None:
                        bonus[kr] = 0.0
                lg = lg + bonus
            rep_hits = []
            rep = 2.0 + (LAM if t_star is not None else 0.0)
            for wd in out[-16:]:
                kr = tgt_i.get(wd)
                if kr is not None:
                    lg[kr] -= rep
                    rep_hits.append(wd)
            order = np.argsort(lg)
            c3 = ctx_words[-3:] if len(ctx_words) >= 3 else \
                [None] * (3 - len(ctx_words)) + ctx_words
            sup = M["table_support"](c3[0], c3[1], c3[2])
            pool = [int(k) for k in order[-10:] if k in sup]
            admitted = []
            if t_star is not None:
                Scol = sa["S"][:, t_star]
                admitted = [int(k) for k in order[-25:]
                            if Scol[k] > 2.0 and int(k) not in pool
                            and targets[k] not in out]
                pool = pool + admitted
            top = (np.array(sorted(pool, key=lambda k: lg[k])[-5:])
                   if pool else order[-3:])
            gmarg = None
            lg2 = lg[top].astype(np.float64)
            if ga.get("ready") and len(top) > 1:
                gm = ga["margins"](ctx_words, [targets[k] for k in top])
                gmarg = [round(float(x), 3) for x in gm]
                lg2 = lg2 + (gm - gm.mean()) / (gm.std() + 1e-9)
            sel = np.argsort(lg2)[-TOPK:]
            top_f, lg2_f = top[sel], lg2[sel]
            z = lg2_f - lg2_f.max()
            pr = np.exp(z); pr /= pr.sum()
            k = int(rng.choice(top_f, p=pr)) if len(top_f) > 1 \
                else int(top_f[0])
            wd = targets[k]

            # decompose the raw head logit per block for the pool words
            cands = [int(x) for x in top]
            contribs = {}
            for c in cands:
                col = Wm_c[:-1, c]
                parts = {}
                for name, a, b in layout:
                    parts[name] = round(float(Fz[a:b] @ col[a:b])
                                        * s_cal / TEMP, 3)
                parts["bias"] = round(float(Wm_c[-1, c]) * s_cal / TEMP, 3)
                contribs[targets[c]] = parts
            blocks = {name: strip(F1n[a:b]) for name, a, b in layout}
            gvals = [round(float(x), 3)
                     for x in F1n[layout[-1][1]:layout[-1][2]]]
            steps.append({
                "ctx": [ctx_words[-W:][i] if i < len(ctx_words[-W:]) else ""
                        for i in range(min(W, len(ctx_words)))][-8:],
                "blocks": blocks, "genome_vals": gvals,
                "pool": [{"w": targets[int(x)],
                          "logit": round(float(lg[int(x)]), 3),
                          "steer": round(float(bonus[int(x)]), 3),
                          "admitted": int(x) in admitted} for x in top],
                "grammar": gmarg,
                "final": [{"w": targets[int(x)],
                           "p": round(float(pr[j]), 3)}
                          for j, x in enumerate(top_f)],
                "contribs": contribs, "pick": wd,
                "rep_penalized": rep_hits,
            })
            out.append(wd)
            ctx_words.append(wd)
            win = win[1:] + [w2i.get(wd, -1)]
        traces.append({"prompt": prompt, "output": " ".join(out),
                       "topic": (sa["topics"][t_star]
                                 if t_star is not None else None),
                       "steps": steps})
        print(f"[trace] {prompt!r} -> {' '.join(out)}", flush=True)

    doc = {"model": "module-40 wiki prose (V=5000, W=16, bank skip5k)",
           "layout": [{"name": n, "a": a, "b": b} for n, a, b in layout],
           "genomes": genome_defs, "s_cal": s_cal, "temp": TEMP,
           "lam": LAM, "traces": traces,
           "built": round(time.time() - t0)}
    with open(os.path.join(RD, "lm_demo_trace.json"), "w") as f:
        json.dump(doc, f)
    print(f"[trace] saved lm_demo_trace.json "
          f"({os.path.getsize(os.path.join(RD, 'lm_demo_trace.json')) // 1024}KB, "
          f"{round(time.time() - t0)}s)", flush=True)
    print("TRACE DONE", flush=True)


if __name__ == "__main__":
    main()
