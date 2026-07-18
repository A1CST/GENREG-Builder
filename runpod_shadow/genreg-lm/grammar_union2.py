"""grammar_union2.py - PER-STEP grammar union.

Round 1 (post-hoc rerank) was a null: 8 candidates from the same decode
are near-identical in grammar, so picking among them adds nothing. The
real union is per-step: at every decode step the grammar specialist
scores each candidate word in the merit pool - margin of (last T-1
context words + candidate) - and its z-score joins the pool logits.
Grammar shapes the path, not the postmortem.

  python lm/grammar_union2.py
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
WGS = [0.0, 1.0, 2.0, 4.0]

from topic_steer import PROMPTS, build_judge
from coherence_decode import build_judge_ngram, cont_table_scorer, distinct2
from grammar_union import load_grammar_scorer
import radial_temporal as rt


def main():
    t0 = time.time()

    def log(m):
        print(m, flush=True)

    import torch
    import lm_word_infer as lwi
    lwi._build()
    M = lwi._M
    sa = lwi._steer_assets()
    step, w2i, targets = M["step"], M["w2i"], M["targets"]
    W, s_cal = M["W"], M["s_cal"]
    tgt_i = M["tgt_i"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    judge_ng = build_judge_ngram()
    topic_judge = build_judge(sa["topics"])
    cont_score = cont_table_scorer()
    _, gm = load_grammar_scorer(torch, dev)

    # per-window grammar margin, batched over candidate words
    zp = np.load(os.path.join(RD, "embed_rs_prev.npz"), allow_pickle=True)
    zn = np.load(os.path.join(RD, "embed_rs_next.npz"), allow_pickle=True)
    gvocab = {str(w): i for i, w in enumerate(zp["vocab"])}
    gE = np.concatenate([zp["feat"], zn["feat"]], 1).astype(np.float32)
    T = gm["T"]
    gcmu = np.array(gm["cmu"], np.float32)
    gcsd = np.array(gm["csd"], np.float32)
    gfmu = torch.tensor(gm["fmu"], device=dev)
    gfsd = torch.tensor(gm["fsd"], device=dev)
    ghm = torch.tensor(gm["head_mu"], device=dev)
    ghs = torch.tensor(gm["head_sd"], device=dev)
    gWm = torch.tensor(gm["head_W"], device=dev)

    def gram_margins(ctx_words, cand_words):
        """Margin of (last T-1 ctx + cand) for each candidate, batched."""
        base = ctx_words[-(T - 1):]
        if len(base) < T - 1:
            base = [None] * (T - 1 - len(base)) + list(base)
        Xb = np.zeros((len(cand_words), T, gE.shape[1]), np.float32)
        for t, w in enumerate(base):
            i = gvocab.get(w) if w else None
            if i is not None:
                Xb[:, t] = gE[i]
        for c, w in enumerate(cand_words):
            i = gvocab.get(w)
            if i is not None:
                Xb[c, T - 1] = gE[i]
        Xb = np.clip((Xb - gcmu) / gcsd, -8, 8)
        F = torch.tensor(Xb, device=dev)
        cols = [rt._finite(torch, rt.temporal_feat(torch, F, g))
                for g in gm["genomes"]]
        Ft = ((torch.stack(cols, 1) - gfmu) / gfsd).clamp(-8, 8)
        s = torch.hstack([(Ft - ghm) / ghs,
                          torch.ones(len(cand_words), 1, device=dev)]) @ gWm
        return (s[:, 1] - s[:, 0]).cpu().numpy()

    def gen(prompt, seed, wg, temp=0.7, topk=3, lam=1.5):
        rng = np.random.default_rng(seed)
        words = [w for w in prompt.lower().split() if w]
        win = [w2i.get(w, -1) for w in words][-W:]
        while len(win) < W:
            win.insert(0, -1)
        ctx_words = list(words)
        p = sa["topic_probs"](words)
        t_star = int(p.argmax()) if p is not None and float(p.max()) >= 0.30 \
            else None
        out = []
        for _ in range(24):
            lg = step(win).detach().cpu().numpy().astype(np.float64)
            lg = lg * s_cal / temp
            if t_star is not None:
                bonus = lam * sa["S"][:, t_star].copy()
                for wd in out:
                    kr = tgt_i.get(wd)
                    if kr is not None:
                        bonus[kr] = 0.0
                lg = lg + bonus
            rep = 2.0 + (lam if t_star is not None else 0.0)
            for wd in out[-16:]:
                kr = tgt_i.get(wd)
                if kr is not None:
                    lg[kr] -= rep
            order = np.argsort(lg)
            c3 = ctx_words[-3:] if len(ctx_words) >= 3 else \
                [None] * (3 - len(ctx_words)) + ctx_words
            sup = M["table_support"](c3[0], c3[1], c3[2])
            pool = [int(k) for k in order[-10:] if k in sup]
            if t_star is not None:
                Scol = sa["S"][:, t_star]
                pool += [int(k) for k in order[-25:]
                         if Scol[k] > 2.0 and int(k) not in pool
                         and targets[k] not in out]
            top = (np.array(sorted(pool, key=lambda k: lg[k])[-5:])
                   if pool else order[-3:])
            if wg > 0 and len(top) > 1:       # the per-step grammar vote
                gmg = gram_margins(ctx_words, [targets[k] for k in top])
                zg = (gmg - gmg.mean()) / (gmg.std() + 1e-9)
                lg2 = lg[top] + wg * zg
            else:
                lg2 = lg[top]
            sel = np.argsort(lg2)[-topk:]
            top = top[sel]
            if len(top) == 1:
                k = int(top[0])
            else:
                z = lg2[sel] - lg2[sel].max()
                pr = np.exp(z); pr /= pr.sum()
                k = int(rng.choice(top, p=pr))
            wd = targets[k]
            out.append(wd)
            ctx_words.append(wd)
            win = win[1:] + [w2i.get(wd, -1)]
        return out, t_star

    res = {"wgs": WGS, "rows": []}
    for wg in WGS:
        jl, d2s, held, total, samples = [], [], 0, 0, []
        for topic, prompts in PROMPTS.items():
            for prompt in prompts:
                cands, t_star = [], None
                for s in range(8):
                    o, ts = gen(prompt, 1000 + s, wg)
                    cands.append(o)
                    t_star = ts if ts is not None else t_star
                cs = np.array([cont_score(c) for c in cands])
                if t_star is not None:
                    ts_ = np.array([sa["tscore"](c, t_star) for c in cands])
                    zc = (cs - cs.mean()) / (cs.std() + 1e-9)
                    zt = (ts_ - ts_.mean()) / (ts_.std() + 1e-9)
                    out = cands[int(np.argmax(zc + zt))]
                else:
                    out = cands[int(np.argmax(cs))]
                text = " ".join(out)
                jl.append(judge_ng(out)); d2s.append(distinct2(out))
                _, jt = topic_judge(text)
                held += int(jt == topic); total += 1
                samples.append({"topic": topic, "prompt": prompt,
                                "completion": text, "judge": jt})
        row = {"wg": wg, "judge_logprob": round(float(np.mean(jl)), 4),
               "distinct2": round(float(np.mean(d2s)), 4),
               "hold": round(held / total, 4), "samples": samples}
        res["rows"].append(row)
        log(f"[union2] wg={wg}: judge {row['judge_logprob']} "
            f"d2 {row['distinct2']} hold {row['hold']}")

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "grammar_union2_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("UNION2 RESULT: " + json.dumps([{k: v for k, v in r.items()
                                         if k != "samples"}
                                        for r in res["rows"]]))
    print("UNION2 DONE", flush=True)


if __name__ == "__main__":
    main()
