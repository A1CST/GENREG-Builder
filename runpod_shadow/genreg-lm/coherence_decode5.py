"""coherence_decode5.py - fluency round 5: the MERIT POOL.

Round 4 overshot (hold 0.94, judge -10.4): admitted topical words were
positionally privileged by the pool slice and lam was double-boosted.
Fix: the union pool (table-supported + admitted topical) is ranked by the
STEERED LOGITS and top-k is taken on merit; lam stays flat at 1.5.

Round 3's lesson: at sharp/guided decode no rerank can recover topic-hold
because the candidates contain no topic words - (1) the steering bonus is
constant while the scaled logits sharpen (lam is out-competed exactly
when fluency improves), and (2) the guided pool comes from continuation
tables, which rarely offer topical words after dialogue-register context.

Fixes, both in the candidate pool:
  lam_eff  = lam * (0.9 / temp)   steering keeps its measured strength
                                  relative to the softmax scale it was
                                  tuned at (temp 0.9)
  pool     = (model top-10 INTERSECT table support)  UNION
             (strong topic words [S > 2.0] inside model top-25)
             - fluency stays table-anchored, topical words are ADMITTED
             when the model itself rates them plausible.

Arms: t0.7/top3 and t0.5/top3, each single + best-of-8 hybrid (w=1).

  python lm/coherence_decode4.py
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

from topic_steer import PROMPTS, build_judge, load_topic_model
from coherence_decode import build_judge_ngram, cont_table_scorer, distinct2
from coherence_decode2 import load_guide_tables

ARMS = [{"temp": 0.7, "topk": 3}, {"temp": 0.5, "topk": 3}]
LAM = 1.5


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

    judge_ng = build_judge_ngram()
    topic_judge = build_judge(sa["topics"])
    cont_score = cont_table_scorer()
    uni_c, bi_c, tri_c, quad_t, skipA_t, skipB_t = load_guide_tables()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tm, responses, topic_probs, hm, hs, Wm = load_topic_model(torch, dev)

    def tscore(words, t_star):
        r = responses(words)
        if r is None:
            return 0.0
        acc = r.mean(0, keepdim=True)
        lg = torch.hstack([(acc - hm) / hs,
                           torch.ones(1, 1, device=dev)]) @ Wm
        return float(lg[0, t_star])

    def table_support(w0, w1, w2, n_each=20):
        sup = set()
        for d in (tri_c.get((w1, w2)), bi_c.get(w2), quad_t.get((w0, w1, w2)),
                  skipA_t.get((w0, w2)), skipB_t.get((w0, w1))):
            if d:
                for w in sorted(d, key=d.get, reverse=True)[:n_each]:
                    k = tgt_i.get(w)
                    if k is not None:
                        sup.add(k)
        return sup

    def gen(prompt, seed, temp, topk):
        rng = np.random.default_rng(seed)
        words = [w for w in prompt.lower().split() if w]
        win = [w2i.get(w, -1) for w in words][-W:]
        while len(win) < W:
            win.insert(0, -1)
        ctx_words = list(words)
        p = sa["topic_probs"](words)
        t_star = int(p.argmax()) if p is not None and float(p.max()) >= 0.30 \
            else None
        lam_eff = LAM
        out = []
        for _ in range(24):
            lg = step(win).detach().cpu().numpy().astype(np.float64)
            lg = lg * s_cal / temp
            if t_star is not None:
                bonus = lam_eff * sa["S"][:, t_star].copy()
                for wd in out:
                    kr = tgt_i.get(wd)
                    if kr is not None:
                        bonus[kr] = 0.0
                lg = lg + bonus
            rep = 2.0 + (lam_eff if t_star is not None else 0.0)
            for wd in out[-16:]:
                kr = tgt_i.get(wd)
                if kr is not None:
                    lg[kr] -= rep
            c3 = ctx_words[-3:] if len(ctx_words) >= 3 else \
                [None] * (3 - len(ctx_words)) + ctx_words
            sup = table_support(c3[0], c3[1], c3[2])
            order = np.argsort(lg)
            pool = [k for k in order[-10:] if k in sup]
            if t_star is not None:
                Scol = sa["S"][:, t_star]
                topical = [int(k) for k in order[-25:]
                           if Scol[k] > 2.0 and k not in pool
                           and targets[k] not in out]
                pool = pool + topical
            top = (np.array(sorted(pool, key=lambda k: lg[k])[-topk:])
                   if pool else order[-3:])
            if len(top) == 1:
                k = int(top[0])
            else:
                z = lg[top] - lg[top].max()
                pr = np.exp(z); pr /= pr.sum()
                k = int(rng.choice(top, p=pr))
            wd = targets[k]
            out.append(wd)
            ctx_words.append(wd)
            win = win[1:] + [w2i.get(wd, -1)]
        return out, t_star

    res = {"arms": []}
    for arm in ARMS:
        cache = {}
        for topic, prompts in PROMPTS.items():
            for prompt in prompts:
                cands, t_star = [], None
                for s in range(8):
                    o, ts = gen(prompt, 100 + s, arm["temp"], arm["topk"])
                    cands.append(o)
                    t_star = ts if ts is not None else t_star
                cache[(topic, prompt)] = (cands, t_star)

        for mode in ("single", "best8"):
            jl, d2s, held, total, samples = [], [], 0, 0, []
            for (topic, prompt), (cands, t_star) in cache.items():
                if mode == "single":
                    out = cands[0]
                else:
                    cs = np.array([cont_score(c) for c in cands])
                    if t_star is not None:
                        ts_ = np.array([tscore(c, t_star) for c in cands])
                        zc = (cs - cs.mean()) / (cs.std() + 1e-9)
                        zt = (ts_ - ts_.mean()) / (ts_.std() + 1e-9)
                        pick = int(np.argmax(zc + zt))
                    else:
                        pick = int(np.argmax(cs))
                    out = cands[pick]
                text = " ".join(out)
                jl.append(judge_ng(out)); d2s.append(distinct2(out))
                _, jt = topic_judge(text)
                held += int(jt == topic); total += 1
                samples.append({"topic": topic, "prompt": prompt,
                                "completion": text, "judge": jt})
            row = {"tag": f"t{arm['temp']} top{arm['topk']} {mode}",
                   **arm, "mode": mode,
                   "judge_logprob": round(float(np.mean(jl)), 4),
                   "distinct2": round(float(np.mean(d2s)), 4),
                   "hold": round(held / total, 4), "samples": samples}
            res["arms"].append(row)
            log(f"[coh5] {row['tag']}: judge {row['judge_logprob']} "
                f"d2 {row['distinct2']} hold {row['hold']}")

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "coherence_decode5_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("COH5 RESULT: " + json.dumps([{k: v for k, v in a.items()
                                       if k != "samples"}
                                      for a in res["arms"]]))
    print("COH5 DONE", flush=True)


if __name__ == "__main__":
    main()
