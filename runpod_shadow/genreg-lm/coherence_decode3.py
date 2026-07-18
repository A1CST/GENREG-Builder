"""coherence_decode3.py - fluency round 3: HYBRID best-of-8 rerank.

Round 2 found the fluency-hold trade: guided t0.7 best-of-8 is the
fluency record (judge -6.88) but its reranker scores PURE fluency
(continuation tables), so among 8 candidates it systematically prefers
the least topical. Hybrid rerank: z(cont_logprob) + w * z(topic_score)
across the 8 candidates, where topic_score is the persistence topic
model's own head logit for the prompt's topic over the completion words
(decode-time model knowledge - the corpus-count judge stays independent).
One generation pass, re-ranked for every w.

  python lm/coherence_decode3.py
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
WS = [0.0, 1.0, 2.0, 4.0]

from topic_steer import PROMPTS, build_judge, load_topic_model
from coherence_decode import build_judge_ngram, cont_table_scorer, distinct2
from coherence_decode2 import load_guide_tables


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
        """Topic head logit for t_star over the completion words."""
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

    def gen(prompt, seed, temp=0.7, topk=3, lam=1.5):
        """Round 2's winning arm: guided t0.7 top3, steered lam=1.5."""
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
            c3 = ctx_words[-3:] if len(ctx_words) >= 3 else \
                [None] * (3 - len(ctx_words)) + ctx_words
            sup = table_support(c3[0], c3[1], c3[2])
            pool = [k for k in np.argsort(lg)[-10:] if k in sup]
            top = np.array(pool[-topk:] if pool else np.argsort(lg)[-3:])
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

    cache = {}
    for topic, prompts in PROMPTS.items():
        for prompt in prompts:
            cands, t_star = [], None
            for s in range(8):
                out, ts = gen(prompt, seed=100 + s)
                cands.append(out)
                t_star = ts if ts is not None else t_star
            cache[(topic, prompt)] = (cands, t_star)
    log(f"[coh3] generated 8x{len(cache)} candidates "
        f"({round(time.time() - t0)}s)")

    res = {"ws": WS, "rows": []}
    for w in WS:
        jl, d2s, held, total, samples = [], [], 0, 0, []
        for (topic, prompt), (cands, t_star) in cache.items():
            cs = np.array([cont_score(c) for c in cands])
            if t_star is not None and w > 0:
                ts_ = np.array([tscore(c, t_star) for c in cands])
                zc = (cs - cs.mean()) / (cs.std() + 1e-9)
                zt = (ts_ - ts_.mean()) / (ts_.std() + 1e-9)
                pick = int(np.argmax(zc + w * zt))
            else:
                pick = int(np.argmax(cs))
            out = cands[pick]
            text = " ".join(out)
            jl.append(judge_ng(out)); d2s.append(distinct2(out))
            _, jt = topic_judge(text)
            held += int(jt == topic); total += 1
            samples.append({"topic": topic, "prompt": prompt,
                            "completion": text, "judge": jt})
        row = {"w": w, "judge_logprob": round(float(np.mean(jl)), 4),
               "distinct2": round(float(np.mean(d2s)), 4),
               "hold": round(held / total, 4), "samples": samples}
        res["rows"].append(row)
        log(f"[coh3] w={w}: judge {row['judge_logprob']} "
            f"d2 {row['distinct2']} hold {row['hold']}")

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "coherence_decode3_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("COH3 RESULT: " + json.dumps([{k: v for k, v in r.items()
                                       if k != "samples"}
                                      for r in res["rows"]]))
    print("COH3 DONE", flush=True)


if __name__ == "__main__":
    main()
