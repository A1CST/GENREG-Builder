"""coherence_decode2.py - fluency round 2: SHARP decode + TABLE-GUIDED decode.

The user's verdict on round 1: fluency is still garbage. Two hypotheses,
both decode-time and both measurable:

  SHARP    the crank model is 0.56 top-1 but round 1 never went below
           temp 0.7 / top-3 - near-uniform top-5 sampling throws away most
           of that accuracy. Arms: temp 0.5/top3, temp 0.3/top2, GREEDY
           (top-1 + repetition penalty).
  GUIDED   the exposure gap: generated context drifts off the
           table-covered manifold (has-target top-1 0.7262) into the blind
           zone (0.3123). Arm: candidate pool = model top-10 INTERSECTED
           with table-supported words for the current context (tri/bi +
           quad/skip top-20); fallback to model top-3 when tables are
           silent.

Steering fixed at the deployed lam=1.5 (the goal still needs topic-hold);
one unsteered greedy arm isolates steering's fluency cost. Same judges as
round 1 (independent n-gram slice, distinct-2, corpus topic judge),
samples verbatim. Best arm additionally gets best-of-8.

  python lm/coherence_decode2.py
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401

import json
import os
import pickle
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_ROOT, "radial_data")
N_GEN = 24
LAM = 1.5

from topic_steer import PROMPTS, build_judge
from coherence_decode import build_judge_ngram, cont_table_scorer, distinct2

ARMS = [
    {"tag": "t0.9 top5 (deployed r1)", "temp": 0.9, "topk": 5, "guided": 0, "lam": LAM},
    {"tag": "t0.5 top3",               "temp": 0.5, "topk": 3, "guided": 0, "lam": LAM},
    {"tag": "t0.3 top2",               "temp": 0.3, "topk": 2, "guided": 0, "lam": LAM},
    {"tag": "greedy",                  "temp": 0.3, "topk": 1, "guided": 0, "lam": LAM},
    {"tag": "greedy UNSTEERED",        "temp": 0.3, "topk": 1, "guided": 0, "lam": 0.0},
    {"tag": "guided t0.7",             "temp": 0.7, "topk": 3, "guided": 1, "lam": LAM},
    {"tag": "guided t0.5",             "temp": 0.5, "topk": 3, "guided": 1, "lam": LAM},
]


def load_guide_tables():
    with open(os.path.join(RD, "lm_cont_tables.pkl"), "rb") as f:
        uni_c, bi_c, tri_c = pickle.load(f)
    with open(os.path.join(RD, "lm_skip5k_tables.pkl"), "rb") as f:
        quad_t, skipA_t, skipB_t = pickle.load(f)
    return uni_c, bi_c, tri_c, quad_t, skipA_t, skipB_t


def main():
    t0 = time.time()
    log_lines = []

    def log(m):
        log_lines.append(m); print(m, flush=True)

    log("[coh2] loading generator (inference pack)...")
    import lm_word_infer as lwi
    lwi._build()
    M = lwi._M
    sa = lwi._steer_assets()
    step, w2i, targets = M["step"], M["w2i"], M["targets"]
    W, s_cal = M["W"], M["s_cal"]
    tgt_i = M["tgt_i"]
    log(f"[coh2] ready in {M['build_seconds']}s; s_cal {s_cal}")

    judge_ng = build_judge_ngram()
    topic_judge = build_judge(sa["topics"])
    rerank = cont_table_scorer()
    uni_c, bi_c, tri_c, quad_t, skipA_t, skipB_t = load_guide_tables()

    def table_support(w0, w1, w2, n_each=20):
        """Target-vocab words the tables expect after this context."""
        sup = set()
        for d in (tri_c.get((w1, w2)), bi_c.get(w2), quad_t.get((w0, w1, w2)),
                  skipA_t.get((w0, w2)), skipB_t.get((w0, w1))):
            if d:
                for w in sorted(d, key=d.get, reverse=True)[:n_each]:
                    k = tgt_i.get(w)
                    if k is not None:
                        sup.add(k)
        return sup

    def gen(prompt, temp, topk, guided, lam, seed):
        rng = np.random.default_rng(seed)
        words = [w for w in prompt.lower().split() if w]
        win = [w2i.get(w, -1) for w in words][-W:]
        while len(win) < W:
            win.insert(0, -1)
        ctx_words = list(words)
        p = sa["topic_probs"](words) if lam > 0 else None
        t_star = int(p.argmax()) if p is not None and float(p.max()) >= 0.30 \
            else None
        out = []
        for _ in range(N_GEN):
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
            if guided:
                c3 = ctx_words[-3:] if len(ctx_words) >= 3 else \
                    [None] * (3 - len(ctx_words)) + ctx_words
                sup = table_support(c3[0], c3[1], c3[2])
                pool = [k for k in np.argsort(lg)[-10:] if k in sup]
                top = np.array(pool[-topk:] if pool
                               else np.argsort(lg)[-3:])
            else:
                top = np.argsort(lg)[-topk:]
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
        return out

    res = {"arms": [], "n_gen": N_GEN}
    best = None
    for arm in ARMS:
        jl, d2, held, total, samples = [], [], 0, 0, []
        for topic, prompts in PROMPTS.items():
            for pi, prompt in enumerate(prompts):
                out = gen(prompt, arm["temp"], arm["topk"], arm["guided"],
                          arm["lam"], seed=17 + pi)
                text = " ".join(out)
                jl.append(judge_ng(out))
                d2.append(distinct2(out))
                _, jt = topic_judge(text)
                held += int(jt == topic); total += 1
                samples.append({"topic": topic, "prompt": prompt,
                                "completion": text, "judge": jt})
        row = {**arm, "judge_logprob": round(float(np.mean(jl)), 4),
               "distinct2": round(float(np.mean(d2)), 4),
               "hold": round(held / total, 4), "samples": samples}
        res["arms"].append(row)
        log(f"[coh2] {arm['tag']}: judge {row['judge_logprob']} "
            f"d2 {row['distinct2']} hold {row['hold']}")
        score = row["judge_logprob"] + 2.0 * row["hold"] \
            + 2.0 * row["distinct2"]              # fluency+hold+no-loops
        if arm["lam"] > 0 and (best is None or score > best[1]):
            best = (row, score)

    # best-of-8 on the winning steered arm
    win_arm = best[0]
    jl, d2, held, total, samples = [], [], 0, 0, []
    for topic, prompts in PROMPTS.items():
        for pi, prompt in enumerate(prompts):
            cands = [gen(prompt, win_arm["temp"], win_arm["topk"],
                         win_arm["guided"], win_arm["lam"], seed=100 + s)
                     for s in range(8)]
            out = max(cands, key=rerank)
            text = " ".join(out)
            jl.append(judge_ng(out)); d2.append(distinct2(out))
            _, jt = topic_judge(text)
            held += int(jt == topic); total += 1
            samples.append({"topic": topic, "prompt": prompt,
                            "completion": text, "judge": jt})
    row = {"tag": win_arm["tag"] + " best-of-8", "temp": win_arm["temp"],
           "topk": win_arm["topk"], "guided": win_arm["guided"],
           "lam": win_arm["lam"],
           "judge_logprob": round(float(np.mean(jl)), 4),
           "distinct2": round(float(np.mean(d2)), 4),
           "hold": round(held / total, 4), "samples": samples}
    res["arms"].append(row)
    log(f"[coh2] {row['tag']}: judge {row['judge_logprob']} "
        f"d2 {row['distinct2']} hold {row['hold']}")

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "coherence_decode2_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("COH2 RESULT: " + json.dumps([{k: v for k, v in a.items()
                                       if k != "samples"}
                                      for a in res["arms"]]))
    print("COH2 DONE", flush=True)


if __name__ == "__main__":
    main()
