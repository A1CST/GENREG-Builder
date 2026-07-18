"""coherence_decode.py - the COHERENCE lever: decode-grid + best-of-k rerank.

Modules 33-35 closed vocabulary and topic-holding; the open gap to the goal
(consistent sentences that hold topic) is local coherence. This measures the
cheapest levers - all decode-time, no retraining:

  grid: temp {0.7, 0.9} x top-k {3, 5}, steering fixed at the deployed
        config (lam=1.5, evidence floor), plus best-of-8 rerank arms where
        8 seeded continuations are generated and the one with the best
        mean continuation-table log-prob (the model's OWN tables - decode-
        time knowledge, not the judge) is kept.

Honest metrics, none of which is the reranker:
  - judge log-prob: mean trigram log P(w|context) under an n-gram model
    built from a FRESH corpus slice (seek 50MB; cont tables use 30-46MB,
    train/test regions 10-21MB - three disjoint slices)
  - distinct-2: unique bigrams / bigrams (loop/repetition detector)
  - topic-hold under the module-33 corpus judge (held-out articles)
  - samples verbatim per config

  python coherence_decode.py
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
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")

N_GEN = 24
LAM = 1.5
GRID = [
    {"temp": 0.9, "topk": 5, "best_of": 1},   # deployed decode (baseline)
    {"temp": 0.7, "topk": 5, "best_of": 1},
    {"temp": 0.9, "topk": 3, "best_of": 1},
    {"temp": 0.7, "topk": 3, "best_of": 1},
    {"temp": 0.9, "topk": 5, "best_of": 8},
    {"temp": 0.7, "topk": 3, "best_of": 8},
]

from topic_steer import PROMPTS, build_judge


def build_judge_ngram():
    """Independent n-gram judge from a corpus slice none of the model's
    tables or data regions touch (seek 50MB, 16MB)."""
    from radial_lm import _clean
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as fh:
        fh.seek(50_000_000)
        toks = _clean(fh.read(16_000_000)).split()
    uni, bi, tri = {}, {}, {}
    for i in range(len(toks) - 2):
        uni[toks[i]] = uni.get(toks[i], 0) + 1
        bi.setdefault(toks[i], {})
        bi[toks[i]][toks[i + 1]] = bi[toks[i]].get(toks[i + 1], 0) + 1
        key = (toks[i], toks[i + 1])
        tri.setdefault(key, {})
        tri[key][toks[i + 2]] = tri[key].get(toks[i + 2], 0) + 1
    Vj = len(uni)
    n_uni = sum(uni.values())

    def logp(w1, w2, w):
        if (w1, w2) in tri:
            d = tri[(w1, w2)]
            return np.log((d.get(w, 0) + 0.5) / (sum(d.values()) + 0.5 * Vj))
        if w2 in bi:
            d = bi[w2]
            return np.log((d.get(w, 0) + 0.5) / (sum(d.values()) + 0.5 * Vj))
        return np.log((uni.get(w, 0) + 0.5) / (n_uni + 0.5 * Vj))

    def score(words):
        if len(words) < 3:
            return None
        return float(np.mean([logp(words[i], words[i + 1], words[i + 2])
                              for i in range(len(words) - 2)]))
    return score


def cont_table_scorer():
    """Reranker: the model's own continuation tables (NOT the judge)."""
    with open(os.path.join(RD, "lm_cont_tables.pkl"), "rb") as f:
        uni_c, bi_c, tri_c = pickle.load(f)
    Vc = len(uni_c)
    n_uni = sum(uni_c.values())

    def logp(w1, w2, w):
        if (w1, w2) in tri_c:
            d = tri_c[(w1, w2)]
            return np.log((d.get(w, 0) + 0.5) / (sum(d.values()) + 0.5 * Vc))
        if w2 in bi_c:
            d = bi_c[w2]
            return np.log((d.get(w, 0) + 0.5) / (sum(d.values()) + 0.5 * Vc))
        return np.log((uni_c.get(w, 0) + 0.5) / (n_uni + 0.5 * Vc))

    def score(words):
        if len(words) < 3:
            return -1e9
        return float(np.mean([logp(words[i], words[i + 1], words[i + 2])
                              for i in range(len(words) - 2)]))
    return score


def distinct2(words):
    if len(words) < 2:
        return 1.0
    bgs = list(zip(words, words[1:]))
    return round(len(set(bgs)) / len(bgs), 4)


def main():
    t0 = time.time()
    log_lines = []

    def log(m):
        log_lines.append(m); print(m, flush=True)

    log("[coh] building generator (frozen replay)...")
    import lm_word_infer as lwi
    lwi._build()
    M = lwi._M
    sa = lwi._steer_assets()
    log(f"[coh] ready: V={M['V']} steer={'ok' if sa.get('ready') else 'OFF'}")
    step, w2i, targets = M["step"], M["w2i"], M["targets"]
    W, s_cal = M["W"], M["s_cal"]

    judge_ng = build_judge_ngram()
    topic_judge = build_judge(sa["topics"])
    rerank = cont_table_scorer()

    def gen(prompt, temp, topk, seed):
        rng = np.random.default_rng(seed)
        words = [w for w in prompt.lower().split() if w]
        win = [w2i.get(w, -1) for w in words][-W:]
        while len(win) < W:
            win.insert(0, -1)
        p = sa["topic_probs"](words)
        t_star = int(p.argmax()) if p is not None and float(p.max()) >= 0.30 \
            else None
        out = []
        for _ in range(N_GEN):
            lg = step(win).detach().cpu().numpy().astype(np.float64)
            lg = lg * s_cal / temp
            if t_star is not None:
                bonus = LAM * sa["S"][:, t_star].copy() \
                    if isinstance(sa["S"], np.ndarray) \
                    else LAM * sa["S"][:, t_star].cpu().numpy()
                for wd in out:
                    kr = M["tgt_i"].get(wd)
                    if kr is not None:
                        bonus[kr] = 0.0
                lg = lg + bonus
            for wd in out[-16:]:
                kr = M["tgt_i"].get(wd)
                if kr is not None:
                    lg[kr] -= 2.0 + LAM
            top = np.argsort(lg)[-topk:]
            z = lg[top] - lg[top].max()
            pr = np.exp(z); pr /= pr.sum()
            k = int(rng.choice(top, p=pr))
            out.append(targets[k])
            win = win[1:] + [w2i.get(targets[k], -1)]
        return out

    res = {"grid": GRID, "n_gen": N_GEN, "lam": LAM, "configs": []}
    for cfg in GRID:
        jl, d2, held, total = [], [], 0, 0
        samples = []
        for topic, prompts in PROMPTS.items():
            for pi, prompt in enumerate(prompts):
                if cfg["best_of"] == 1:
                    out = gen(prompt, cfg["temp"], cfg["topk"], seed=17 + pi)
                else:
                    cands = [gen(prompt, cfg["temp"], cfg["topk"], seed=100 + s)
                             for s in range(cfg["best_of"])]
                    out = max(cands, key=rerank)
                text = " ".join(out)
                jl.append(judge_ng(out))
                d2.append(distinct2(out))
                _, jt = topic_judge(text)
                held += int(jt == topic); total += 1
                samples.append({"topic": topic, "prompt": prompt,
                                "completion": text, "judge": jt})
        tag = (f"temp={cfg['temp']} top{cfg['topk']}" +
               (f" best-of-{cfg['best_of']}" if cfg["best_of"] > 1 else ""))
        row = {"tag": tag, **cfg,
               "judge_logprob": round(float(np.mean(jl)), 4),
               "distinct2": round(float(np.mean(d2)), 4),
               "hold": round(held / total, 4), "samples": samples}
        res["configs"].append(row)
        log(f"[coh] {tag}: judge logprob {row['judge_logprob']} "
            f"distinct2 {row['distinct2']} hold {row['hold']}")

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "coherence_decode_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("COH RESULT: " + json.dumps([{k: v for k, v in c.items()
                                      if k != 'samples'}
                                     for c in res["configs"]]))
    print("COH DONE", flush=True)


if __name__ == "__main__":
    main()
