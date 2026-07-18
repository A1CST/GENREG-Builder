"""grammar_union.py - UNION the specialists (user's architecture):
local continuation (tables+model) x topic (persistence) x GRAMMAR
(temporal-on-syntactic, kid_grammar_model.json, test 0.6642).

The grammar model scores a completion by its real-vs-shuffled logit
margin averaged over T=10 sliding windows. Union = the best-of-8 hybrid
rerank gains a third term: z(cont) + z(topic) + w_g * z(grammar).
One generation pass at the deployed merit decode, re-ranked per w_g.

  python lm/grammar_union.py
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
WGS = [0.0, 0.5, 1.0, 2.0]

from topic_steer import PROMPTS, build_judge, load_topic_model
from coherence_decode import build_judge_ngram, cont_table_scorer, distinct2
import radial_temporal as rt


def load_grammar_scorer(torch, dev):
    with open(os.path.join(RD, "kid_grammar_model.json")) as f:
        gm = json.load(f)
    zp = np.load(os.path.join(RD, "embed_rs_prev.npz"), allow_pickle=True)
    zn = np.load(os.path.join(RD, "embed_rs_next.npz"), allow_pickle=True)
    vocab = {str(w): i for i, w in enumerate(zp["vocab"])}
    E = np.concatenate([zp["feat"], zn["feat"]], 1).astype(np.float32)
    T, D = gm["T"], gm["D"]
    cmu = np.array(gm["cmu"], np.float32)
    csd = np.array(gm["csd"], np.float32)
    fmu = torch.tensor(gm["fmu"], device=dev)
    fsd = torch.tensor(gm["fsd"], device=dev)
    hm = torch.tensor(gm["head_mu"], device=dev)
    hs = torch.tensor(gm["head_sd"], device=dev)
    Wm = torch.tensor(gm["head_W"], device=dev)

    def score(words):
        """Mean real-logit margin over sliding T-word windows (stride 5);
        windows with <8/10 in-vocab words are skipped."""
        wins = []
        for a in range(0, max(1, len(words) - T + 1), 5):
            seq = words[a:a + T]
            if len(seq) < T:
                break
            ids = [vocab.get(w) for w in seq]
            if sum(i is not None for i in ids) < 8:
                continue
            X = np.zeros((T, E.shape[1]), np.float32)
            for t, i in enumerate(ids):
                if i is not None:
                    X[t] = E[i]
            wins.append(np.clip((X - cmu) / csd, -8, 8))
        if not wins:
            return 0.0
        F = torch.tensor(np.stack(wins), device=dev)
        cols = [rt._finite(torch, rt.temporal_feat(torch, F, g))
                for g in gm["genomes"]]
        Ft = ((torch.stack(cols, 1) - fmu) / fsd).clamp(-8, 8)
        s = torch.hstack([(Ft - hm) / hs,
                          torch.ones(len(wins), 1, device=dev)]) @ Wm
        return float((s[:, 1] - s[:, 0]).mean())    # real minus shuffled
    return score, gm


def main():
    t0 = time.time()

    def log(m):
        print(m, flush=True)

    import torch
    import lm_word_infer as lwi
    lwi._build()
    sa = lwi._steer_assets()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    judge_ng = build_judge_ngram()
    topic_judge = build_judge(sa["topics"])
    cont_score = cont_table_scorer()
    gram_score, gm = load_grammar_scorer(torch, dev)
    log(f"[union] grammar model loaded: test {gm['test_acc']} "
        f"({len(gm['genomes'])} genomes)")

    # 8 candidates per prompt at the DEPLOYED merit decode
    cache = {}
    for topic, prompts in PROMPTS.items():
        for prompt in prompts:
            cands = []
            for s in range(8):
                r = lwi.complete(prompt, n_words=24, seed=1000 + s,
                                 best_of=1)
                cands.append((r["completion"].split(),
                              sa["topics"].index(r["topic"])
                              if r.get("topic") in sa["topics"] else None))
            cache[(topic, prompt)] = cands
    log(f"[union] generated 8x{len(cache)} at deployed merit decode "
        f"({round(time.time() - t0)}s)")

    res = {"wgs": WGS, "rows": []}
    for wg in WGS:
        jl, d2s, held, total, samples = [], [], 0, 0, []
        for (topic, prompt), cands in cache.items():
            outs = [c[0] for c in cands]
            t_star = next((c[1] for c in cands if c[1] is not None), None)
            cs = np.array([cont_score(o) for o in outs])
            gs = np.array([gram_score(o) for o in outs])
            zc = (cs - cs.mean()) / (cs.std() + 1e-9)
            zg = (gs - gs.mean()) / (gs.std() + 1e-9)
            if t_star is not None:
                ts_ = np.array([sa["tscore"](o, t_star) for o in outs])
                zt = (ts_ - ts_.mean()) / (ts_.std() + 1e-9)
            else:
                zt = np.zeros(len(outs))
            pick = int(np.argmax(zc + zt + wg * zg))
            out = outs[pick]
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
        log(f"[union] wg={wg}: judge {row['judge_logprob']} "
            f"d2 {row['distinct2']} hold {row['hold']}")

    res["grammar_model"] = {k: gm[k] for k in
                            ("test_acc", "val_acc", "anchor")}
    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "grammar_union_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("UNION RESULT: " + json.dumps([{k: v for k, v in r.items()
                                        if k != "samples"}
                                       for r in res["rows"]]))
    print("UNION DONE", flush=True)


if __name__ == "__main__":
    main()
