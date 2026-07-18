"""intent_union.py - measure the 4th vote before it earns a default.

Prompts of known intent (questions / statements / exclaims); decode at
intent_lam in {0, 0.5, 1.0, 2.0}. Metrics, none of which is the steering
table: RESPONSE-FIT = mean held-out adjacency log-odds of the completion
words for the prompt's intent (intent_judge_counts.json, disjoint 100 to
120MB slice); fluency = the independent n-gram judge; plus the intent
classifier's read of each prompt, and samples verbatim.

  python lm/intent_union.py
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
LAMS = [0.0, 0.5, 1.0, 2.0]

from coherence_decode import build_judge_ngram, distinct2

PROMPTS = {
    "question": ["what is the name of the city",
                 "how did the war come to an end",
                 "where was the first church built",
                 "why did the people leave the town",
                 "who was the leader of the party",
                 "when did the team win the game"],
    "statement": ["the city was founded in the early period",
                  "the river runs through the northern region",
                  "the album was released by the band",
                  "the school was built near the station",
                  "the family moved to the small village",
                  "the company produced a new engine"],
    "exclaim": ["what a wonderful thing that was",
                "how amazing the whole show became",
                "that was such an incredible game",
                "this is the best day of the year"],
}


def main():
    t0 = time.time()

    def log(m):
        print(m, flush=True)

    import lm_word_infer as li
    li._build()
    ia = li._intent_assets()
    log(f"[iu] intent assets: {ia.get('ready')} "
        f"(classifier balanced {ia.get('balanced')})")
    with open(os.path.join(RD, "intent_judge_counts.json")) as f:
        j = json.load(f)
    jc, jt = j["counts"], j["totals"]
    V = len(jc)
    gtot = sum(jt)
    NAMES = j["names"]

    def response_fit(words, cls):
        vals = []
        for w in words:
            cs = jc.get(w)
            if not cs or sum(cs) < 3:
                continue
            vals.append(np.log((cs[cls] + 0.5) / (jt[cls] + 0.5 * V))
                        - np.log((sum(cs) + 0.5) / (gtot + 0.5 * V)))
        return float(np.mean(vals)) if vals else 0.0

    judge_ng = build_judge_ngram()
    res = {"lams": LAMS, "rows": []}
    for lam_i in LAMS:
        fits, jl, d2s, samples = [], [], [], []
        cls_ok = 0
        total = 0
        for intent, prompts in PROMPTS.items():
            cls = NAMES.index(intent)
            for pi, prompt in enumerate(prompts):
                r = li.complete(prompt, n_words=20, seed=30 + pi,
                                best_of=8, intent_lam=lam_i)
                words = r["completion"].split()
                fits.append(response_fit(words, cls))
                jl.append(judge_ng(words))
                d2s.append(distinct2(words))
                if lam_i > 0:
                    cls_ok += int(r.get("intent") == intent)
                total += 1
                if pi < 2:
                    samples.append({"intent": intent, "prompt": prompt,
                                    "completion": r["completion"],
                                    "detected": r.get("intent")})
        row = {"intent_lam": lam_i,
               "response_fit": round(float(np.mean(fits)), 4),
               "judge_logprob": round(float(np.mean(jl)), 4),
               "distinct2": round(float(np.mean(d2s)), 4),
               "classifier_agree": round(cls_ok / total, 3) if lam_i > 0
               else None,
               "samples": samples}
        res["rows"].append(row)
        log(f"[iu] lam_i={lam_i}: response-fit {row['response_fit']} "
            f"| fluency {row['judge_logprob']} | d2 {row['distinct2']}"
            + (f" | prompt-intent agree {row['classifier_agree']}"
               if lam_i > 0 else ""))

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "intent_union_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("IU RESULT: " + json.dumps([{k: v for k, v in r.items()
                                     if k != "samples"}
                                    for r in res["rows"]]))
    print("IU DONE", flush=True)


if __name__ == "__main__":
    main()
