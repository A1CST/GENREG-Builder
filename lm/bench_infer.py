"""bench_infer.py - inference benchmark for the reddit-facing numbers:
model load time (from pack), tokens/s plain and polished, and verbatim
polished samples from the full three-specialist decode.

  python lm/bench_infer.py
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_ROOT, "radial_data")

PROMPTS = [
    "the stars in the night sky",
    "he was born in the small town",
    "the judge in the court room",
    "the chemical reaction in the water",
    "she played the piano and sang",
    "one of the most important discoveries",
]


def main():
    import lm_word_infer as li
    t0 = time.time()
    li._build()
    load_s = round(time.time() - t0, 1)
    print(f"LOAD: {load_s}s (pack -> ready)", flush=True)

    # warm-up (first completion pays steer/grammar asset loads)
    li.complete(PROMPTS[0], n_words=4, seed=0)

    n_words = 24
    t0 = time.time()
    for i, p in enumerate(PROMPTS[:3]):
        li.complete(p, n_words=n_words, seed=10 + i, best_of=1)
    plain_s = (time.time() - t0) / 3
    print(f"PLAIN: {plain_s:.2f}s per {n_words} words = "
          f"{n_words / plain_s:.1f} tok/s", flush=True)

    results = []
    t0 = time.time()
    for i, p in enumerate(PROMPTS):
        t1 = time.time()
        r = li.complete(p, n_words=n_words, seed=20 + i, best_of=8)
        dt = time.time() - t1
        results.append({"prompt": p, "completion": r["completion"],
                        "topic": r.get("topic"), "seconds": round(dt, 1)})
        print(f"POLISH {dt:5.1f}s | {r.get('topic') or '-':<10} | "
              f"{r['completion']}", flush=True)
    pol_s = (time.time() - t0) / len(PROMPTS)
    out = {"load_s": load_s, "plain_s_per_24w": round(plain_s, 2),
           "plain_tok_s": round(n_words / plain_s, 1),
           "polish_s_per_24w": round(pol_s, 1),
           "polish_tok_s": round(n_words / pol_s, 2),
           "samples": results}
    with open(os.path.join(RD, "bench_infer_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("BENCH: " + json.dumps({k: v for k, v in out.items()
                                  if k != "samples"}), flush=True)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
