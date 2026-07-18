"""lm_crank.py - the modules-14-16 crank ported to the V=5000 word model.

Recipe (validated on the V=500 line, 0.30 -> 0.57): PROBE the failure
slice, measure HEADROOM under new environment tables, and only then
retrain with the tables as channels. This file is the probe + the table
builder; the retrain is gated on the probe's numbers.

Tables (same independent corpus slice as the existing cont tables,
30-46MB - disjoint from train/test regions and the judge slice):
  quad   (w-3, w-2, w-1) -> next     the true 4-gram, module 16's winner
  skipA  (w-3, w-1)      -> next     far-skip, module 14/15's winner
  skipB  (w-3, w-2)      -> next

  python lm_crank.py probe
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
TBL = os.path.join(RD, "lm_skip5k_tables.pkl")


def build_tables():
    if os.path.exists(TBL):
        with open(TBL, "rb") as f:
            return pickle.load(f)
    from radial_lm import _clean
    t0 = time.time()
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as fh:
        fh.seek(30_000_000)              # the cont-tables slice
        toks = _clean(fh.read(16_000_000)).split()
    quad, skipA, skipB = {}, {}, {}
    for i in range(3, len(toks)):
        w = toks[i]
        q = (toks[i - 3], toks[i - 2], toks[i - 1])
        quad.setdefault(q, {})
        quad[q][w] = quad[q].get(w, 0) + 1
        a = (toks[i - 3], toks[i - 1])
        skipA.setdefault(a, {})
        skipA[a][w] = skipA[a].get(w, 0) + 1
        b = (toks[i - 3], toks[i - 2])
        skipB.setdefault(b, {})
        skipB[b][w] = skipB[b].get(w, 0) + 1
    with open(TBL, "wb") as f:
        pickle.dump((quad, skipA, skipB), f)
    print(f"[crank] tables built: quad {len(quad):,} skipA {len(skipA):,} "
          f"skipB {len(skipB):,} keys ({round(time.time() - t0)}s)",
          flush=True)
    return quad, skipA, skipB


def top5(dist):
    if not dist:
        return []
    return sorted(dist, key=dist.get, reverse=True)[:5]


def probe():
    t0 = time.time()
    import radial_lm_word as rw
    vocab, _, _ = rw._load_embed()
    z = np.load(os.path.join(RD, "lm_word.npz"), allow_pickle=True)
    ctx_te, yte = z["ctx_te"], z["yte"]
    targets = [str(w) for w in z["targets"]]
    W = ctx_te.shape[1]
    with open(os.path.join(RD, "lm_cont_tables.pkl"), "rb") as f:
        uni_c, bi_c, tri_c = pickle.load(f)
    quad, skipA, skipB = build_tables()

    def word(i, slot):
        j = int(ctx_te[i, slot])
        return vocab[j] if j >= 0 else None

    n = len(yte)
    blind = 0
    blind_q = blind_a = blind_b = blind_any = 0
    covered = 0
    for i in range(n):
        w1, w2, w3 = word(i, W - 3), word(i, W - 2), word(i, W - 1)
        true = targets[int(yte[i])]
        if (w2, w3) in tri_c:
            cur = tri_c[(w2, w3)]
        elif w3 in bi_c:
            cur = bi_c[w3]
        else:
            cur = uni_c
        if true in top5(cur):
            covered += 1
            continue
        blind += 1
        hq = true in top5(quad.get((w1, w2, w3), {}))
        ha = true in top5(skipA.get((w1, w3), {}))
        hb = true in top5(skipB.get((w1, w2), {}))
        blind_q += hq; blind_a += ha; blind_b += hb
        blind_any += (hq or ha or hb)
    out = {
        "n_test": n, "has_target": covered, "blind": blind,
        "blind_frac": round(blind / n, 4),
        "headroom_quad": round(blind_q / blind, 4),
        "headroom_skipA": round(blind_a / blind, 4),
        "headroom_skipB": round(blind_b / blind, 4),
        "headroom_any": round(blind_any / blind, 4),
        "answerable_overall": round((covered + blind_any) / n, 4),
        "seconds": round(time.time() - t0),
    }
    with open(os.path.join(RD, "lm_crank_probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("PROBE: " + json.dumps(out), flush=True)
    print(f"VERDICT: {out['blind_frac']:.1%} of test is blind to the current "
          f"tables; new tables answer {out['headroom_any']:.1%} of the blind "
          f"slice -> {'CRANK JUSTIFIED' if out['headroom_any'] > 0.25 else 'headroom too thin'}",
          flush=True)
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    if "probe" in sys.argv or len(sys.argv) == 1:
        probe()
