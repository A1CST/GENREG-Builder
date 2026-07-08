"""Fluency experiment A: does widening the Order genome's lookback context
(C=4 -> C=8) reduce word-salad symptoms (dangling prepositions, unclosed
phrases)? The current Order genome only sees the last 4 classes — it has no
way to know "this sentence opened a PP three words ago and never closed it."
A wider window is the cheapest possible test of whether more context alone
helps, before building anything structurally new (see exp_obligation.py for
the stateful alternative).

Trains a fresh Order genome at C=8 (can't reuse the C=4 champion — context
width is baked into its shape) and reports val_ppl vs the existing C=4
champion's baseline, so the honest comparison is apples-to-apples on the
SAME held-out setup.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import wordpipe as wp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "exp_order_context.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

NCL = 32

log("=== baseline: C=4 (matches the currently shipped Order genome) ===")
base = wp.run_class_lm(NCL, gens=1000, pop=200, C=4, E=10, H=64, seed=7, log=log)
log(f"C=4  val_ppl={base['val_ppl']}  unigram_ppl={base['unigram_ppl']}  "
   f"beats_unigram={base['beats_unigram']}")

log("\n=== experiment: C=8 (double the lookback) ===")
wide = wp.run_class_lm(NCL, gens=1000, pop=200, C=8, E=10, H=64, seed=7, log=log)
log(f"C=8  val_ppl={wide['val_ppl']}  unigram_ppl={wide['unigram_ppl']}  "
   f"beats_unigram={wide['beats_unigram']}")

log(f"\nppl improvement from widening context: {base['val_ppl']:.3f} -> {wide['val_ppl']:.3f} "
   f"({'better' if wide['val_ppl'] < base['val_ppl'] else 'WORSE or flat'})")

out = os.path.join(HERE, "exp_order_c8.pkl")
with open(out, "wb") as f:
    pickle.dump({"C4": base, "C8": wide}, f)
log(f"saved {out}")
log("DONE")
