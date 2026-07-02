"""AUTOMATED SCALING: how far can the invented memory faculties be pushed?
Only the WORLD is made harder (longer delays, longer sequences, more pairs). No reward changes,
no wiring in. Logs where each capability holds vs degrades. Results -> scaling_results.txt.

Honest note: a low score at large sizes may be a REACHABILITY limit (the search needs more
generations or a curriculum) rather than a hard capability ceiling. We do not paper over that.
"""
import numpy as np
import memory_invent as MI
import spatial_seq as SS
import content_address as CA

LOG = open("scaling_results.txt", "w")


def out(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def mean_over(fn, seeds=2):
    return float(np.mean([fn(s) for s in range(seeds)]))


out("=" * 64)
out("SCALING SWEEP — invented memory under progressively harder worlds")
out("=" * 64)

# ---- 1. invented single-slot memory: how far back can the cue be? ----
out("\n[1] INVENTED MEMORY (one world slot): recall vs delay")
out("    cue seen L-1 steps before the response; chance = %.2f" % (1 / MI.V))
for L in [2, 3, 4, 6, 8, 12]:
    MI.L = L
    sc = mean_over(lambda s: MI.recall(MI.evolve(gens=400, seed=s), np.random.default_rng(0)))
    bar = "#" * int(sc * 30)
    out(f"    delay L={L:2}: recall={sc:.2f}  {bar}")

# ---- 2. spatial sequence memory: how long a sequence can it organize? ----
out("\n[2] SPATIAL SEQUENCE MEMORY: fraction reproduced vs sequence length")
for n in [2, 3, 4, 5, 6, 8]:
    SS.NCUE = n; Wt = n + 2
    sc = mean_over(lambda s: SS.recall(SS.evolve(gens=450, W=Wt, seed=s), np.random.default_rng(0)))
    bar = "#" * int(sc * 30)
    out(f"    length={n} (tape {Wt}): reproduced={sc:.2f}  {bar}")

# ---- 3. content-addressed retrieval: how many key->value pairs? ----
out("\n[3] CONTENT-ADDRESSED RETRIEVAL: recall-by-key vs number of pairs K")
out("    (pairs in random arrival order; chance = %.2f)" % (1 / CA.V))
for k in [2, 3]:
    CA.K = k; CA.W = k
    sc = mean_over(lambda s: CA.score(CA.evolve(gens=500, seed=s), np.random.default_rng(0)))
    bar = "#" * int(sc * 30)
    out(f"    pairs K={k}: retrieval={sc:.2f}  {bar}")

out("\n" + "=" * 64)
out("done — see scaling_results.txt")
LOG.close()
