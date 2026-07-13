"""Local half of the Phase-2 structural decomposition (lighter jobs, run
here in parallel with the primary's Altern/Agree batch): Order-bigram
(K=1, vs the shipped K=4 Order-context) and Semantic split by distance
(Sem-adjacent = offset 1, Sem-window = offsets 2-4). See
run_decompose_structural.py for the full context/rationale.
"""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "decompose_structural_local.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import wordpipe as wp
from genreg_train import sem_decompose as semd

NCL = 32
t0 = time.time()
out = {}

log("=== ORDER-BIGRAM (K=1) ===")
r = wp.run_class_lm(NCL, gens=1000, pop=200, C=1, E=10, H=64, seed=7, log=log)
out["order_bigram"] = {"champ": r["champ"], "val_ppl": r.get("val_ppl")}

log("\n=== SEM-ADJACENT ===")
r = semd.train_sem_adjacent(log=log)
out["sem_adjacent"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

log("\n=== SEM-WINDOW ===")
r = semd.train_sem_window(log=log)
out["sem_window"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

OUT = os.path.join(HERE, "structural_decompose_local.pkl")
with open(OUT, "wb") as f:
    pickle.dump(out, f)
log(f"\nsaved {OUT}")
log(f"TOTAL TIME: {time.time()-t0:.0f}s")
log("DONE")
