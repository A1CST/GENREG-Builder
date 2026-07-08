"""Train all Phase-2 structural decompositions in one batch job (user
directive: break up every structural genome for full observability). Order
decomposes into Order-bigram (K=1) + Order-context (K=4, already shipped,
not retrained here); Alternation into Altern-rhythm + Altern-func-chain;
Agreement into Agree-modal + Agree-number; Semantic into Sem-adjacent +
Sem-window. No-repeat/Opener/Closer/Boundary/Commas were assessed and left
as-is (already single-question, no real compound to split). Runs entirely
on the I2 primary.
"""
import os
import pickle
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "genreg_train" not in sys.modules:
    _pkg = types.ModuleType("genreg_train")
    _pkg.__path__ = [os.path.join(ROOT, "genreg_train")]
    sys.modules["genreg_train"] = _pkg

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "decompose_structural.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import evolang
evolang.CORPUS_PATH = os.path.join(ROOT, "corpora", "wikipedia", "wiki_corpus.txt")
evolang._IDS = None
if not os.path.exists(evolang.CORPUS_PATH):
    log("FATAL: wiki corpus missing"); sys.exit(1)

import time
from genreg_train import wordpipe as wp
from genreg_train import altern_decompose as altd
from genreg_train import agree_decompose as agrd
from genreg_train import sem_decompose as semd

NCL, C, D = 32, 4, 24
t0 = time.time()
log("building corpus + classes...")
wp.build_word_corpus(4000); wp.induce_word_classes(NCL)
log(f"done in {time.time()-t0:.0f}s")

out = {}

log("\n=== ORDER-BIGRAM (K=1) ==="); r = wp.run_class_lm(NCL, gens=1000, pop=200, C=1, E=10, H=64, seed=7, log=log)
out["order_bigram"] = {"champ": r["champ"], "val_ppl": r.get("val_ppl")}

log("\n=== ALTERN-RHYTHM ==="); r = altd.train_altern_rhythm(log=log)
out["altern_rhythm"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

log("\n=== ALTERN-FUNC-CHAIN ==="); r = altd.train_altern_funcchain(log=log)
out["altern_funcchain"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

log("\n=== AGREE-MODAL ==="); r = agrd.train_agree_modal(log=log)
out["agree_modal"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

log("\n=== AGREE-NUMBER ==="); r = agrd.train_agree_number(log=log)
out["agree_number"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

log("\n=== SEM-ADJACENT ==="); r = semd.train_sem_adjacent(log=log)
out["sem_adjacent"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

log("\n=== SEM-WINDOW ==="); r = semd.train_sem_window(log=log)
out["sem_window"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

OUT_DIR = os.path.join(ROOT, "corpora", "wikipedia", "build")
OUT = os.path.join(OUT_DIR, "structural_decompose.pkl")
with open(OUT, "wb") as f:
    pickle.dump(out, f)
log(f"\nsaved {OUT}")
log(f"TOTAL TIME: {time.time()-t0:.0f}s")
log("DONE")
