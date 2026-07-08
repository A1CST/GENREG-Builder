"""Retrain the core WordPipe pipeline (order/sel/bisel/bound/comma/agree/
altern/sem/rep/open/close + chunk index) on the WIKIPEDIA corpus instead of
the archaic Gutenberg-novel dump (project/EEC-main/engine/corpus.txt --
"thou"/"shalt"/19th-century-novel vocabulary). User directive: a modern
corpus. corpora/wikipedia/wiki_corpus.txt is already on disk (316MB, built
for the relation genomes) -- this repoints the SHARED corpus loader
(genreg_train/evolang.py) at it and retrains everything downstream.

Compute-heavy: corpus tokenization + word-class induction over 316MB, then
11 separate GA training runs. Must run on the I2 primary, not locally.
"""
import os
import pickle
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "genreg_train" not in sys.modules:
    # bypass genreg_train/__init__.py's eager import of the unrelated RL-engine
    # subsystem (trainer.py -> engine_api.py), not deployed to compute nodes
    _pkg = types.ModuleType("genreg_train")
    _pkg.__path__ = [os.path.join(ROOT, "genreg_train")]
    sys.modules["genreg_train"] = _pkg

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "retrain_wiki.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import evolang
evolang.CORPUS_PATH = os.path.join(evolang.ROOT, "corpora", "wikipedia", "wiki_corpus.txt")
evolang._IDS = None   # drop any cached char-id array so the swap actually takes effect
# HARD FAIL if the file isn't actually on this node -- evolang._build_ids()
# silently falls back to a 4-sentence placeholder on OSError, which would
# otherwise waste hours "training" on garbage without ever erroring.
if not os.path.exists(evolang.CORPUS_PATH):
    log(f"FATAL: {evolang.CORPUS_PATH} not found on this node — aborting before any training.")
    sys.exit(1)
log(f"corpus source repointed to: {evolang.CORPUS_PATH} "
   f"({os.path.getsize(evolang.CORPUS_PATH) / 1e6:.0f} MB, confirmed present)")

from genreg_train import wordpipe as wp
from genreg_train import agreement as ag
from genreg_train import altern as al
from genreg_train import sem_compat as sc
from genreg_train import repetition as rp
from genreg_train import sent_open as so
from genreg_train import sent_close as sclose

NCL, C, D = 32, 4, 24
GENS = {"order": 1000, "sel": 900, "bisel": 900, "bound": 600, "comma": 600}

t0 = time.time()
log("building word corpus + inducing classes on the WIKIPEDIA corpus (one-time, heavy)...")
ids, vocab, stoi = wp.build_word_corpus(4000)
wp.induce_word_classes(NCL)
log(f"corpus tokens: {len(ids)}  vocab size: {len(vocab)}  built in {time.time()-t0:.0f}s")

champs = {}

log("\n=== ORDER ==="); r = wp.run_class_lm(NCL, gens=GENS["order"], pop=200, C=C, E=10, H=64, seed=7, log=log)
champs["order"] = r["champ"]

log("\n=== SELECTION ==="); r = wp.run_selection(NCL, gens=GENS["sel"], pop=200, D=D, K=7, seed=7, log=log)
champs["sel"] = r["champ"]

log("\n=== BIDIRECTIONAL ==="); r = wp.run_biselection(NCL, gens=GENS["bisel"], pop=200, D=D, K=7, seed=7, log=log)
champs["bisel"] = r["champ"]

log("\n=== BOUNDARY ==="); r = wp.run_boundary(NCL, gens=GENS["bound"], pop=200, seed=7, log=log)
champs["bound"] = r["champ"]

log("\n=== COMMA ==="); r = wp.run_comma(NCL, gens=GENS["comma"], pop=200, seed=7, log=log)
champs["comma"] = r["champ"]

log("\n=== AGREEMENT ==="); r = ag.train_agreement(log=log)
champs["agree"] = r["champ"]

log("\n=== ALTERNATION ==="); r = al.train_altern(log=log)
champs["altern"] = r["champ"]

log("\n=== SEMANTIC ==="); r = sc.train_sem(log=log)
champs["sem"] = r["champ"]

log("\n=== NO-REPEAT ==="); r = rp.train_rep(log=log)
champs["rep"] = r["champ"]

log("\n=== OPENER ==="); r = so.train_open(log=log)
champs["open"] = r["champ"]

log("\n=== CLOSER ==="); r = sclose.train_close(log=log)
champs["close"] = r["champ"]

OUT_DIR = os.path.join(evolang.ROOT, "corpora", "wikipedia", "build")
OUT = os.path.join(OUT_DIR, "wordpipe_wiki_genomes.pkl")
with open(OUT, "wb") as f:
    pickle.dump(champs, f)
log(f"\nsaved {OUT}")

log("building chunk index (lookup, not trained)...")
try:
    chunks = wp.build_chunk_index(NCL)
    CHUNK_OUT = os.path.join(OUT_DIR, "wordpipe_wiki_chunks.pkl")
    with open(CHUNK_OUT, "wb") as f:
        pickle.dump(chunks, f)
    log(f"saved {CHUNK_OUT}")
except Exception as exc:
    log(f"chunk index build failed (non-fatal, chunks toggle just won't work): {exc}")

log(f"\nTOTAL TIME: {time.time()-t0:.0f}s")
log("DONE")
