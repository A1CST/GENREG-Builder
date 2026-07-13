"""Full retrain on the COMBINED corpus (Wikipedia + Cornell Movie Dialogs,
24.4% dialogue by weight) -- user directive: real questions/exclamations/
turn-taking that Wikipedia alone can't provide, retraining everything
downstream rather than bolting a new corpus onto old champions. Trains, in
one batch:
  - the 11 core genomes (Order/Selection/Bidirectional/Boundary/Comma/
    Agreement/Alternation/Semantic/No-repeat/Opener/Closer)
  - the 7 structural-decomposition sub-genomes (Order-bigram, Altern-rhythm/
    func-chain, Agree-modal/number, Sem-adjacent/window)
  - the 3 intent genomes (sent_type, sent_type_exclaim, intent_punct
    punctuation-sequence)
  - fresh Order-backward + Selection-backward (via the reversed-cache trick)
Runs entirely on the I2 primary.
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
    _pkg = types.ModuleType("genreg_train")
    _pkg.__path__ = [os.path.join(ROOT, "genreg_train")]
    sys.modules["genreg_train"] = _pkg

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "retrain_combined.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import evolang
evolang.CORPUS_PATH = os.path.join(ROOT, "corpora", "combined", "combined_corpus.txt")
evolang._IDS = None
if not os.path.exists(evolang.CORPUS_PATH):
    log(f"FATAL: {evolang.CORPUS_PATH} not found on this node — aborting."); sys.exit(1)
log(f"corpus source: {evolang.CORPUS_PATH} ({os.path.getsize(evolang.CORPUS_PATH)/1e6:.0f} MB, confirmed present)")

from genreg_train import wordpipe as wp
from genreg_train import agreement as ag
from genreg_train import altern as al
from genreg_train import sem_compat as sc
from genreg_train import repetition as rp
from genreg_train import sent_open as so
from genreg_train import sent_close as sclose
from genreg_train import altern_decompose as altd
from genreg_train import agree_decompose as agrd
from genreg_train import sem_decompose as semd
from genreg_train import sent_type as st
from genreg_train import sent_type_exclaim as ste
from genreg_train import intent_punct as ip

NCL, C, D = 32, 4, 24
GENS = {"order": 1000, "sel": 900, "bisel": 900, "bound": 600, "comma": 600}
OUT_DIR = os.path.join(ROOT, "corpora", "combined")

t0 = time.time()
log("building word corpus + inducing classes on the COMBINED corpus (one-time, heavy)...")
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

with open(os.path.join(OUT_DIR, "combined_genomes.pkl"), "wb") as f:
    pickle.dump(champs, f)
log(f"\nsaved combined_genomes.pkl ({time.time()-t0:.0f}s elapsed)")

log("\nbuilding chunk index...")
try:
    chunks = wp.build_chunk_index(NCL)
    with open(os.path.join(OUT_DIR, "combined_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    log("saved combined_chunks.pkl")
except Exception as exc:
    log(f"chunk index failed (non-fatal): {exc}")

# ---- structural decomposition ----
struct = {}
log("\n=== ORDER-BIGRAM (K=1) ==="); r = wp.run_class_lm(NCL, gens=1000, pop=200, C=1, E=10, H=64, seed=7, log=log)
struct["order_bigram"] = {"champ": r["champ"], "val_ppl": r.get("val_ppl")}
log("\n=== ALTERN-RHYTHM ==="); r = altd.train_altern_rhythm(log=log)
struct["altern_rhythm"] = {"champ": r["champ"], "val_acc": r["val_acc"]}
log("\n=== ALTERN-FUNC-CHAIN ==="); r = altd.train_altern_funcchain(log=log)
struct["altern_funcchain"] = {"champ": r["champ"], "val_acc": r["val_acc"]}
log("\n=== AGREE-MODAL ==="); r = agrd.train_agree_modal(log=log)
struct["agree_modal"] = {"champ": r["champ"], "val_acc": r["val_acc"]}
log("\n=== AGREE-NUMBER ==="); r = agrd.train_agree_number(log=log)
struct["agree_number"] = {"champ": r["champ"], "val_acc": r["val_acc"]}
log("\n=== SEM-ADJACENT ==="); r = semd.train_sem_adjacent(log=log)
struct["sem_adjacent"] = {"champ": r["champ"], "val_acc": r["val_acc"]}
log("\n=== SEM-WINDOW ==="); r = semd.train_sem_window(log=log)
struct["sem_window"] = {"champ": r["champ"], "val_acc": r["val_acc"]}

with open(os.path.join(OUT_DIR, "combined_structural_decompose.pkl"), "wb") as f:
    pickle.dump(struct, f)
log(f"\nsaved combined_structural_decompose.pkl ({time.time()-t0:.0f}s elapsed)")

# ---- intent genomes ----
log("\n=== SENT_TYPE (question vs statement) ==="); r = st.train_sent_type(log=log)
intent = {"sent_type": {"champ": r["champ"], "val_acc": r["val_acc"], "question_rate": r["question_rate"]}}
log("\n=== SENT_TYPE_EXCLAIM ==="); r = ste.train_sent_type_exclaim(log=log)
intent["sent_type_exclaim"] = {"champ": r["champ"], "val_acc": r["val_acc"], "exclaim_rate": r["exclaim_rate"]}
log("\n=== INTENT_PUNCT (punctuation-sequence) ==="); r = ip.train_intent_punct(log=log)
intent["intent_punct"] = {"champ": r["champ"], "val_ppl": r["val_ppl"], "unigram_ppl": r["unigram_ppl"],
                          "C": r["C"], "E": r["E"], "H": r["H"]}

with open(os.path.join(OUT_DIR, "combined_intent.pkl"), "wb") as f:
    pickle.dump(intent, f)
log(f"\nsaved combined_intent.pkl ({time.time()-t0:.0f}s elapsed)")

# ---- backward genomes ----
log("\n=== BACKWARD (Order-backward + Selection-backward) ===")
w2c, cids_fwd, nc, _ = wp.induce_word_classes(NCL)
ids_rev = ids[::-1].copy()
cids_rev = w2c[ids_rev]
wp._WORDCACHE[4000] = (ids_rev, vocab, stoi)
key = (NCL, 40, 4000)
wp._CLASSCACHE[key] = (w2c, cids_rev, nc, vocab)
log("reversed sequence caches installed")

r_ord = wp.run_class_lm(NCL, gens=1000, pop=200, C=C, E=10, H=64, seed=7, log=log)
r_bisel = wp.run_biselection(NCL, gens=1500, pop=200, D=D, K=7, seed=7, log=log)

wp._WORDCACHE[4000] = (ids, vocab, stoi)
wp._CLASSCACHE[key] = (w2c, cids_fwd, nc, vocab)

with open(os.path.join(OUT_DIR, "combined_backward.pkl"), "wb") as f:
    pickle.dump({"order_bwd": r_ord["champ"], "bisel_bwd": r_bisel["champ"]}, f)
log(f"saved combined_backward.pkl")

log(f"\nTOTAL TIME: {time.time()-t0:.0f}s")
log("DONE")
