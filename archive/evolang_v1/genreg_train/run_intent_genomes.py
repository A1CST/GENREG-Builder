"""Train the punctuation-sequence genome (discourse skeleton) and the
exclaim-affinity genome (generalizes sent_type to 3-way statement/question/
exclaim), and run decisive probes on both. User directive: the punctuation
mark is the intent anchor -- these two genomes are the mineable, no-external-
labeling foundation for that architecture. Runs on the I2 primary.
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
LOG = os.path.join(HERE, "intent_genomes.log")
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

from genreg_train import intent_punct as ip
from genreg_train import sent_type_exclaim as ste

log("=== PUNCTUATION-SEQUENCE genome (discourse skeleton) ===")
r1 = ip.train_intent_punct(log=log)
log(f"\nval_ppl={r1['val_ppl']}  unigram_ppl={r1['unigram_ppl']}  "
   f"beats_unigram={r1['val_ppl'] < r1['unigram_ppl']}")

log("\nPROBE — sample a mark sequence from the trained model and eyeball its shape:")
import numpy as np
rng = np.random.default_rng(3)
seed_ctx = np.array([ip.MARK_ID["."]] * r1["C"], dtype=np.int64)
seq = ip.gen_mark_seq(r1["champ"], 40, seed_ctx, rng, C=r1["C"])
log("  " + " ".join(ip.MARKS[m] for m in seq))

with open(os.path.join(HERE, "intent_punct_champ.pkl"), "wb") as f:
    pickle.dump(r1, f)
log("saved intent_punct_champ.pkl")

log("\n\n=== EXCLAIM-AFFINITY genome ===")
r2 = ste.train_sent_type_exclaim(log=log)
log(f"\nval_acc={r2['val_acc']}  corpus exclaim-rate={r2['exclaim_rate']}")

vocab = r2["vocab"]
stoi = {w: i for i, w in enumerate(vocab)}
scores = ste.word_scores(r2["champ"], vocab)

log("\nPROBE — exclaim-openers should score HIGH, statement-openers LOW:")
e_words = ["wow", "amazing", "incredible", "never", "how", "what", "oh", "help",
          "stop", "run", "look", "watch", "beware", "alas", "hooray"]
s_words = ["the", "he", "she", "it", "in", "on", "at", "i", "we", "they", "of", "and"]
e_scores = [(w, float(scores[stoi[w]])) for w in e_words if w in stoi]
s_scores = [(w, float(scores[stoi[w]])) for w in s_words if w in stoi]
missing = [w for w in e_words + s_words if w not in stoi]
if missing:
    log(f"OOV (skipped): {missing}")
for w, s in e_scores:
    log(f"   E  {w:10s} {s:+.3f}")
for w, s in s_scores:
    log(f"   S  {w:10s} {s:+.3f}")
if e_scores and s_scores:
    e_mean = sum(s for _, s in e_scores) / len(e_scores)
    s_mean = sum(s for _, s in s_scores) / len(s_scores)
    log(f"\nmean exclaim-opener score: {e_mean:+.3f}   mean statement-opener score: {s_mean:+.3f}")
    log(f"separation: {'OK -- E > S' if e_mean > s_mean else 'BAD -- no separation'}")

with open(os.path.join(HERE, "sent_type_exclaim_champ.pkl"), "wb") as f:
    pickle.dump({"champ": r2["champ"], "val_acc": r2["val_acc"],
                "exclaim_rate": r2["exclaim_rate"]}, f)
log("saved sent_type_exclaim_champ.pkl")
log("\nDONE")
