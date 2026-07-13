"""Train parallelism.py and run the decisive probe: for known same-type
coordinated pairs (dog/cat, ran/jumped, red/blue...), does the real pairing
score higher than a mismatched, different-type content word?
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
LOG = os.path.join(HERE, "parallelism.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import parallelism as pl

res = pl.train_parallelism(log=log)
log(f"\nval_acc={res['val_acc']}")

vocab = res["vocab"]; champ = res["champ"]
PROBES = [
    ("dog", "cat", "table"), ("man", "woman", "chair"), ("red", "blue", "slow"),
    ("ran", "jumped", "table"), ("king", "queen", "book"), ("mother", "father", "stone"),
    ("happy", "sad", "wooden"), ("sword", "shield", "cloud"), ("hand", "arm", "river"),
    ("son", "daughter", "window"),
]
log("\nPROBE — same-type pair should score HIGH, mismatched-type pair LOW:")
ok = 0; n_valid = 0
for w, good, bad in PROBES:
    sg = pl.score_pair(champ, vocab, w, good)
    sb = pl.score_pair(champ, vocab, w, bad)
    if sg is None or sb is None:
        log(f"   {w} {good}/{bad}: OOV"); continue
    n_valid += 1
    passed = sg > sb
    ok += passed
    log(f"   {w:10s} {good:8s}={sg:+.2f}  {bad:8s}={sb:+.2f}  {'OK' if passed else 'BAD'}")
log(f"\n{ok}/{n_valid} correct")

with open(os.path.join(HERE, "parallelism_champ.pkl"), "wb") as f:
    pickle.dump({"champ": champ, "val_acc": res["val_acc"]}, f)
log("saved parallelism_champ.pkl")
log("DONE")
