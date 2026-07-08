"""Train collocation.py and run the decisive probe: does the correct
preposition score higher than a mismatched one for known verb-preposition
collocations (depend ON, look AT, consist OF...)?
"""
import os
import pickle
import sys
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
LOG = os.path.join(HERE, "collocation.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import collocation as co

res = co.train_collocation(log=log)
log(f"\nval_acc={res['val_acc']}")

vocab = res["vocab"]; champ = res["champ"]
PROBES = [
    ("depend", "on", "at"), ("look", "at", "on"), ("consist", "of", "at"),
    ("believe", "in", "of"), ("wait", "for", "at"), ("agree", "with", "of"),
    ("deal", "with", "at"), ("belong", "to", "of"), ("care", "for", "at"),
    ("rely", "on", "of"),
]
log("\nPROBE — correct preposition should score HIGH, mismatched LOW:")
ok = 0
for w, good, bad in PROBES:
    sg = co.score_pair(champ, vocab, w, good)
    sb = co.score_pair(champ, vocab, w, bad)
    if sg is None or sb is None:
        log(f"   {w} {good}/{bad}: OOV"); continue
    passed = sg > sb
    ok += passed
    log(f"   {w:10s} {good:5s}={sg:+.2f}  {bad:5s}={sb:+.2f}  {'OK' if passed else 'BAD'}")
log(f"\n{ok}/{len(PROBES)} correct")

with open(os.path.join(HERE, "collocation_champ.pkl"), "wb") as f:
    pickle.dump({"champ": champ, "val_acc": res["val_acc"]}, f)
log("saved collocation_champ.pkl")
log("DONE")
