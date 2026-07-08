"""Train clause_count.py and run the decisive probe. Unlike sent_type/
sent_lenplan we have no strong lexical prior for which OPENERS predict a
later coordinating conjunction, so the probe here is data-driven: hold out
the empirical per-word compound-rate (from the mined counts) and check
whether the genome's score correlates with it (Spearman rank correlation)
instead of hand-picked word lists. Training val_acc is NOT the verdict.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "clause_count.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import clause_count as cc

res = cc.train_clause_count(log=log)
log(f"\nval_acc={res['val_acc']}  corpus compound-rate={res['compound_rate']}")

vocab = res["vocab"]
compound_init, simple_init, _, stoi = cc.sentence_initial_ids_by_clause()
scores = cc.word_scores(res["champ"], vocab)

# empirical per-word compound-rate, for words seen >=30 times as an opener
from collections import Counter
c_counts = Counter(compound_init.tolist())
s_counts = Counter(simple_init.tolist())
word_ids = set(c_counts) | set(s_counts)
rows = []
for wid in word_ids:
    n = c_counts.get(wid, 0) + s_counts.get(wid, 0)
    if n < 30:
        continue
    rate = c_counts.get(wid, 0) / n
    rows.append((wid, n, rate, float(scores[wid])))

log(f"\n{len(rows)} opener words with >=30 occurrences")
ranks_rate = {wid: i for i, (wid, n, r, s) in enumerate(sorted(rows, key=lambda x: x[2]))}
ranks_score = {wid: i for i, (wid, n, r, s) in enumerate(sorted(rows, key=lambda x: x[3]))}
d2 = sum((ranks_rate[wid] - ranks_score[wid]) ** 2 for wid, n, r, s in rows)
n_r = len(rows)
spearman = 1 - (6 * d2) / (n_r * (n_r ** 2 - 1)) if n_r > 1 else 0.0
log(f"\nPROBE — Spearman rank correlation (genome score vs empirical compound-rate): {spearman:+.3f}")
log("verdict: " + ("OK — meaningful positive correlation" if spearman > 0.3
                    else "WEAK/BAD — little to no learnable signal"))

log("\ntop 10 by genome score:")
for wid, n, r, s in sorted(rows, key=lambda x: -x[3])[:10]:
    log(f"   {vocab[wid]:12s} n={n:5d} empirical_rate={r:.3f} score={s:+.3f}")
log("bottom 10 by genome score:")
for wid, n, r, s in sorted(rows, key=lambda x: x[3])[:10]:
    log(f"   {vocab[wid]:12s} n={n:5d} empirical_rate={r:.3f} score={s:+.3f}")

import pickle
with open(os.path.join(HERE, "clause_count_champ.pkl"), "wb") as f:
    pickle.dump({"champ": res["champ"], "val_acc": res["val_acc"],
                "compound_rate": res["compound_rate"], "spearman": spearman}, f)
log("saved clause_count_champ.pkl")
log("DONE")
