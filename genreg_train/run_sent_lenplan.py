"""Train sent_lenplan.py and run the decisive probe: do known long-sentence
openers (subordinators/connectives: although, because, since, while, when,
after...) score higher than known short-sentence openers (yes, no, then,
so, he, it...)? Training val_acc is NOT the verdict — see project convention.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "sent_lenplan.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import sent_lenplan as sl

res = sl.train_sent_lenplan(log=log)
log(f"\nval_acc={res['val_acc']}  median_len={res['median_len']}")

vocab = res["vocab"]
stoi = {w: i for i, w in enumerate(vocab)}
scores = sl.word_scores(res["champ"], vocab)

log("\nPROBE — long-sentence openers should score HIGH, short-sentence openers LOW:")
long_words = ["although", "because", "since", "while", "when", "after", "before",
             "if", "though", "unless", "despite", "however", "meanwhile", "given"]
short_words = ["yes", "no", "then", "so", "he", "it", "she", "ok", "now", "still", "here"]
long_scores = [(w, float(scores[stoi[w]])) for w in long_words if w in stoi]
short_scores = [(w, float(scores[stoi[w]])) for w in short_words if w in stoi]
missing = [w for w in long_words + short_words if w not in stoi]
if missing:
    log(f"OOV (skipped): {missing}")
for w, s in long_scores:
    log(f"   L  {w:10s} {s:+.3f}")
for w, s in short_scores:
    log(f"   S  {w:10s} {s:+.3f}")
l_mean = sum(s for _, s in long_scores) / len(long_scores)
s_mean = sum(s for _, s in short_scores) / len(short_scores)
log(f"\nmean long-opener score: {l_mean:+.3f}   mean short-opener score: {s_mean:+.3f}")
log(f"separation: {'OK — L > S' if l_mean > s_mean else 'BAD — no separation'}")

import pickle
with open(os.path.join(HERE, "sent_lenplan_champ.pkl"), "wb") as f:
    pickle.dump({"champ": res["champ"], "val_acc": res["val_acc"],
                "median_len": res["median_len"]}, f)
log("saved sent_lenplan_champ.pkl")
log("DONE")
