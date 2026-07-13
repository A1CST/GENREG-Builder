"""Train sent_type.py and run the decisive probe: do known question-openers
(do/does/will/what/is/how...) score higher than known statement-openers
(the/he/in/i...)? Training val_acc is NOT the verdict — see project convention.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "sent_type.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import sent_type as st

res = st.train_sent_type(log=log)
log(f"\nval_acc={res['val_acc']}  corpus question-rate={res['question_rate']}")

vocab = res["vocab"]
stoi = {w: i for i, w in enumerate(vocab)}
scores = st.word_scores(res["champ"], vocab)

log("\nPROBE — question-openers should score HIGH, statement-openers LOW:")
q_words = ["do", "does", "did", "will", "would", "could", "can", "is", "are", "was",
          "what", "who", "why", "how", "when", "where", "which", "shall"]
s_words = ["the", "he", "she", "it", "in", "on", "at", "i", "we", "they", "of", "and"]
q_scores = [(w, float(scores[stoi[w]])) for w in q_words if w in stoi]
s_scores = [(w, float(scores[stoi[w]])) for w in s_words if w in stoi]
for w, s in q_scores:
    log(f"   Q  {w:8s} {s:+.3f}")
for w, s in s_scores:
    log(f"   S  {w:8s} {s:+.3f}")
q_mean = sum(s for _, s in q_scores) / len(q_scores)
s_mean = sum(s for _, s in s_scores) / len(s_scores)
log(f"\nmean Q-opener score: {q_mean:+.3f}   mean S-opener score: {s_mean:+.3f}")
log(f"separation: {'OK — Q > S' if q_mean > s_mean else 'BAD — no separation'}")

import pickle
with open(os.path.join(HERE, "sent_type_champ.pkl"), "wb") as f:
    pickle.dump({"champ": res["champ"], "val_acc": res["val_acc"],
                "question_rate": res["question_rate"]}, f)
log("saved sent_type_champ.pkl")
log("DONE")
