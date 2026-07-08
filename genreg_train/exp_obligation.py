"""Fluency experiment C: sweep OBLIG_GAMMA (the open-obligation suppression
strength — see wordpipe_service.py's _bound_prob/_track_oblig) and measure
its REAL effect on generation: does it reduce the rate of sentences ending
on a dangling preposition/determiner/article, without blowing up sentence
length or collapsing diversity? Same battery discipline as every other
genome in this project — the number that matters is the generation-time
effect, not a training-loss number (there IS no training loss here; this is
a single evolved-in-spirit scalar swept like CLOSE_GAMMA was).
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import wordpipe_service as ws  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "exp_obligation.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

log("loading pipeline…")
ws.SERVICE.ensure()
while ws.SERVICE.loading:
    time.sleep(1)
if ws.SERVICE.err:
    log("LOAD ERROR:", ws.SERVICE.err); sys.exit(1)
log("ready.")

EN_BASE = dict(vocab=True, order=True, sel="bi", altern=True, agree=True, sem=True, rep=True,
              open=True, close=True, bound=True, commas=True, chunks=False,
              hyper=False, mero=False, synant=False, oblig=False)

OBLIG_OPEN_WORDS = set(w for i, w in enumerate(ws.SERVICE.vocab) if ws.SERVICE.oblig_open[i])


def measure(en, n_samples=25, seed0=100):
    """Generate n_samples texts, return (dangling_rate, mean_sent_len, distinct_ratio)."""
    dangling = 0
    total_sents = 0
    lens = []
    all_words = []
    for i in range(n_samples):
        text = ws.SERVICE.generate(en, n=200, seed=seed0 + i)
        sents = [s.strip() for s in re.split(r"(?<=[.])\s+", text) if s.strip()]
        for s in sents:
            words = re.findall(r"[a-zA-Z']+", s)
            if not words:
                continue
            total_sents += 1
            lens.append(len(words))
            all_words.extend(w.lower() for w in words)
            if words[-1].lower() in OBLIG_OPEN_WORDS:
                dangling += 1
    dangling_rate = dangling / max(1, total_sents)
    mean_len = sum(lens) / max(1, len(lens))
    distinct_ratio = len(set(all_words)) / max(1, len(all_words))
    return dangling_rate, mean_len, distinct_ratio, total_sents


log("\n=== baseline: oblig OFF ===")
base = measure(EN_BASE)
log(f"dangling-rate={base[0]:.3f}  mean-sent-len={base[1]:.1f}  distinct-ratio={base[2]:.3f}  "
   f"(n_sents={base[3]})")

log("\n=== sweep OBLIG_GAMMA with oblig ON ===")
results = {}
en_on = dict(EN_BASE, oblig=True)
for gamma in (0.5, 1.0, 2.0, 4.0, 8.0):
    ws.OBLIG_GAMMA = gamma
    r = measure(en_on)
    results[gamma] = r
    log(f"gamma={gamma:>4}  dangling-rate={r[0]:.3f}  mean-sent-len={r[1]:.1f}  "
       f"distinct-ratio={r[2]:.3f}  (n_sents={r[3]})")

log("\n=== summary ===")
log(f"baseline (oblig OFF):  dangling-rate={base[0]:.3f}  len={base[1]:.1f}  distinct={base[2]:.3f}")
for gamma, r in results.items():
    delta = base[0] - r[0]
    len_drift = abs(r[1] - base[1]) / base[1]
    distinct_drift = base[2] - r[2]
    verdict = "CANDIDATE" if delta > 0.05 and len_drift < 0.25 and distinct_drift < 0.05 else "reject"
    log(f"gamma={gamma:>4}  dangling-rate-drop={delta:+.3f}  len-drift={len_drift:+.1%}  "
       f"distinct-drop={distinct_drift:+.3f}  -> {verdict}")
log("DONE")
