"""Generation-time validation for exp_order_context.py's C=8 Order genome —
perplexity alone is not the verdict (project convention: gate on the real
generation effect, not the training-loss number). Swaps the live service's
Order champion + context constant, measures the same dangling-rate/length/
distinct-ratio battery as exp_obligation.py, before vs after.
"""
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import wordpipe_service as ws  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "exp_order_context_gencheck.log")
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

EN = dict(vocab=True, order=True, sel="bi", altern=True, agree=True, sem=True, rep=True,
         open=True, close=True, bound=True, commas=True, chunks=False,
         hyper=False, mero=False, synant=False, oblig=False)

OBLIG_OPEN_WORDS = set(w for i, w in enumerate(ws.SERVICE.vocab) if ws.SERVICE.oblig_open[i])


def measure(en, n_samples=25, seed0=100):
    dangling = 0; total_sents = 0; lens = []; all_words = []
    for i in range(n_samples):
        text = ws.SERVICE.generate(en, n=200, seed=seed0 + i)
        sents = [s.strip() for s in re.split(r"(?<=[.])\s+", text) if s.strip()]
        for s in sents:
            words = re.findall(r"[a-zA-Z']+", s)
            if not words:
                continue
            total_sents += 1; lens.append(len(words))
            all_words.extend(w.lower() for w in words)
            if words[-1].lower() in OBLIG_OPEN_WORDS:
                dangling += 1
    dangling_rate = dangling / max(1, total_sents)
    mean_len = sum(lens) / max(1, len(lens))
    distinct_ratio = len(set(all_words)) / max(1, len(all_words))
    return dangling_rate, mean_len, distinct_ratio, total_sents


log("=== baseline: shipped C=4 Order champion ===")
base = measure(EN)
log(f"dangling-rate={base[0]:.3f}  mean-sent-len={base[1]:.1f}  distinct-ratio={base[2]:.3f}  "
   f"(n_sents={base[3]})")
sample_before = ws.SERVICE.generate(EN, n=120, seed=7)
log("sample (C=4): " + sample_before[:400])

log("\n=== swapping in the C=8 Order champion ===")
with open(os.path.join(HERE, "exp_order_c8.pkl"), "rb") as f:
    d = pickle.load(f)
c8_champ = d["C8"]["champ"]
orig_champ = ws.SERVICE.champs["order"]
orig_C = ws.C
ws.SERVICE.champs["order"] = c8_champ
ws.C = 8
wide = measure(EN)
log(f"dangling-rate={wide[0]:.3f}  mean-sent-len={wide[1]:.1f}  distinct-ratio={wide[2]:.3f}  "
   f"(n_sents={wide[3]})")
sample_after = ws.SERVICE.generate(EN, n=120, seed=7)
log("sample (C=8): " + sample_after[:400])

log("\n=== summary ===")
delta = base[0] - wide[0]
len_drift = abs(wide[1] - base[1]) / base[1]
distinct_drift = base[2] - wide[2]
verdict = "SHIP" if delta > 0.03 and len_drift < 0.2 and distinct_drift < 0.05 else \
         ("NEUTRAL" if abs(delta) <= 0.03 and len_drift < 0.2 else "reject")
log(f"C=4 dangling={base[0]:.3f} vs C=8 dangling={wide[0]:.3f}  (drop={delta:+.3f})")
log(f"len drift={len_drift:+.1%}  distinct drift={distinct_drift:+.3f}  -> {verdict}")

ws.SERVICE.champs["order"] = orig_champ
ws.C = orig_C
log("DONE")
