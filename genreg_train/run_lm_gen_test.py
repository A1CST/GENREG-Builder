"""Generation test for the round-3 LM (runs ON the primary — no local
tests, per user rule). Loads BOTH artifacts (lm_intent.pkl for opener/
length intent machinery, lm_sem.pkl for sem_next + grammar_real word
choice) through the real Service.generate() path and logs the output for
a spread of seed words and rng seeds. Writes corpora/combined/
lm_gen_test.log (artifact-fetch whitelisted).
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "genreg_train" not in sys.modules:
    _pkg = types.ModuleType("genreg_train")
    _pkg.__path__ = [os.path.join(ROOT, "genreg_train")]
    sys.modules["genreg_train"] = _pkg

LOG = os.path.join(ROOT, "corpora", "combined", "lm_gen_test.log")
open(LOG, "w").close()


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


from genreg_train import lm_service  # noqa: E402

svc = lm_service.Service()
svc.ensure()
log(f"service ready={svc.ready} err={svc.err} sem_loaded={svc.sem is not None}")
if not svc.ready or svc.sem is None:
    log("ABORT: artifacts not in place")
    sys.exit(1)

sem_meta = svc.sem["splits"]["sem_next"]
gram_meta = svc.sem["splits"]["grammar_real"]
log(f"sem_next: holdout-cand-acc={sem_meta['holdout_acc']:.4f} "
    f"(baseline {sem_meta['majority_baseline']:.4f})  "
    f"vocab-top1={sem_meta['vocab_top1']:.4f} "
    f"(baseline {sem_meta['vocab_top1_baseline']:.4f})")
log(f"grammar_real: holdout-balanced-acc={gram_meta['holdout_balanced_acc']:.4f}")
log("")

SEEDS = ["the", "what", "please", "she", "how", "why", "there", "my"]
for w in SEEDS:
    for s in (0, 1, 2):
        r = svc.generate(w, seed=s)
        if r.get("err"):
            log(f"{w!r} (rng {s}) -> ERR: {r['err']}")
            continue
        fills = [t for t in r["trace"] if t.get("action") == "fill" and "sem_z" in t]
        mean_gram = (sum(t["gram_z"] for t in fills) / len(fills)) if fills else 0.0
        log(f"{w!r:>9} (rng {s}) -> {r['text']!r}")
        log(f"          intent={r['mark_intent']}  words={len(r['words'])}  "
            f"mean-gram-z-of-picks={mean_gram:+.2f}")
log("")
log("DONE")
