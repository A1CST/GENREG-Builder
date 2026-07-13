"""Full guardrail battery for the round-1 experimental toggles (sent_type,
lenplan, pronominal, and the Revision stage) — the SAME battery discipline
the original 13 shipped genomes went through (adj-hit rate, distinct-word
ratio, dangling-ending rate, mean sentence length), not the lightweight
20-30-sample spot-checks used when each genome was first wired. This is
what actually decides shipped vs stays-experimental, not the probe alone.
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import wordpipe_service as ws  # noqa: E402
from genreg_train import wordpipe as wp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "battery_round1.log")
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
              hyper=False, mero=False, synant=False, oblig=False,
              sent_type=False, lenplan=False, pronominal=False)

ids, vocab, stoi = wp.build_word_corpus(4000)
BIGSET = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
OBLIG_OPEN_WORDS = set(w for i, w in enumerate(ws.SERVICE.vocab) if ws.SERVICE.oblig_open[i])


def measure(en, n_samples=60, seed0=1000, n=220):
    """Generate n_samples texts, return the full guardrail battery."""
    dangling = 0
    hits = 0; pairs = 0
    total_sents = 0
    lens = []
    all_words = []
    for i in range(n_samples):
        text = ws.SERVICE.generate(en, n=n, seed=seed0 + i)
        sents = [s.strip() for s in re.split(r"(?<=[.?])\s+", text) if s.strip()]
        for s in sents:
            words = re.findall(r"[a-zA-Z']+", s)
            if not words:
                continue
            total_sents += 1
            lens.append(len(words))
            all_words.extend(w.lower() for w in words)
            wids = [stoi.get(w.lower(), 0) for w in words]
            for k in range(len(wids) - 1):
                if wids[k] and wids[k + 1]:
                    pairs += 1
                    if (wids[k], wids[k + 1]) in BIGSET:
                        hits += 1
            if words[-1].lower() in OBLIG_OPEN_WORDS:
                dangling += 1
    return {
        "dangling_rate": dangling / max(1, total_sents),
        "mean_len": sum(lens) / max(1, len(lens)),
        "distinct_ratio": len(set(all_words)) / max(1, len(all_words)),
        "adj_hit": hits / max(1, pairs),
        "n_sents": total_sents,
    }


def report(name, base, on):
    log(f"\n=== {name} ===")
    log(f"  OFF: adj-hit={base['adj_hit']:.3f}  distinct={base['distinct_ratio']:.3f}  "
       f"dangling={base['dangling_rate']:.3f}  mean-len={base['mean_len']:.1f}  (n={base['n_sents']})")
    log(f"  ON:  adj-hit={on['adj_hit']:.3f}  distinct={on['distinct_ratio']:.3f}  "
       f"dangling={on['dangling_rate']:.3f}  mean-len={on['mean_len']:.1f}  (n={on['n_sents']})")
    adj_drop = base["adj_hit"] - on["adj_hit"]
    distinct_drop = base["distinct_ratio"] - on["distinct_ratio"]
    dangling_rise = on["dangling_rate"] - base["dangling_rate"]
    len_drift = abs(on["mean_len"] - base["mean_len"]) / base["mean_len"]
    verdict = ("SHIP-CANDIDATE" if adj_drop < 0.02 and distinct_drop < 0.02
               and dangling_rise < 0.02 and len_drift < 0.15
               else "STAYS EXPERIMENTAL — regresses a guardrail")
    log(f"  deltas: adj-hit {adj_drop:+.3f}  distinct {distinct_drop:+.3f}  "
       f"dangling {dangling_rise:+.3f}  len-drift {len_drift:+.1%}")
    log(f"  VERDICT: {verdict}")
    return verdict


log("\n--- baseline (all experimental toggles OFF) ---")
base = measure(EN_BASE)
log(f"adj-hit={base['adj_hit']:.3f}  distinct={base['distinct_ratio']:.3f}  "
   f"dangling={base['dangling_rate']:.3f}  mean-len={base['mean_len']:.1f}  (n={base['n_sents']})")

on_sent_type = measure(dict(EN_BASE, sent_type=True))
report("Sentence type (sent_type)", base, on_sent_type)

on_lenplan = measure(dict(EN_BASE, lenplan=True))
report("Sentence length plan (lenplan)", base, on_lenplan)

on_pronom = measure(dict(EN_BASE, pronominal=True))
report("Pronominalization (pronominal)", base, on_pronom)

log("\n--- Revision stage (Best-of-N vs plain generate(), same toggles) ---")
def measure_revision(en, n_samples=15, seed0=2000, n_sentences=8, n_candidates=6):
    dangling = 0; hits = 0; pairs = 0; total_sents = 0; lens = []; all_words = []
    for i in range(n_samples):
        text = ws.SERVICE.generate_revision(en, n_sentences=n_sentences,
                                            n_candidates=n_candidates, seed=seed0 + i)
        sents = [s.strip() for s in re.split(r"(?<=[.?])\s+", text) if s.strip()]
        for s in sents:
            words = re.findall(r"[a-zA-Z']+", s)
            if not words:
                continue
            total_sents += 1
            lens.append(len(words))
            all_words.extend(w.lower() for w in words)
            wids = [stoi.get(w.lower(), 0) for w in words]
            for k in range(len(wids) - 1):
                if wids[k] and wids[k + 1]:
                    pairs += 1
                    if (wids[k], wids[k + 1]) in BIGSET:
                        hits += 1
            if words[-1].lower() in OBLIG_OPEN_WORDS:
                dangling += 1
    return {
        "dangling_rate": dangling / max(1, total_sents),
        "mean_len": sum(lens) / max(1, len(lens)),
        "distinct_ratio": len(set(all_words)) / max(1, len(all_words)),
        "adj_hit": hits / max(1, pairs),
        "n_sents": total_sents,
    }

base_rev = measure(EN_BASE, n_samples=15, seed0=2000, n=160)
on_rev = measure_revision(EN_BASE)
report("Revision stage (best-of-N vs plain generate)", base_rev, on_rev)

log("\nDONE")
