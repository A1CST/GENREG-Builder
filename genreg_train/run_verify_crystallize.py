"""Verify the crystallize extension to generate_intent_first(): a forward
polish sweep over the same class skeleton the backward pass built, re-
picking each word against its LEFT neighbor too (the backward pass only
ever scored against the RIGHT neighbor). A/B: single-pass-backward vs
crystallized, same seeds, real samples + the same battery metrics used
throughout this session (adj-hit, distinct, dangling-rate, func-func-
adjacency) on the underlying word choices. Runs on the I2 primary.
"""
import os
import re
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
LOG = os.path.join(HERE, "verify_crystallize.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import evolang
evolang.CORPUS_PATH = os.path.join(ROOT, "corpora", "combined", "combined_corpus.txt")
evolang._IDS = None
if not os.path.exists(evolang.CORPUS_PATH):
    log("FATAL: combined corpus missing"); sys.exit(1)

from genreg_train import wordpipe_service as ws
from genreg_train import altern as al
from genreg_train import wordpipe as wp

# demo/genomes.pkl is NOT pushed to the primary (not in PUSH_WHITELIST) --
# point CACHE at the combined-corpus training output directly, same fix
# already used in run_verify_decomposed_sel.py. Without this, self.champs
# is silently EMPTY on the primary (CACHE file doesn't exist there) and
# every "X in self.champs" check quietly evaluates False -- no crash, but
# every rerank (altern/agree/sem) and the crystallize branch itself go
# inert without any error.
ws.CACHE = os.path.join(ROOT, "corpora", "combined", "combined_genomes.pkl")

log("loading Service...")
svc = ws.Service()
svc._load()
if not svc.ready:
    log(f"LOAD FAILED: {svc.err}"); sys.exit(1)
log("ready.")

ids, vocab, stoi = wp.build_word_corpus(4000)
BIGSET = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
is_content = svc.is_content
CLOSED_CLASS = al.PREPS | al.ARTICLES | al.DET | al.TO

en_base = {"vocab": True, "altern": True, "agree": True, "sem": True, "rep": True}
en_cryst = dict(en_base, crystallize=True)


def measure(en, n_samples=25, seed0=50000):
    ff_pairs = 0; total_pairs = 0
    hits = 0; total_bg = 0
    all_words = []
    dangling = 0; total_sents = 0
    for i in range(n_samples):
        r = svc.generate_intent_first(en, n_marks=12, seed=seed0 + i)
        text = r.get("text", "")
        words = re.findall(r"[a-zA-Z']+", text)
        wids = [stoi.get(w.lower(), 0) for w in words]
        all_words.extend(w.lower() for w in words)
        for k in range(len(wids) - 1):
            if wids[k] and wids[k + 1]:
                total_pairs += 1
                if not is_content[wids[k]] and not is_content[wids[k + 1]]:
                    ff_pairs += 1
                total_bg += 1
                if (wids[k], wids[k + 1]) in BIGSET:
                    hits += 1
        for s in re.split(r"(?<=[.?!])\s+", text):
            sw = re.findall(r"[a-zA-Z']+", s)
            if not sw:
                continue
            total_sents += 1
            if sw[-1].lower() in CLOSED_CLASS:
                dangling += 1
    return {
        "ff_rate": ff_pairs / max(1, total_pairs),
        "adj_hit": hits / max(1, total_bg),
        "distinct": len(set(all_words)) / max(1, len(all_words)),
        "dangling": dangling / max(1, total_sents),
    }


log("\n=== battery: single-pass backward vs crystallized (forward polish added) ===")
r_base = measure(en_base)
log(f"single-pass:   {r_base}")
r_cryst = measure(en_cryst)
log(f"crystallized:  {r_cryst}")
log("\ndeltas (crystallized minus single-pass):")
for k in r_base:
    log(f"  {k}: {r_cryst[k]-r_base[k]:+.4f}")

log("\n=== real samples, single-pass ===")
for seed in range(4):
    r = svc.generate_intent_first(en_base, n_marks=12, seed=60000 + seed)
    log(f"marks: {' '.join(r.get('marks', []))}")
    log(r.get("text", ""))
    log("")

log("\n=== real samples, crystallized ===")
for seed in range(4):
    r = svc.generate_intent_first(en_cryst, n_marks=12, seed=60000 + seed)
    log(f"marks: {' '.join(r.get('marks', []))}")
    log(r.get("text", ""))
    log("")

log("DONE")
