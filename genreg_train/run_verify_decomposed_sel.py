"""Verify the decomposed-Selection wiring in wordpipe_service.py: import the
real Service class (not a standalone reimplementation), load it against the
wiki corpus + wiki-trained champions, and generate samples with the default
gammas (all three at 1.0, mathematically equivalent to the original
wp._fill_bisel ratio). Confirms no crash and output is consistent with the
earlier run_verify_wiki.py samples. Runs on the I2 primary.
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

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "verify_decomposed_sel.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import evolang
evolang.CORPUS_PATH = os.path.join(ROOT, "corpora", "wikipedia", "wiki_corpus.txt")
evolang._IDS = None
if not os.path.exists(evolang.CORPUS_PATH):
    log("FATAL: wiki corpus missing"); sys.exit(1)

# wordpipe_service.CACHE points at demo/genomes.pkl, already swapped to the
# wiki-trained champions locally -- but this runs on the primary, which needs
# its OWN copy at that path. Point CACHE at the primary's fetched artifact.
from genreg_train import wordpipe_service as ws
ws.CACHE = os.path.join(ROOT, "corpora", "wikipedia", "build", "wordpipe_wiki_genomes.pkl")

log("loading Service...")
svc = ws.Service()
svc._load()
if not svc.ready:
    log(f"LOAD FAILED: {svc.err}"); sys.exit(1)
log("ready.")

en = {"vocab": True, "order": True, "sel": "bi", "altern": True, "agree": True,
      "sem": True, "rep": True, "open": True, "close": True, "bound": True,
      "commas": True, "chunks": False}

log("\n=== default gammas (1.0, 1.0, 1.0) -- should match run_verify_wiki.py's samples ===")
for seed in range(4):
    log(svc.generate(en, n=120, seed=seed))
    log("")

log("\n=== SEL_FREQ_GAMMA=0.25 via live module override (sanity: decomposition actually wired) ===")
ws.SEL_FREQ_GAMMA = 0.25
for seed in range(4):
    log(svc.generate(en, n=120, seed=seed))
    log("")

log("DONE")
