"""lm_word_wiki.py - THE PROSE CORPUS CRANK (module 40): massively
increase the corpus (user's call) and switch the register to prose.

The composition architecture is proven (modules 38-39); the foundation is
the constraint. One lever, scaled ~15x, register switched:
  corpus   combined (75% wiki + dialogue x6, 8MB region) -> PURE WIKI
           PROSE (wiki_corpus.txt, 316MB)
  windows  train region 8MB -> 120MB, 150k windows (200k OOMed the head fit)
  tables   cont + quad/skip from 16MB -> 150MB slice (top-20 pruned per
           key - the tables ARE most of the model; density is the lever)
  judge    a disjoint wiki slice (295MB+)
W=16 and V=5000 UNCHANGED - the corpus is the isolated variable.
Regions: windows 10-130MB, test ~131-135MB, tables 140-290MB - disjoint.

  python lm/lm_word_wiki.py
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401
import os

import radial_lm
import lm_crank
import radial_lm_word as rw

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI = os.path.join(_ROOT, "corpora", "wikipedia", "wiki_corpus.txt")

radial_lm.CORPUS = WIKI
rw.W = 16
rw.V = 5000
rw.EXTRA_TABLES = True
rw.TRAIN_MB = 120.0
rw.TEST_MB = 4.0
rw.TBL_PKL = "lm_cont_tables_wiki.pkl"
rw.TBL_SEEK = 140_000_000
rw.TBL_MB = 150
rw.TBL_TOPK = 20
lm_crank.CORPUS = WIKI
lm_crank.SKIP_PKL = "lm_skip5k_wiki.pkl"
lm_crank.TBL_SEEK = 140_000_000
lm_crank.TBL_MB = 150
lm_crank.TBL_TOPK = 20

if __name__ == "__main__":
    rw.make_word_data(n_train=150000, n_test=20000)
    rw.run(max_spaces=1)
