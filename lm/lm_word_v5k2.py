"""lm_word_v5k2.py - the crank retrain: V=5000 generator + quad/skip
continuation tables in the bank (lm_crank.py probe: 40.1% of test blind,
62.4% of it answerable by the new tables -> crank justified).

Same data (lm_word.npz, untouched), same pipeline, max_spaces=1 (the
model is anchor+head; see lm_word_v5k.py). The ONLY lever is
EXTRA_TABLES: bank grows 17,304 -> 27,688 cols (quad + skipA as
[vec|prob], skipB as vec). Checkpoint gets bank="skip5k" so
lm_word_infer replays the matching bank.

  python lm_word_v5k2.py
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import radial_lm_word as rw

rw.W = 16
rw.V = 5000
rw.EXTRA_TABLES = True

if __name__ == "__main__":
    rw.run(max_spaces=1)
