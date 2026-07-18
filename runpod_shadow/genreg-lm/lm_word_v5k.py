"""lm_word_v5k.py - retrain the word generator with a 5,000-word target
vocabulary (user's call: crank V to 5K).

Module 33 measured the topic-hold bottleneck: the V=2000 dialogue-heavy
target list simply lacks topical words to emit (chemistry's best available
steer word was "energy"). Top-5000 of the train region reaches much deeper
into content vocabulary while the corpus mix (75% wiki / 25% dialogue)
stays unchanged. W=16 and the whole pipeline match the deployed checkpoint;
the ONLY lever moved is V (+ the data regen it requires).

Overwrites radial_data/lm_word.npz and lm_model_word.json - run backups
first (lm_word_v2000_backup.npz / lm_model_word_v2000.json).
lm_cont_tables.pkl is raw word->count dicts, V-independent: reused as-is.

  python lm_word_v5k.py
"""
import radial_lm_word as rw

rw.W = 16
rw.V = 5000

if __name__ == "__main__":
    rw.make_word_data(n_train=150000, n_test=20000)
    # max_spaces=1: space 0 froze 5 genomes at +0.0000 val gain (attempts
    # 3+4), exactly the deployed V=2000 shape (1 space, 4 genomes) - the
    # model IS anchor+head here. Space 1 earns nothing and its scorer's
    # full-bank copies are what OOM the 96GB pod at V=5000.
    rw.run(max_spaces=1)
