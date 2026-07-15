"""Train round 3 of the LM rebuild on the combined corpus: the "sem" and
"grammar" genome groups, both inside a feature space BUILT from the corpus
itself (PPMI + eigendecomposition — see lm_sem.py's module docstring and
GENREG_RULES SS II templates there).

  sem_next     — intent-conditioned next-word ranking via an evolved query
                 into the fixed feature space. Success bar: beat the
                 majority-frequency candidate baseline round 2 FAILED
                 (21.16% vs 26.92%).
  grammar_real — real-vs-shuffled word-order discriminator. Success bar:
                 meaningfully above 50% balanced.

No lookup tables in the model: candidates at inference come from the genome
scoring the whole vocabulary through the feature space; the follower-pool
mechanism from round 2 is NOT written into this artifact. Hard negatives at
mining time are training-data construction only.

Meant to run on the I2 primary:

    python run_job.py --node http://10.0.0.15:8800 genreg_train/run_lm_sem.py --watch
    python run_job.py --node http://10.0.0.15:8800 genreg_train/run_lm_sem.py smoke --watch

`smoke` = 6MB corpus slice, 60 gens, 60K examples — a fast end-to-end
correctness pass on the primary (per user rule: no local test runs), not a
result. Writes corpora/combined/lm_sem.pkl + lm_sem.log (smoke mode writes
lm_sem_smoke.pkl + lm_sem_smoke.log instead so a smoke can never clobber a
real artifact).
"""
import os
import pickle
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "genreg_train" not in sys.modules:
    _pkg = types.ModuleType("genreg_train")
    _pkg.__path__ = [os.path.join(ROOT, "genreg_train")]
    sys.modules["genreg_train"] = _pkg

SMOKE = "smoke" in sys.argv[1:]
TAG = "_smoke" if SMOKE else ""
LOG = os.path.join(ROOT, "corpora", "combined", f"lm_sem{TAG}.log")
OUT = os.path.join(ROOT, "corpora", "combined", f"lm_sem{TAG}.pkl")
open(LOG, "w").close()


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


from genreg_train import lm_intent as li          # noqa: E402
from genreg_train import lm_sem as ls             # noqa: E402
import numpy as np                                # noqa: E402

MAX_CHARS = 6_000_000 if SMOKE else None
# With echo negatives the gen-0 bootstrap no longer IS the answer —
# evolution genuinely climbs (smoke 6: 0.098 -> 0.191 in 60 gens, still
# rising) — so the full sweep gets real generation budget (SS I.3 caps a
# component sweep at 2-8k; these are well inside).
SEM_GENS = 60 if SMOKE else 800
GRAM_GENS = 60 if SMOKE else 400
N_NEXT = 60_000 if SMOKE else 1_000_000
N_GRAM = 60_000 if SMOKE else 1_000_000

t0 = time.time()
log(f"=== LM round 3 {'SMOKE ' if SMOKE else ''}run ===")
log("loading corpus...")
text = li.load_text(max_chars=MAX_CHARS)
log(f"corpus chars: {len(text):,}")
log("tokenizing...")
tokens = li.tokenize(text)
del text
log(f"tokens: {len(tokens):,}")
vocab, stoi = li.build_vocab(tokens)
log(f"vocab size: {len(vocab):,}")

log("\n=== building the feature environment (PPMI + eig, from THIS corpus) ===")
words = np.asarray([stoi.get(t, 0) for t in tokens if li.MARK_ID.get(t) is None],
                   dtype=np.int32)
feats = ls.build_feature_space(words, V=len(vocab), log=log)
log("  nearest-neighbor probes (SS VII.3 — inspect the space before training):")
ls.nn_probe(feats, vocab, stoi, log=log)

log("\n=== mining sem_next examples (mixed negatives, intent-labeled) ===")
sem_left, sem_intent, sem_cand = li.mine_next_word_examples(
    tokens, stoi, ctx_k=6, n_samples=N_NEXT)
# Mixed negatives: keep 3 hard (same-preceding-word followers, positions
# 1..3), one RANDOM corpus draw (global separation — smoke 2's full-vocab
# top-1 of 0.0003 vs 0.0412 baseline was the SS XI train/inference
# mismatch), and one ECHO negative drawn from the example's OWN context
# window. The echo negative is the anti-repetition landscape pressure:
# the bootstrapped query ("score words similar to the context") makes
# echoing the context the gen-0 attractor, and the first full run's
# generation looped ("the the the...") because nothing in the fitness
# ever punished it. SS XII: make the target the only stable attractor.
_rng = np.random.default_rng(3)
sem_cand[:, 4] = words[_rng.integers(0, len(words), len(sem_cand))]
_echo = sem_left[np.arange(len(sem_cand)),
                 _rng.integers(0, sem_left.shape[1], len(sem_cand))]
_bad = (_echo == 0) | (_echo == sem_cand[:, 0])   # pad slot / legitimate repeat
_echo[_bad] = words[_rng.integers(0, len(words), int(_bad.sum()))]
sem_cand[:, 5] = _echo
log(f"examples: {len(sem_cand):,} (1 true + 3 hard + 1 random + 1 echo negative each)")

# SS VII.1 baselines on the SAME candidate sets
word_freq = np.bincount(words.astype(np.int64), minlength=len(vocab))
freq_pick = word_freq[sem_cand].argmax(axis=1)
sem_baseline = float((freq_pick == 0).mean())
top_word = int(np.argsort(-word_freq[1:])[0]) + 1
vocab_top1_baseline = float((sem_cand[:, 0] == top_word).mean())
log(f"  majority-frequency candidate baseline: {sem_baseline:.4f}")
log(f"  always-'{vocab[top_word]}' full-vocab top-1 baseline: {vocab_top1_baseline:.4f}")

# log-frequency ENVIRONMENT feature (normalized): a corpus statistic the
# genome reads through one evolved scalar weight (wf, init 0) — see
# lm_sem.py's template for why this is environment, not a lookup table.
logfreq = np.log1p(word_freq.astype(np.float64))
logfreq = ((logfreq - logfreq.mean()) / (logfreq.std() + 1e-9)).astype(np.float32)
logfreq[0] = float(logfreq.min())

log(f"\n=== training sem_next ({ls.SEM_SPLIT['desc']}) ===")
sem_export, sem_acc, sem_topk, _ = ls.train_sem_next(
    feats, sem_left, sem_intent, sem_cand, logfreq=logfreq, gens=SEM_GENS, log=log)
log(f"  FINAL sem_next: holdout-cand-acc={sem_acc:.4f} "
    f"(chance=0.1667, majority-frequency baseline={sem_baseline:.4f})")
log(f"  FINAL sem_next full-vocab: top-1={sem_topk[1]:.4f} "
    f"(always-most-frequent={vocab_top1_baseline:.4f})  top-5={sem_topk[5]:.4f}")

log("\n=== mining grammar_real examples (real vs shuffled windows) ===")
gram_X, gram_y = ls.mine_grammar_examples(tokens, stoi, li.MARK_ID,
                                          n_samples=N_GRAM)
log(f"examples: {len(gram_y):,} (50/50 real/shuffled by construction)")

log(f"\n=== training grammar_real ({ls.GRAM_SPLIT['desc']}) ===")
gram_export, gram_acc = ls.train_grammar(feats, gram_X, gram_y, gens=GRAM_GENS, log=log)
log(f"  FINAL grammar_real: holdout-balanced-acc={gram_acc:.4f} (chance=0.5)")

# SS VII.3 — inspect actual predictions: top vocab words the sem genome ranks
# for a few real prefixes, and grammar logits on a real vs shuffled window.
log("\n=== inspection: actual predictions on real prefixes ===")
probe_prefixes = [("the dog ran into the", 0), ("what did you think of", 2),
                  ("she opened the door and", 0)]
for phrase, intent in probe_prefixes:
    ids = [stoi.get(w, 0) for w in phrase.split()]
    left = np.asarray(([0] * 6 + ids)[-6:], dtype=np.int32)
    scores = ls.sem_vocab_scores_export(sem_export, feats, logfreq, left, intent)
    top = np.argsort(-scores)[:8]
    log(f"  '{phrase}' [{li.MARKS[intent]}] -> " +
        ", ".join(vocab[j] for j in top))
real_win = np.asarray([stoi.get(w, 0) for w in
                       "she opened the door and walked".split()], dtype=np.int32)
shuf_win = real_win[[3, 0, 5, 2, 4, 1]]
lr = float(ls.grammar_logit_export(gram_export, feats, real_win)[0])
lsh = float(ls.grammar_logit_export(gram_export, feats, shuf_win)[0])
log(f"  grammar logit real={lr:+.3f} vs shuffled={lsh:+.3f} "
    f"({'PASS' if lr > lsh else 'FAIL'})")

artifact = {
    "feats": feats, "logfreq": logfreq, "vocab": vocab, "stoi": stoi, "ctx_k": 6,
    "splits": {
        "sem_next": {"genome": sem_export, "group": ls.SEM_GROUP,
                     "desc": ls.SEM_SPLIT["desc"], "holdout_acc": sem_acc,
                     "chance": 1 / 6, "majority_baseline": sem_baseline,
                     "vocab_top1": sem_topk[1], "vocab_top5": sem_topk[5],
                     "vocab_top1_baseline": vocab_top1_baseline,
                     "n_examples": len(sem_cand)},
        "grammar_real": {"genome": gram_export, "group": ls.GRAM_GROUP,
                         "desc": ls.GRAM_SPLIT["desc"],
                         "holdout_balanced_acc": gram_acc, "chance": 0.5,
                         "n_examples": len(gram_y)},
    },
}
with open(OUT, "wb") as fh:
    pickle.dump(artifact, fh)
log(f"\nsaved {OUT}  ({time.time() - t0:.0f}s total)")
log("DONE")
