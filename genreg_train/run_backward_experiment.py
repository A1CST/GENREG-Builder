"""Backward generation experiment (user's idea): instead of generating
left-to-right and hoping to land on a valid ending, pick a valid ENDING
first and grow the sentence backward toward the opening. Every word is
chosen to serve what's already to its right, instead of a random walk that
might dead-end on "of the".

Implementation: Order and Bidirectional-Selection are generic autoregressive
predictors over whatever sequence they're trained on. Train them on the
corpus read BACKWARD (same word->class mapping, so results stay compatible
with the existing class table / word features / Closer / Boundary), which
gives "Order-backward" (predicts the PRECEDING class from the last C classes
in reversed order) and "Selection-backward" (scores a candidate against the
word to its RIGHT, already placed, and the class to its LEFT, about to be
placed) for free -- zero new algorithm, just reversed training data.

Generation: seed with a Closer-endorsed ending word, grow the class sequence
backward via Order-backward, fill words backward via Selection-backward,
reverse the assembled list to get correct reading order. Compares verb-
presence rate and completion quality against forward generation on the same
corpus/champions. Runs on the I2 primary.
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
LOG = os.path.join(HERE, "backward_experiment.log")
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

import pickle
import numpy as np
from genreg_train import wordpipe as wp
from genreg_train import agreement as ag
from genreg_train import altern as al
from genreg_train import repetition as rp

NCL, C, D = 32, 4, 24

log("building FORWARD corpus + classes (baseline, and source of the word->class map)...")
ids_fwd, vocab, stoi = wp.build_word_corpus(4000)
w2c, cids_fwd, nc, _ = wp.induce_word_classes(NCL)
table, _, _, _, _ = wp.build_class_words(NCL)
feat, _ = wp.word_features(4000, D)
cents = wp.class_centroids(NCL, D)
logfreq = np.log1p(np.bincount(ids_fwd, minlength=len(vocab)).astype(np.float32))
is_content = rp.content_mask(vocab)

with open(os.path.join(ROOT, "corpora", "wikipedia", "build", "wordpipe_wiki_genomes.pkl"), "rb") as f:
    champs = pickle.load(f)
log(f"loaded forward champions: {list(champs.keys())}")

# ---- reverse the SEQUENCE caches only, keep the SAME word->class mapping ----
ids_rev = ids_fwd[::-1].copy()
cids_rev = w2c[ids_rev]
wp._WORDCACHE[4000] = (ids_rev, vocab, stoi)
key = (NCL, 40, 4000)
wp._CLASSCACHE[key] = (w2c, cids_rev, nc, vocab)
log(f"reversed sequence caches installed (ids/cids), word->class mapping unchanged")

log("\n=== ORDER-BACKWARD (K=4, trained on reversed sequence) ===")
r_ord = wp.run_class_lm(NCL, gens=1000, pop=200, C=C, E=10, H=64, seed=7, log=log)
champ_order_bwd = r_ord["champ"]

log("\n=== SELECTION-BACKWARD (Bidirectional, trained on reversed sequence) ===")
r_bisel = wp.run_biselection(NCL, gens=1500, pop=200, D=D, K=7, seed=7, log=log)
champ_bisel_bwd = r_bisel["champ"]

# restore forward caches for the baseline comparison
wp._WORDCACHE[4000] = (ids_fwd, vocab, stoi)
wp._CLASSCACHE[key] = (w2c, cids_fwd, nc, vocab)

OUT = os.path.join(ROOT, "corpora", "wikipedia", "build", "backward_genomes.pkl")
with open(OUT, "wb") as f:
    pickle.dump({"order_bwd": champ_order_bwd, "bisel_bwd": champ_bisel_bwd}, f)
log(f"saved {OUT}")

# ---- generation: forward (existing champs) vs backward (new champs) ----
altern_feats = al.func_feats(vocab)
agree_feats = ag.gram_feats(vocab)
altern_classfeat = np.zeros((nc, al.NF), np.float32)
agree_classfeat = np.zeros((nc, ag.NG), np.float32)
for cl, (mem, p) in table.items():
    altern_classfeat[cl] = p @ altern_feats[mem]
    agree_classfeat[cl] = p @ agree_feats[mem]
close_scores = altern_feats @ champs["close"]
freq = np.bincount(ids_fwd, minlength=len(vocab)).astype(np.float64)
close_center = float((close_scores * freq).sum() / freq.sum())
open_scores = altern_feats @ champs["open"]

VERBLIKE = ag.MODALS | ag.COPULA | ag.AUX | ag.FIN_3SG | ag.FIN_NON3 | ag.BARE | ag.PARTICIPLE
def has_verb(words):
    for w in words:
        wl = w.lower()
        if wl in VERBLIKE or wl.endswith("ed") or wl.endswith("ing"):
            return True
    return False


def gen_forward(n, seed):
    rng = np.random.default_rng(seed)
    order_reranks = [(altern_classfeat, champs["altern"], 2.0), (agree_classfeat, champs["agree"], 1.0)]
    cls_seq = wp.gen_class_seq(champs["order"], C, n, cids_fwd[500:500 + C], rng, 0.8, reranks=order_reranks)
    reranks = [(altern_feats, champs["altern"], 3.0), (agree_feats, champs["agree"], 2.5), (feat, champs["sem"], 2.5)]
    recent, parts, prev, cur = [], [], None, 0
    for j, cl in enumerate(cls_seq):
        cl = int(cl)
        if cl not in table:
            continue
        mem = table[cl][0]
        bonus = rp.penalty(champs["rep"], recent, mem, is_content) if prev is not None else None
        if prev is not None and cur == 0:
            ob = 4.0 * open_scores[mem]
            bonus = ob if bonus is None else bonus + ob
        if prev is not None:
            nxt = next((int(cls_seq[k]) for k in range(j + 1, len(cls_seq)) if int(cls_seq[k]) in table), cl)
            w = wp._fill_bisel(prev, cl, nxt, table, feat, logfreq, cents, champs["bisel"], rng, reranks=reranks, bonus=bonus)
        else:
            w = int(rng.choice(mem, p=table[cl][1]))
        parts.append(vocab[w]); prev = w; recent.append(w); cur += 1
    return parts


def gen_backward(n, seed):
    """Seed with a Closer-endorsed ending word, grow backward via
    Order-backward + Selection-backward, reverse to reading order."""
    rng = np.random.default_rng(seed)
    # pick a plausible ENDING word: highest closer-score content words, sampled softly
    order = np.argsort(-close_scores)
    top_end = order[:200]
    end_probs = np.exp((close_scores[top_end] - close_scores[top_end].max()) / 0.7)
    end_probs /= end_probs.sum()
    end_word = int(rng.choice(top_end, p=end_probs))
    end_cls = int(w2c[end_word])

    seed_ctx = np.array([end_cls] * C, dtype=np.int64)
    cls_seq_bwd = wp.gen_class_seq(champ_order_bwd, C, n - 1, seed_ctx, rng, 0.8)  # grows "backward" in index
    full_cls_bwd = [end_cls] + list(cls_seq_bwd)   # index 0 = sentence END, growing toward the opening

    words_bwd = [end_word]   # words_bwd[0] = last word of the sentence
    prev = end_word          # "prev" in backward-training semantics = the word already placed to the RIGHT
    for i in range(1, len(full_cls_bwd)):
        cl = full_cls_bwd[i]
        if cl not in table:
            continue
        mem = table[cl][0]
        # next_cls here = the class BEFORE this position (not yet placed) -- use the
        # next entry in full_cls_bwd as a lookahead target, same shape _fill_bisel expects
        next_cls = full_cls_bwd[i + 1] if i + 1 < len(full_cls_bwd) else cl
        w = wp._fill_bisel(prev, cl, next_cls, table, feat, logfreq, cents, champ_bisel_bwd, rng)
        words_bwd.append(w)
        prev = w
    words_fwd_order = [vocab[w] for w in reversed(words_bwd)]
    return words_fwd_order


log("\n=== FORWARD generation (existing champions) ===")
fwd_texts = []
n_sents_f = has_verb_f = 0
for seed in range(20):
    words = gen_forward(16, seed)
    text = " ".join(words)
    fwd_texts.append(text)
    n_sents_f += 1
    if has_verb(words):
        has_verb_f += 1
for t in fwd_texts[:5]:
    log("  " + t)
log(f"forward: {has_verb_f}/{n_sents_f} contain a verb-like word ({100*has_verb_f/n_sents_f:.0f}%)")

log("\n=== BACKWARD generation (new champions, grown from a Closer-endorsed ending) ===")
bwd_texts = []
n_sents_b = has_verb_b = 0
for seed in range(20):
    words = gen_backward(16, seed + 1000)
    text = " ".join(words)
    bwd_texts.append(text)
    n_sents_b += 1
    if has_verb(words):
        has_verb_b += 1
for t in bwd_texts[:5]:
    log("  " + t)
log(f"backward: {has_verb_b}/{n_sents_b} contain a verb-like word ({100*has_verb_b/n_sents_b:.0f}%)")

log("\nDONE")
