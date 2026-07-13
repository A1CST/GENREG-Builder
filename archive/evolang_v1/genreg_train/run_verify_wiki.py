"""Verification-only job: load the wiki-trained champions (from
run_retrain_wiki.py's output) + the wiki corpus/vocab/classes, and generate
several real sample sentences with the SAME generation logic as
wordpipe_service.py's Service.generate() (Order skeleton -> Bidirectional
selection -> Boundary/Comma punctuation), so the actual text can be pulled
back and inspected before deciding whether to swap demo/genomes.pkl.
Runs entirely on the I2 primary -- no local heavy compute.
"""
import os
import pickle
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
LOG = os.path.join(HERE, "verify_wiki.log")
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

import numpy as np
from genreg_train import wordpipe as wp

NCL, C, D = 32, 4, 24
CHAMPS_PATH = os.path.join(ROOT, "corpora", "wikipedia", "build", "wordpipe_wiki_genomes.pkl")
CHUNKS_PATH = os.path.join(ROOT, "corpora", "wikipedia", "build", "wordpipe_wiki_chunks.pkl")

log("loading champions...")
with open(CHAMPS_PATH, "rb") as f:
    champs = pickle.load(f)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)
log(f"champion keys: {list(champs.keys())}")

log("building corpus + classes (same corpus, should hit cache)...")
ids, vocab, stoi = wp.build_word_corpus(4000)
table, w2c, vocab2, nc, cids = wp.build_class_words(NCL)
feat, _ = wp.word_features(4000, D)
cents = wp.class_centroids(NCL, D)
logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
log(f"vocab size: {len(vocab)}  classes: {nc}")

# Alternation/Agreement class-feature centroids (needed for order reranks)
from genreg_train import altern as al
from genreg_train import agreement as ag
altern_feats = al.func_feats(vocab)
agree_feats = ag.gram_feats(vocab)
altern_classfeat = np.zeros((nc, altern_feats.shape[1]), np.float32)
agree_classfeat = np.zeros((nc, agree_feats.shape[1]), np.float32)
for cl, (mem, p) in table.items():
    altern_classfeat[cl] = p @ altern_feats[mem]
    agree_classfeat[cl] = p @ agree_feats[mem]

ORDER_ALTERN_GAMMA, ORDER_AGREE_GAMMA = 2.0, 1.0
ALTERN_GAMMA, AGREE_GAMMA, SEM_GAMMA = 3.0, 2.5, 2.5
OPEN_GAMMA, CLOSE_GAMMA = 4.0, 0.5

open_scores = altern_feats @ champs["open"] if "open" in champs else None
close_scores = altern_feats @ champs["close"] if "close" in champs else None
close_center = 0.0
if close_scores is not None:
    freq = np.bincount(ids, minlength=len(vocab)).astype(np.float64)
    close_center = float((close_scores * freq).sum() / freq.sum())

cfreq = np.bincount(cids, minlength=nc).astype(np.float64)
cprob = cfreq / cfreq.sum()


def bound_prob(cl, cur, prev_w):
    pb = wp.boundary_prob(champs["bound"], cl, cur)
    if close_scores is not None and prev_w is not None and 0 < pb < 1:
        pb = min(1.0, pb * np.exp(CLOSE_GAMMA * (close_scores[prev_w] - close_center)))
    return pb


def generate(n, seed):
    rng = np.random.default_rng(seed)
    order_reranks = [(altern_classfeat, champs["altern"], ORDER_ALTERN_GAMMA),
                     (agree_classfeat, champs["agree"], ORDER_AGREE_GAMMA)]
    cls_seq = wp.gen_class_seq(champs["order"], C, n, cids[500:500 + C], rng, 0.8,
                               reranks=order_reranks)
    reranks = [(altern_feats, champs["altern"], ALTERN_GAMMA),
              (agree_feats, champs["agree"], AGREE_GAMMA),
              (feat, champs["sem"], SEM_GAMMA)]
    recent, parts, prev, cur, clause, j = [], [], None, 0, 0, 0
    while j < len(cls_seq):
        cl = int(cls_seq[j])
        if cl not in table:
            if cur >= 4 and rng.random() < bound_prob(cl, cur, prev):
                parts.append("."); cur = 0; clause = 0
            j += 1; continue
        mem = table[cl][0]
        bonus = None
        if prev is not None:
            from genreg_train import repetition as rp
            is_content = rp.content_mask(vocab)
            bonus = rp.penalty(champs["rep"], recent, mem, is_content)
        if open_scores is not None and prev is not None and cur == 0:
            ob = OPEN_GAMMA * open_scores[mem]
            bonus = ob if bonus is None else bonus + ob
        if prev is not None:
            nxt = next((int(cls_seq[k]) for k in range(j + 1, len(cls_seq)) if int(cls_seq[k]) in table), cl)
            w = wp._fill_bisel(prev, cl, nxt, table, feat, logfreq, cents, champs["bisel"], rng,
                               reranks=reranks, bonus=bonus)
        else:
            w = int(rng.choice(mem, p=table[cl][1]))
        parts.append(vocab[w]); prev = w; recent.append(w)
        cur += 1; clause += 1; j += 1
        pb = bound_prob(cl, cur, prev)
        if cur >= 4 and rng.random() < pb:
            parts.append("."); cur = 0; clause = 0
        elif clause >= 3 and rng.random() < wp.boundary_prob(champs["comma"], cl, clause):
            parts.append(","); clause = 0
    import re
    text = " ".join(parts).replace(" .", ".").replace(" ,", ",")
    return re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


log("\n=== SAMPLES (wiki-trained genomes, wiki corpus) ===")
for seed in range(10):
    log(f"\n--- seed {seed} ---")
    log(generate(120, seed))

log("\nDONE")
