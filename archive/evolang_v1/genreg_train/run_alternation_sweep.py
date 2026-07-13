"""Cheap experiment (not a retrain): does turning UP Alternation strength
actually break the "of the X of the Y" function-word chains in the
wiki-trained genomes' output? Sweeps ORDER_ALTERN_GAMMA/ALTERN_GAMMA well
past their current defaults (2.0/3.0), measures the REAL effect on
generated text (function-function adjacency rate, adj-hit rate, distinct
ratio, dangling-ending rate), same battery discipline as the rest of this
project -- number that matters is generation-time effect, not a training
metric. Runs on the I2 primary.
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
LOG = os.path.join(HERE, "alternation_sweep.log")
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
from genreg_train import altern as al
from genreg_train import agreement as ag
from genreg_train import repetition as rp

NCL, C, D = 32, 4, 24
CHAMPS_PATH = os.path.join(ROOT, "corpora", "wikipedia", "build", "wordpipe_wiki_genomes.pkl")

log("loading champions + corpus...")
with open(CHAMPS_PATH, "rb") as f:
    champs = pickle.load(f)
ids, vocab, stoi = wp.build_word_corpus(4000)
table, w2c, vocab2, nc, cids = wp.build_class_words(NCL)
feat, _ = wp.word_features(4000, D)
cents = wp.class_centroids(NCL, D)
logfreq = np.log1p(np.bincount(ids, minlength=len(vocab)).astype(np.float32))
is_content = rp.content_mask(vocab)
altern_feats = al.func_feats(vocab)
agree_feats = ag.gram_feats(vocab)
altern_classfeat = np.zeros((nc, altern_feats.shape[1]), np.float32)
agree_classfeat = np.zeros((nc, agree_feats.shape[1]), np.float32)
for cl, (mem, p) in table.items():
    altern_classfeat[cl] = p @ altern_feats[mem]
    agree_classfeat[cl] = p @ agree_feats[mem]
ORDER_AGREE_GAMMA, AGREE_GAMMA, SEM_GAMMA = 1.0, 2.5, 2.5
OPEN_GAMMA, CLOSE_GAMMA = 4.0, 0.5
open_scores = altern_feats @ champs["open"]
close_scores = altern_feats @ champs["close"]
freq = np.bincount(ids, minlength=len(vocab)).astype(np.float64)
close_center = float((close_scores * freq).sum() / freq.sum())
BIGSET = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
OBLIG_OPEN_WORDS = set()  # not tracked here; dangling measured via al.PREPS/DET/ARTICLES/TO below
CLOSED_CLASS = al.PREPS | al.ARTICLES | al.DET | al.TO


def bound_prob(cl, cur, prev_w):
    pb = wp.boundary_prob(champs["bound"], cl, cur)
    if prev_w is not None and 0 < pb < 1:
        pb = min(1.0, pb * np.exp(CLOSE_GAMMA * (close_scores[prev_w] - close_center)))
    return pb


def generate(n, seed, order_altern_gamma, altern_gamma):
    rng = np.random.default_rng(seed)
    order_reranks = [(altern_classfeat, champs["altern"], order_altern_gamma),
                     (agree_classfeat, champs["agree"], ORDER_AGREE_GAMMA)]
    cls_seq = wp.gen_class_seq(champs["order"], C, n, cids[500:500 + C], rng, 0.8,
                               reranks=order_reranks)
    reranks = [(altern_feats, champs["altern"], altern_gamma),
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
        bonus = rp.penalty(champs["rep"], recent, mem, is_content) if prev is not None else None
        if prev is not None and cur == 0:
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
    text = " ".join(parts).replace(" .", ".").replace(" ,", ",")
    return re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def measure(order_altern_gamma, altern_gamma, n_samples=25, seed0=5000, n=180):
    ff_pairs = 0; total_pairs = 0
    hits = 0; total_bg = 0
    all_words = []
    dangling = 0; total_sents = 0
    for i in range(n_samples):
        text = generate(n, seed0 + i, order_altern_gamma, altern_gamma)
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
        for s in re.split(r"(?<=[.])\s+", text):
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


SWEEP = [(2.0, 3.0), (4.0, 6.0), (6.0, 9.0), (8.0, 12.0), (12.0, 18.0)]
log("\nsweeping (ORDER_ALTERN_GAMMA, ALTERN_GAMMA) -- current default is (2.0, 3.0):")
for oag, ag_ in SWEEP:
    r = measure(oag, ag_)
    log(f"  ({oag:5.1f}, {ag_:5.1f})  func-func-adjacency={r['ff_rate']:.3f}  "
       f"adj-hit={r['adj_hit']:.3f}  distinct={r['distinct']:.3f}  dangling={r['dangling']:.3f}")

log("\nsample at strongest setting (12.0, 18.0):")
for seed in range(4):
    log(generate(120, 9000 + seed, 12.0, 18.0))
    log("")

log("DONE")
