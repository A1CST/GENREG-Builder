"""Decomposition experiment (no retrain needed): Selection's champion is
literally two questions summed into one score --
    s = (feat[prev] @ ML) @ cf.T + cf @ (MR @ cents[next_cls]) + beta * logfreq[mem]
                context-fit term (bilinear)                      frequency-bias term
The frequency term rewards common words REGARDLESS of context -- exactly
the mechanism that keeps pulling every ambiguous slot toward "the"/"of"/
"in", producing "of the X of the Y" chains. Since beta is already a
separate scalar (not entangled inside ML/MR), we can decompose the score
into "context-fit" + FREQ_GAMMA * "frequency-bias" and sweep FREQ_GAMMA
independently -- no new training, testing whether the decomposition itself
helps before deciding it's worth a real two-genome split. Runs on the I2
primary.
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
LOG = os.path.join(HERE, "selection_decompose.log")
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
ORDER_ALTERN_GAMMA, ORDER_AGREE_GAMMA = 2.0, 1.0
ALTERN_GAMMA, AGREE_GAMMA, SEM_GAMMA = 3.0, 2.5, 2.5
OPEN_GAMMA, CLOSE_GAMMA = 4.0, 0.5
open_scores = altern_feats @ champs["open"]
close_scores = altern_feats @ champs["close"]
freq = np.bincount(ids, minlength=len(vocab)).astype(np.float64)
close_center = float((close_scores * freq).sum() / freq.sum())
BIGSET = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
CLOSED_CLASS = al.PREPS | al.ARTICLES | al.DET | al.TO
ML, MR, BETA = champs["bisel"]


def bound_prob(cl, cur, prev_w):
    pb = wp.boundary_prob(champs["bound"], cl, cur)
    if prev_w is not None and 0 < pb < 1:
        pb = min(1.0, pb * np.exp(CLOSE_GAMMA * (close_scores[prev_w] - close_center)))
    return pb


def _apply_reranks(s, prev_w, mem, reranks):
    return wp._apply_reranks(s, prev_w, mem, reranks)


def fill_bisel_decomposed(prev_w, cl, next_cls, rng, freq_gamma, reranks=None, bonus=None, temp=0.7):
    """Same math as wp._fill_bisel, but the frequency-bias term is scaled by
    freq_gamma independently of the context-fit term (which stays at its
    trained weight)."""
    mem = table[cl][0]
    cf = feat[mem]
    context_fit = (feat[prev_w] @ ML) @ cf.T + cf @ (MR @ cents[next_cls])
    s = context_fit + freq_gamma * BETA * logfreq[mem]
    s = _apply_reranks(s, prev_w, mem, reranks)
    if bonus is not None:
        s = s + bonus
    s = s / temp; s -= s.max(); p = np.exp(s); p /= p.sum()
    return int(rng.choice(mem, p=p))


def generate(n, seed, freq_gamma):
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
        bonus = rp.penalty(champs["rep"], recent, mem, is_content) if prev is not None else None
        if prev is not None and cur == 0:
            ob = OPEN_GAMMA * open_scores[mem]
            bonus = ob if bonus is None else bonus + ob
        if prev is not None:
            nxt = next((int(cls_seq[k]) for k in range(j + 1, len(cls_seq)) if int(cls_seq[k]) in table), cl)
            w = fill_bisel_decomposed(prev, cl, nxt, rng, freq_gamma, reranks=reranks, bonus=bonus)
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


def measure(freq_gamma, n_samples=25, seed0=7000, n=180):
    ff_pairs = 0; total_pairs = 0
    hits = 0; total_bg = 0
    all_words = []
    dangling = 0; total_sents = 0
    bigram_counter = {}
    for i in range(n_samples):
        text = generate(n, seed0 + i, freq_gamma)
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
                key = (wids[k], wids[k + 1])
                bigram_counter[key] = bigram_counter.get(key, 0) + 1
        for s in re.split(r"(?<=[.])\s+", text):
            sw = re.findall(r"[a-zA-Z']+", s)
            if not sw:
                continue
            total_sents += 1
            if sw[-1].lower() in CLOSED_CLASS:
                dangling += 1
    top_bigram_count = max(bigram_counter.values()) if bigram_counter else 0
    top_bigram = max(bigram_counter, key=bigram_counter.get) if bigram_counter else None
    return {
        "ff_rate": ff_pairs / max(1, total_pairs),
        "adj_hit": hits / max(1, total_bg),
        "distinct": len(set(all_words)) / max(1, len(all_words)),
        "dangling": dangling / max(1, total_sents),
        "top_bigram_count": top_bigram_count,
        "top_bigram": top_bigram,
    }


log("\nsweeping FREQ_GAMMA (frequency-bias weight, decomposed from context-fit; 1.0 = current trained default):")
for fg in (1.0, 0.5, 0.25, 0.1, 0.0):
    r = measure(fg)
    top = f"{vocab[r['top_bigram'][0]]} {vocab[r['top_bigram'][1]]}" if r["top_bigram"] else "-"
    log(f"  freq_gamma={fg:4.2f}  func-func-adj={r['ff_rate']:.3f}  adj-hit={r['adj_hit']:.3f}  "
       f"distinct={r['distinct']:.3f}  dangling={r['dangling']:.3f}  "
       f"most-repeated='{top}'x{r['top_bigram_count']}")

log("\nsamples at freq_gamma=0.25:")
for seed in range(4):
    log(generate(120, 9700 + seed, 0.25))
    log("")

log("\nsamples at freq_gamma=0.0 (context-fit ONLY, no frequency bias at all):")
for seed in range(4):
    log(generate(120, 9800 + seed, 0.0))
    log("")

log("DONE")
