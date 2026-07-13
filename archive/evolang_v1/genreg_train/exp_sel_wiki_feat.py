"""Fluency experiment B: does Selection improve if it's scored over the
richer Wikipedia-trained 128d distributional space (51M words, 30K vocab)
instead of the small in-pipeline 24d SVD space (built from the ~48MB novel
corpus, 4000 words)? Bigger, cleaner distributional statistics might just
produce better local word choice without any new constraint machinery.

Reuses wp.WordSelPop / wp._sel_pool / wp.ga_step directly (same training
loop as wp.run_selection) but swaps the feature matrix for the Wikipedia
crosswalk built the same way genreg_train/rel_wire.py does it for the
relation genomes — so this is an apples-to-apples swap of ONLY the feature
space, nothing else about the training loop changes.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import wordpipe as wp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(HERE, "exp_sel_wiki_feat.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

NCL, K, gens, pop, minibatch, seed = 32, 7, 1500, 200, 512, 1234

log("=== baseline: existing 24d in-pipeline features (word_features) ===")
base_feat, base_vocab = wp.word_features(4000, 24)
log(f"baseline feat shape {base_feat.shape}")

log("\n=== building Wikipedia-crosswalked 128d features for this same vocab ===")
d = np.load(os.path.join(ROOT, "corpora", "wikipedia", "wiki_feats.npz"), allow_pickle=True)
wiki_vocab = list(d["vocab"]); wiki_feat = d["feat"]
wiki_stoi = {w: i for i, w in enumerate(wiki_vocab)}
hits = np.array([wiki_stoi.get(w, -1) for w in base_vocab])
covered = hits >= 0
coverage = float(covered.mean())
wiki_cross = np.zeros((len(base_vocab), wiki_feat.shape[1]), np.float32)
wiki_cross[covered] = wiki_feat[hits[covered]]
log(f"crosswalk coverage: {coverage*100:.1f}% of {len(base_vocab)} pipeline words found in Wikipedia vocab")


def train_with_feat(feat, D, name):
    """Body identical to wp.run_selection, feat matrix injected instead of
    built internally, so this is a true apples-to-apples swap."""
    logfreq = np.log1p(np.bincount(wp.build_word_corpus(4000)[0],
                                   minlength=len(base_vocab)).astype(np.float32))
    table, w2c, _, nc, _ = wp.build_class_words(NCL)
    ids, _, _ = wp.build_word_corpus(4000)
    n_train = int(len(ids) * 0.9)
    rng = np.random.default_rng(seed)
    log(f"[{name}] building position pools…")
    tp_prev, tp_cand = wp._sel_pool(ids[:n_train], table, w2c, feat, logfreq, 60000, K, rng)
    vp_prev, vp_cand = wp._sel_pool(ids[n_train:], table, w2c, feat, logfreq, 8000, K,
                                    np.random.default_rng(seed + 5))
    vcf, vclf = feat[vp_cand], logfreq[vp_cand]
    base_s = vclf
    bm = base_s.max(-1, keepdims=True)
    base_lp = float((base_s - bm - np.log(np.exp(base_s - bm).sum(-1, keepdims=True)))[:, 0].mean())
    base_acc = float((vclf.argmax(1) == 0).mean())

    popn = wp.WordSelPop(pop, D, seed)
    best_val, best_champ = -1e9, None
    for gen in range(1, gens + 1):
        sel = rng.integers(0, len(tp_prev), size=minibatch)
        pf, cf, clf = feat[tp_prev[sel]], feat[tp_cand[sel]], logfreq[tp_cand[sel]]
        fit = popn.fitness(pf, cf, clf)
        pdict = {"M": popn.M, "beta": popn.beta, "sigma": popn.sigma}
        wp.ga_step(pdict, fit, rng)
        popn.M, popn.beta, popn.sigma = pdict["M"], pdict["beta"], pdict["sigma"]
        if gen % 150 == 0 or gen == 1:
            vf = float(popn.fitness(feat[vp_prev], vcf, vclf)[0])
            if vf > best_val:
                best_val = vf; best_champ = popn.champion(0)
            s = popn.scores(feat[vp_prev], vcf, vclf)[0]
            acc = float((s.argmax(1) == 0).mean())
            log(f"  [{name}] gen {gen}: val_logprob={vf:.4f} (base {base_lp:.4f}) "
                f"top1={acc:.3f} (freq-only base {base_acc:.3f})")
    return {"val_logprob": round(best_val, 4), "base_logprob": round(base_lp, 4),
           "beats_freq_baseline": best_val > base_lp, "champ": best_champ, "D": D}


log("\n=== training Selection over 24d (baseline reproduction) ===")
r24 = train_with_feat(base_feat, 24, "sel-24d")
log(f"24d  val_logprob={r24['val_logprob']}  base={r24['base_logprob']}")

log("\n=== training Selection over 128d Wikipedia crosswalk ===")
r128 = train_with_feat(wiki_cross, 128, "sel-128d-wiki")
log(f"128d val_logprob={r128['val_logprob']}  base={r128['base_logprob']}")

log(f"\nlogprob improvement (higher=better): 24d={r24['val_logprob']:.4f} -> "
   f"128d={r128['val_logprob']:.4f} "
   f"({'better' if r128['val_logprob'] > r24['val_logprob'] else 'WORSE or flat'})")

out = os.path.join(HERE, "exp_sel_wiki_feat.pkl")
with open(out, "wb") as f:
    pickle.dump({"r24": r24, "r128": r128, "wiki_cross_feat": wiki_cross,
                "coverage": coverage}, f)
log(f"saved {out}")
log("DONE")
