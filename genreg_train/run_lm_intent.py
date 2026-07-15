"""Train four genome groups of the LM rebuild on the combined corpus
(Wikipedia + Cornell Movie Dialogs, kept from the archived pipeline):

  "punctuation" (5 genomes) — given the words BEFORE a mark, recognize its
  intent. punct_end, punct_question, punct_exclaim, punct_semicolon,
  punct_colon.

  "opener" (2 genomes) — the mirror image: given ONLY the sentence's FIRST
  word, recognize (confirm) what intent the sentence is headed for.
  opener_question, opener_exclaim.

  "length" (1 genome) — length_continue: given a partial sentence, is it
  already complete, or does it need to keep growing? Drives the dynamic-
  length growth decision in hangman-style generation.

  "fill" (1 genome) — fill_word: given the words around a blank, does a
  candidate word fit? A contrastive discriminator (true word vs random
  negatives), not a softmax over the vocabulary — see lm_intent.py's
  module docstring for why.

  "next" (1 genome) — next_word: the intent-conditioned, properly
  autoregressive successor to fill_word (left context only, matching how
  generate() actually runs, plus explicit conditioning on the sentence's
  target end-mark). generate() uses THIS genome for word choice now.

See lm_intent.py's module docstring for why each group is split into small
binary genomes instead of one wide classifier. Meant to run on the I2
primary (421MB corpus, don't run this on a machine someone's using for
anything else):

    python run_job.py --node http://<primary>:8800 genreg_train/run_lm_intent.py --watch
    python run_job.py --node http://<primary>:8800 --fetch corpora/combined/lm_intent.pkl
    python run_job.py --node http://<primary>:8800 --fetch corpora/combined/lm_intent.log

Writes corpora/combined/lm_intent.pkl (all 5 genomes, one artifact) and
corpora/combined/lm_intent.log (this script's own log) — both under the I2
artifact-fetch whitelist.
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

LOG = os.path.join(ROOT, "corpora", "combined", "lm_intent.log")
OUT = os.path.join(ROOT, "corpora", "combined", "lm_intent.pkl")
open(LOG, "w").close()


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


from genreg_train import lm_intent as li

if not os.path.exists(li.CORPUS_PATH):
    log(f"FATAL: corpus missing at {li.CORPUS_PATH}")
    sys.exit(1)

log("loading corpus...")
text = li.load_text()
log(f"corpus chars: {len(text):,}")

log("tokenizing...")
tokens = li.tokenize(text)
log(f"tokens: {len(tokens):,}")

log("building vocab...")
vocab, stoi = li.build_vocab(tokens, vocab_n=4000)
log(f"vocab size: {len(vocab)}")

log("mining (context -> mark) examples for the punctuation group...")
ctx_all, mark_labels = li.mine_examples(tokens, stoi, ctx_k=6)
log(f"examples mined: {len(mark_labels):,}")
mark_counts = {li.MARKS[c]: int((mark_labels == c).sum()) for c in range(li.N_CLASSES)}
log(f"class counts: {mark_counts}")

log("\nmining (first word -> eventual end mark) examples for the opener group...")
opener_ctx_all, opener_labels_all = li.mine_opener_examples(tokens, stoi)
log(f"sentences mined: {len(opener_labels_all):,}")

results = {}


def train_split(split, sub_ctx, sub_labels, ctx_k):
    key = split["key"]
    log(f"\n=== training {key} ({split['group']}) — {split['desc']} ===")
    pos_n = int(sub_labels.sum())
    log(f"  examples in scope: {len(sub_labels):,}  "
       f"({split['positive_name']}={pos_n:,}  {split['negative_name']}={len(sub_labels)-pos_n:,})")

    best_export, bal_acc, raw_acc, cm = li.train_classifier(
        sub_ctx, sub_labels, n_out=2, gens=250, pop=120,
        D=16, H=24, ctx_k=ctx_k, n_per_class=64, seed=0, log=log)

    log(f"  FINAL {key}: balanced-acc={bal_acc:.4f} (chance=0.5)  raw-acc={raw_acc:.4f}")
    log(f"  confusion (rows=true, cols=predicted) [{split['negative_name']}, {split['positive_name']}]:")
    for r in range(2):
        log(f"    " + "  ".join(f"{cm[r][c]:6d}" for c in range(2)))
    recall = {split["negative_name"]: float(cm[0, 0]) / cm[0].sum() if cm[0].sum() else 0.0,
             split["positive_name"]: float(cm[1, 1]) / cm[1].sum() if cm[1].sum() else 0.0}
    log(f"  per-class recall: {recall}")

    results[key] = {
        "genome": best_export, "group": split["group"], "desc": split["desc"],
        "positive_name": split["positive_name"], "negative_name": split["negative_name"],
        "holdout_balanced_acc": bal_acc, "holdout_raw_acc": raw_acc,
        "confusion": cm.tolist(), "recall": recall,
        "n_examples": len(sub_labels), "n_positive": pos_n,
    }


for split in li.SPLITS:
    sub_ctx, sub_labels = li.prepare_split(ctx_all, mark_labels, split)
    train_split(split, sub_ctx, sub_labels, ctx_k=6)

for split in li.OPENER_SPLITS:
    sub_ctx, sub_labels = li.prepare_split(opener_ctx_all, opener_labels_all, split)
    train_split(split, sub_ctx, sub_labels, ctx_k=1)

log("\nmining (partial sentence -> complete-or-continue) examples for the length group...")
length_ctx, length_extra, length_labels = li.mine_length_examples(tokens, stoi, ctx_k=6, max_prefixes=4)
log(f"prefix examples: {len(length_labels):,}  (end={int(length_labels.sum()):,}  "
   f"continue={int((length_labels == 0).sum()):,})")

log(f"\n=== training length_continue (length) — {li.LENGTH_SPLIT['desc']} ===")
len_export, len_bal, len_raw, len_cm = li.train_classifier(
    length_ctx, length_labels, n_out=2, gens=250, pop=120,
    D=16, H=24, ctx_k=6, n_per_class=64, seed=0, log=log,
    extra=length_extra, extra_dim=1)
log(f"  FINAL length_continue: balanced-acc={len_bal:.4f} (chance=0.5)  raw-acc={len_raw:.4f}")
log("  confusion (rows=true, cols=predicted) [continue, end]:")
for r in range(2):
    log("    " + "  ".join(f"{len_cm[r][c]:6d}" for c in range(2)))
len_recall = {"continue": float(len_cm[0, 0]) / len_cm[0].sum() if len_cm[0].sum() else 0.0,
             "end": float(len_cm[1, 1]) / len_cm[1].sum() if len_cm[1].sum() else 0.0}
log(f"  per-class recall: {len_recall}")
results["length_continue"] = {
    "genome": len_export, "group": li.LENGTH_GROUP, "desc": li.LENGTH_SPLIT["desc"],
    "positive_name": "end", "negative_name": "continue",
    "holdout_balanced_acc": len_bal, "holdout_raw_acc": len_raw,
    "confusion": len_cm.tolist(), "recall": len_recall,
    "n_examples": len(length_labels), "n_positive": int(length_labels.sum()),
}

log("\nmining (left/right context -> true word vs corrupted negatives) for the fill group...")
fill_left, fill_right, fill_cand = li.mine_fill_examples(tokens, stoi, ctx_k=6,
                                                         n_samples=1_000_000, n_neg=5)
log(f"fill examples: {len(fill_cand):,}  (1 true + 5 negatives each)")

log(f"\n=== training fill_word (fill) — {li.FILL_SPLIT['desc']} ===")
fill_export, fill_acc = li.train_fill(fill_left, fill_right, fill_cand, gens=250, pop=120,
                                      D=24, ctx_k=6, batch_size=512, seed=0, log=log)
log(f"  FINAL fill_word: holdout-acc={fill_acc:.4f} (chance={1 / 6:.4f}, 1 true + 5 negatives)")
results["fill_word"] = {
    "genome": fill_export, "group": li.FILL_GROUP, "desc": li.FILL_SPLIT["desc"],
    "holdout_acc": fill_acc, "chance": 1 / 6,
    "n_examples": len(fill_cand),
}

log("\nmining (left context + intent -> true next word vs corrupted negatives) for the next group...")
next_left, next_intent, next_cand = li.mine_next_word_examples(tokens, stoi, ctx_k=6,
                                                                n_samples=1_000_000, n_neg=5)
log(f"next-word examples: {len(next_cand):,}  (1 true + 5 negatives each)  "
   f"intent dist: {{{', '.join(f'{li.MARKS[m]}={int((next_intent == m).sum()):,}' for m in range(li.N_INTENTS))}}}")

# majority-class baseline (GENREG_RULES §VII.1): pick the candidate with the
# highest GLOBAL corpus frequency — no context, no evolution. The genome must
# meaningfully beat THIS, not just 1/6 chance.
import numpy as _np
_word_freq = _np.bincount(_np.asarray([stoi.get(t, 0) for t in tokens
                                       if li.MARK_ID.get(t) is None], dtype=_np.int64),
                          minlength=4001)
_freq_pick = _word_freq[next_cand].argmax(axis=1)
next_baseline = float((_freq_pick == 0).mean())
log(f"  majority-frequency baseline (pick most frequent candidate): {next_baseline:.4f}")

log(f"\n=== training next_word (next) — {li.NEXT_SPLIT['desc']} ===")
next_export, next_acc = li.train_next_word(next_left, next_intent, next_cand, gens=250, pop=120,
                                           D=24, ctx_k=6, batch_size=512, seed=0, log=log)
log(f"  FINAL next_word: holdout-acc={next_acc:.4f} (chance={1 / 6:.4f}, "
   f"majority-frequency baseline={next_baseline:.4f})")
results["next_word"] = {
    "genome": next_export, "group": li.NEXT_GROUP, "desc": li.NEXT_SPLIT["desc"],
    "holdout_acc": next_acc, "chance": 1 / 6, "majority_baseline": next_baseline,
    "n_examples": len(next_cand),
}

log("\nbuilding rerank-generation follower pools (top 200 followers per word)...")
followers, global_top = li.build_generation_followers(tokens, stoi, top_n=200)
log(f"follower pools built for {len(followers):,} words (+ global fallback of {len(global_top)})")

log("\n=== summary, all genomes ===")
for key, r in results.items():
    if "holdout_balanced_acc" in r:
        log(f"  {key:>16}  ({r['group']:>11})  balanced-acc={r['holdout_balanced_acc']:.4f}  "
           f"raw-acc={r['holdout_raw_acc']:.4f}")
    else:
        log(f"  {key:>16}  ({r['group']:>11})  holdout-acc={r['holdout_acc']:.4f}")

artifact = {"splits": results, "vocab": vocab, "stoi": stoi, "ctx_k": 6,
           "opener_ctx_k": 1, "marks": li.MARKS, "mark_intent": li.MARK_INTENT,
           "followers": followers, "global_top": global_top}
with open(OUT, "wb") as fh:
    pickle.dump(artifact, fh)
log(f"\nsaved {OUT}")
log("DONE")
