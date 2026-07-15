"""Flask-facing wrapper around the trained genome groups (genreg_train/
lm_intent.py):

  "punctuation" (5 genomes) — given the words BEFORE a mark, recognize its
  intent.
  "opener" (2 genomes) — given ONLY the sentence's FIRST word, recognize
  (confirm) what intent the sentence is headed for.
  "length" (1 genome) — given a partial sentence, is it complete or does it
  need to keep growing?
  "fill" (1 genome) — given the words around a blank, does a candidate word
  fit (contrastive, scored against the whole vocabulary at inference)?
  Trained bidirectionally but superseded for actual generation by "next"
  below — kept trained/shown for comparison.
  "next" (1 genome) — the intent-conditioned, properly autoregressive
  successor to fill: given ONLY the words before this one plus the
  sentence's target end-mark, does a candidate word fit? generate() uses
  THIS genome for word choice.

length_continue + next_word combine into generate(): hangman-style
variable-length generation — grow/stop is genuinely confidence-driven (not
a fixed length), and word choice is scored against the full vocabulary each
step (not forced into a handful of frequent words), now properly
autoregressive (no train/inference context mismatch) and intent-aware.
Fill order is left-to-right in this first version — true out-of-order
multi-blank infill (matching length_continue's training distribution of
contiguous, fully-real prefixes) is future work, not this pass.

Loads the artifact trained on the I2 primary (corpora/combined/
lm_intent.pkl, fetched back via run_job.py --fetch).
"""
import os
import pickle

import numpy as np

from genreg_train import lm_intent as li
from genreg_train import lm_sem as ls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(ROOT, "corpora", "combined", "lm_intent.pkl")
SEM_ARTIFACT = os.path.join(ROOT, "corpora", "combined", "lm_sem.pkl")


class Service:
    def __init__(self):
        self.ready = False
        self.err = None
        self.splits = None            # {key: {..., "genome": export dict}}
        self.vocab = None
        self.stoi = None
        self.ctx_k = 6
        self.followers = None         # legacy round-2 pools (unused by round 3)
        self.global_top = None
        self.sem = None               # round-3 artifact: feats/logfreq/sem/grammar

    def ensure(self):
        if self.ready or self.err:
            return
        self._load()

    def _load(self):
        if not os.path.exists(ARTIFACT):
            self.err = f"not trained yet — missing {ARTIFACT}"
            return
        try:
            with open(ARTIFACT, "rb") as fh:
                art = pickle.load(fh)
            self.splits = art["splits"]
            self.vocab = art["vocab"]
            self.stoi = art["stoi"]
            self.ctx_k = art.get("ctx_k", 6)
            self.followers = art.get("followers")
            self.global_top = art.get("global_top")
            self.ready = (
                self.splits is not None
                and "length_continue" in self.splits
                and "next_word" in self.splits
            )
            if not self.ready:
                self.err = "artifact missing length_continue/next_word — retrain needed"
            if os.path.exists(SEM_ARTIFACT):
                with open(SEM_ARTIFACT, "rb") as fh:
                    self.sem = pickle.load(fh)
        except Exception as exc:                     # pragma: no cover
            self.err = str(exc)

    def _genome_summaries(self, keys):
        genomes = []
        for key in keys:
            r = self.splits.get(key)
            if not r:
                continue
            g = {"key": key, "group": r["group"], "desc": r["desc"],
                "n_examples": r["n_examples"]}
            if "holdout_balanced_acc" in r:
                g.update({
                    "positive_name": r["positive_name"], "negative_name": r["negative_name"],
                    "holdout_balanced_acc": r["holdout_balanced_acc"],
                    "holdout_raw_acc": r["holdout_raw_acc"],
                    "recall": r["recall"], "confusion": r["confusion"],
                })
            else:
                g.update({"holdout_acc": r["holdout_acc"], "chance": r.get("chance", 0.5)})
            genomes.append(g)
        return genomes

    def status(self):
        if not self.ready:
            return {"ready": False, "err": self.err}
        return {
            "ready": True, "err": None,
            "chance": 0.5,
            "vocab_size": len(self.vocab) if self.vocab else None,
            "groups": [
                {"group": li.GROUP, "genomes": self._genome_summaries([s["key"] for s in li.SPLITS])},
                {"group": li.OPENER_GROUP,
                 "genomes": self._genome_summaries([s["key"] for s in li.OPENER_SPLITS])},
                {"group": li.LENGTH_GROUP, "genomes": self._genome_summaries(["length_continue"])},
                {"group": li.FILL_GROUP, "genomes": self._genome_summaries(["fill_word"])},
                {"group": li.NEXT_GROUP, "genomes": self._genome_summaries(["next_word"])},
            ],
        }

    def recognize(self, text):
        """text: a fragment of words (no trailing punctuation expected).
        Returns per-mark probabilities for 'what intent comes next', composed
        from the punctuation group's 5 binary genomes (see li.recognize_mark)."""
        if not self.ready:
            return {"err": self.err or "not ready"}
        words = li.tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        ids = [self.stoi.get(w, 0) for w in words]
        window = ([0] * self.ctx_k + ids)[-self.ctx_k:]
        ctx = np.asarray([window], dtype=np.int32)
        splits_export = {k: v["genome"] for k, v in self.splits.items()}
        ranked_pairs = li.recognize_mark(splits_export, ctx)
        ranked = [{"mark": m, "intent": li.MARK_INTENT[m], "prob": p}
                  for m, p in ranked_pairs]
        return {"context_words": words[-self.ctx_k:], "ranked": ranked}

    def recognize_opener(self, word):
        """word: a single opening word (only the first token is used).
        Returns per-mark probabilities for 'what intent this sentence is
        headed for', composed from the opener group's 2 binary genomes."""
        if not self.ready:
            return {"err": self.err or "not ready"}
        toks = [w for w in li.tokenize(word.lower()) if w.isalpha()]
        if not toks:
            return {"err": "no word given"}
        first_word = toks[0]
        wid = self.stoi.get(first_word, 0)
        splits_export = {k: v["genome"] for k, v in self.splits.items()}
        ranked_pairs = li.recognize_opener(splits_export, wid)
        ranked = [{"mark": m, "intent": li.MARK_INTENT[m], "prob": p}
                  for m, p in ranked_pairs]
        return {"first_word": first_word, "ranked": ranked}

    def generate(self, seed_word, max_words=40, seed=0, temperature=0.7):
        """Hangman-style generation: intent decided once from the seed word
        (via the opener genomes), then a tick loop where EVERY step asks
        length_continue "are we done?" before asking next_word "what word
        goes next?" (intent-conditioned, properly autoregressive — see
        NEXT_GROUP in lm_intent.py) — grow/stop is confidence-driven, not a
        fixed target length.

        Word choice (round 3): NO lookup tables anywhere — the sem_next
        genome scores the ENTIRE vocabulary through the fixed feature
        space (evolved query + evolved bias + evolved logfreq weight +
        evolved bilinear transition), the top pool by THAT score becomes
        the candidates, and the grammar_real genome reranks them by how
        real the word order would read. Both proposal and rerank are
        evolved genomes' forward passes; the round-2 follower pools are
        not consulted. Returns text plus a per-tick trace."""
        if not self.ready:
            return {"err": self.err or "not ready"}
        if self.sem is None:
            return {"err": "round-3 artifact missing (corpora/combined/lm_sem.pkl) "
                           "— train genreg_train/run_lm_sem.py first"}
        toks = [w for w in li.tokenize(seed_word.lower()) if w.isalpha()]
        if not toks:
            return {"err": "no seed word given"}
        opener_word = toks[0]
        opener_id = self.stoi.get(opener_word, 0)

        opener_export = {k: self.splits[k]["genome"]
                         for k in ("opener_question", "opener_exclaim") if k in self.splits}
        target_mark = li.recognize_opener(opener_export, opener_id)[0][0]
        intent_id = li.MARK_ID[target_mark]   # always 0/1/2 (. ! ?) — see li.N_INTENTS

        length_export = self.splits["length_continue"]["genome"]
        next_export = self.splits["next_word"]["genome"]
        rng = np.random.default_rng(seed)

        words = [opener_id]
        trace = [{"action": "fill", "word": opener_word, "note": "opener (seed)"}]
        ctx_k = self.ctx_k
        hit_max = True

        for _ in range(max_words):
            window = ([0] * ctx_k + words)[-ctx_k:]
            ctx_arr = np.asarray([window], dtype=np.int32)
            extra = np.asarray([[min(len(words) / li.LENGTH_MAX_LEN_NORM, 1.0)]], dtype=np.float32)
            logits = li.forward_export(length_export, ctx_arr, extra)[0]
            if int(np.argmax(logits)) == 1:
                trace.append({"action": "end", "mark": target_mark})
                hit_max = False
                break

            # sem_next proposes: score the WHOLE vocabulary, keep the top
            # pool (a genome decision end to end, no tables)
            sem = self.sem
            sem_export = sem["splits"]["sem_next"]["genome"]
            sem_stoi, sem_vocab = sem["stoi"], sem["vocab"]
            left_sem = np.asarray(([0] * ctx_k +
                                   [sem_stoi.get(self.vocab[w], 0) for w in words])[-ctx_k:],
                                  dtype=np.int32)
            scores = ls.sem_vocab_scores_export(sem_export, sem["feats"],
                                                sem["logfreq"], left_sem, intent_id)
            pool = np.argsort(-scores)[:60]
            sem_s = scores[pool]

            # grammar_real reranks: how real does the order read with each
            # candidate appended?
            gram_export = sem["splits"]["grammar_real"]["genome"]
            m = gram_export["m"]
            tail = ([0] * (m - 1) +
                    [sem_stoi.get(self.vocab[w], 0) for w in words])[-(m - 1):]
            wins = np.concatenate(
                [np.tile(np.asarray(tail, np.int32), (len(pool), 1)),
                 pool.astype(np.int32)[:, None]], axis=1)
            gram_s = ls.grammar_logit_export(gram_export, sem["feats"], wins)

            def z(a):
                return (a - a.mean()) / (a.std() + 1e-9)

            combined = z(sem_s) + z(gram_s)
            probs = np.exp((combined - combined.max()) / max(temperature, 1e-3))
            probs /= probs.sum()
            pick = int(rng.choice(len(pool), p=probs))
            next_word_str = sem_vocab[int(pool[pick])]
            next_id = self.stoi.get(next_word_str, 0)
            words.append(next_id)
            trace.append({"action": "fill", "word": next_word_str,
                          "prob": float(probs[pick]), "pool": int(len(pool)),
                          "sem_z": float(z(sem_s)[pick]),
                          "gram_z": float(z(gram_s)[pick])})

        if hit_max:
            trace.append({"action": "end", "mark": target_mark, "note": "hit max length"})

        text_words = [self.vocab[w] for w in words]
        text = " ".join(text_words) + target_mark
        if text:
            text = text[0].upper() + text[1:]
        return {"text": text, "words": text_words, "mark": target_mark,
               "mark_intent": li.MARK_INTENT[target_mark], "trace": trace}


SERVICE = Service()
