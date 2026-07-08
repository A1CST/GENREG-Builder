"""Web backend for the /evolang page — the WordPipe specialist pipeline.

Lazy-loads the trained genomes (demo/genomes.pkl) + corpus caches in a background
thread on first request, then generates text with any subset of the evolved
specialist layers enabled. This is the current evolution-native language model:
a pipeline of tiny gradient-free genomes (order / selection / boundary / chunks),
composed — not one net trained on a loss. See documentation/WORDPIPE_FINDINGS.md.
"""
import os
import pickle
import re
import threading

import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import agreement as ag
from genreg_train import altern as al
from genreg_train import repetition as rp
from genreg_train import rel_wire
from genreg_train import sent_type as st
from genreg_train import sent_lenplan as sl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "demo", "genomes.pkl")
NCL, C, D = 32, 4, 24
AGREE_GAMMA = 2.5         # weight of the agreement re-rank vs the selection score
ALTERN_GAMMA = 3.0        # weight of content-function alternation at word-selection
ORDER_ALTERN_GAMMA = 2.0  # weight of content-function alternation on the ORDER skeleton
ORDER_AGREE_GAMMA = 1.0   # weight of agreement on the ORDER skeleton
SEM_GAMMA = 2.5           # weight of semantic (content co-occurrence) re-rank
OPEN_GAMMA = 4.0          # weight of the sentence-opener re-rank (fires at sentence start)
CLOSE_GAMMA = 0.5         # weight of the sentence-closer (modulates where periods land)
# Standalone relation genomes (Wikipedia-trained, crosswalked into the pipeline
# vocab — see genreg_train/rel_wire.py + genomes.txt "battery note"). Kept
# conservative: these bias toward a SPECIFIC relation (is-a / part-of / same-
# vs-opposite-meaning) between adjacent content words, not general fit, so a
# high gamma reads as a taxonomy/part-chain tic rather than natural prose.
HYPER_GAMMA = 1.0         # weight of the hypernym (is-a) re-rank
MERO_GAMMA = 1.0          # weight of the meronym (part-of) re-rank
SYNANT_GAMMA = 1.0        # weight of the unified synonym/antonym re-rank
# EXPERIMENTAL — open-obligation tracker (see genreg_train/exp_obligation.py).
# The Order genome only looks back 4 classes; it has no notion of "this
# preposition/determiner opened a phrase 6 words ago and it's still unclosed."
# This tracks a live depth counter across the WHOLE sentence (prep/det/article
# pushes, a content word pops) and suppresses the sentence-boundary
# probability while depth > 0. OBLIG_GAMMA is swept/set by exp_obligation.py.
OBLIG_GAMMA = 0.0         # 0 = off; exp_obligation.py sweeps this
# EXPERIMENTAL — sentence-type genome (genreg_train/sent_type.py). Skeleton
# stage was thin (Order/Alternation/Agreement only). Validated on probe: every
# one of 18 hand-picked question-openers (do/will/what/is...) scored above
# every one of 11 statement-openers. At each sentence start, flips a coin at
# the corpus question-rate; if "question", biases the opener toward this
# genome's scores and forces '?' instead of '.' at the close.
SENT_TYPE_GAMMA = 3.0
SENT_TYPE_CHAMP = os.path.join(ROOT, "genreg_train", "sent_type_champ.pkl")
# EXPERIMENTAL — sentence-length-plan genome (genreg_train/sent_lenplan.py).
# Probe passed but weaker than sent_type's: mean long-opener score beat mean
# short-opener score (+0.532 vs -0.125), but several openers tied on identical
# scores — the coarse function-feature space caps how sharp this signal gets.
# At each sentence start, flips a coin at the corpus long-sentence rate
# (48.77% of sentences ran longer than the 14-word median); biases the opener
# toward/away from this genome's scores and reshapes the boundary probability
# to lean toward finishing before/after the median length.
LENPLAN_GAMMA = 2.0
LENPLAN_LEN_GAMMA = 0.06
LENPLAN_LONG_RATE = 207266 / (207266 + 217634)   # from sent_lenplan.log mining pass
SENT_LENPLAN_CHAMP = os.path.join(ROOT, "genreg_train", "sent_lenplan_champ.pkl")
# EXPERIMENTAL — Passage-stage pronominalization. No training, no new champion
# file: reuses the No-repeat genome's `recent` buffer as the entity-recency
# signal, substitutes a single generic pronoun ("it") for a content word
# re-mentioned within PRONOM_WINDOW words, with probability PRONOM_PROB.
PRONOM_WINDOW = 15
PRONOM_PROB = 0.6


class Service:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False
        self.loading = False
        self.err = None
        self.champs = {}
        self.chunks = {}

    def ensure(self):
        with self.lock:
            if self.ready or self.loading:
                return
            self.loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            wp.build_word_corpus(4000); wp.induce_word_classes(NCL)
            self.table, self.w2c, self.vocab, self.nc, self.cids = wp.build_class_words(NCL)
            self.stoi = {w: i for i, w in enumerate(self.vocab)}
            self.feat, _ = wp.word_features(4000, D)
            self.cents = wp.class_centroids(NCL, D)
            ids, _, _ = wp.build_word_corpus(4000)
            self.ids = ids
            self.logfreq = np.log1p(np.bincount(ids, minlength=len(self.vocab)).astype(np.float32))
            self.cprob = (np.bincount(self.cids, minlength=self.nc).astype(np.float64)
                          / len(self.cids))
            try:
                self.chunks = wp.build_chunk_index(NCL)
            except Exception:
                self.chunks = {}
            if os.path.exists(CACHE):
                with open(CACHE, "rb") as f:
                    self.champs = pickle.load(f)
            # re-rank genomes: fixed features over the shared vocab + evolved
            # bilinear heads (from cache). Used to re-rank selection candidates.
            self.agree_feats = ag.gram_feats(self.vocab)     # agreement (finiteness/number)
            self.altern_feats = al.func_feats(self.vocab)    # content-function alternation
            # per-class centroids in each genome's feature space (freq-weighted over
            # members) — lets the constraint heads bias the ORDER skeleton, not just
            # selection. Same head, lifted one level up.
            self.altern_classfeat = np.zeros((self.nc, al.NF), np.float32)
            self.agree_classfeat = np.zeros((self.nc, ag.NG), np.float32)
            for cl, (mem, p) in self.table.items():
                self.altern_classfeat[cl] = p @ self.altern_feats[mem]
                self.agree_classfeat[cl] = p @ self.agree_feats[mem]
            # content mask for the repetition genome (function words may recur)
            self.is_content = rp.content_mask(self.vocab)
            # open-obligation mask: prepositions/determiners/articles/"to" open
            # a phrase that needs a following content word to close — reuses
            # the SAME closed-class sets altern.py already defines, no new
            # hand-labeling.
            _oblig_open_words = al.PREPS | al.ARTICLES | al.DET | al.TO
            self.oblig_open = np.array([w in _oblig_open_words for w in self.vocab], dtype=bool)
            # per-word sentence-opener score (func features · evolved weights)
            self.open_scores = (self.altern_feats @ self.champs["open"]
                                if "open" in self.champs else None)
            # per-word sentence-closer score + emission-weighted centre (so the closer
            # is rate-preserving: it moves periods toward good enders without changing
            # the overall sentence rate).
            if "close" in self.champs:
                self.close_scores = self.altern_feats @ self.champs["close"]
                freq = np.bincount(self.ids, minlength=len(self.vocab)).astype(np.float64)
                self.close_center = float((self.close_scores * freq).sum() / freq.sum())
            else:
                self.close_scores = None; self.close_center = 0.0
            # standalone relation genomes (hypernym/meronym/synant — Wikipedia-
            # trained, crosswalked into this vocab). Absent files just mean
            # those toggles have nothing to do; same graceful-degradation
            # pattern as every other genome here.
            self.rel = rel_wire.load_relation_genomes(self.vocab)
            self.rel_coverage = (round(next(iter(self.rel.values()))["coverage"] * 100)
                                 if self.rel else 0)
            # EXPERIMENTAL sentence-type genome (question-opener bias)
            if os.path.exists(SENT_TYPE_CHAMP):
                with open(SENT_TYPE_CHAMP, "rb") as f:
                    st_d = pickle.load(f)
                self.sent_type_scores = st.word_scores(st_d["champ"], self.vocab)
                self.question_rate = st_d["question_rate"]
            else:
                self.sent_type_scores = None
                self.question_rate = 0.0
            # EXPERIMENTAL sentence-length-plan genome (opener + boundary-shaping bias)
            if os.path.exists(SENT_LENPLAN_CHAMP):
                with open(SENT_LENPLAN_CHAMP, "rb") as f:
                    sl_d = pickle.load(f)
                self.lenplan_scores = sl.word_scores(sl_d["champ"], self.vocab)
                self.lenplan_median = sl_d["median_len"]
            else:
                self.lenplan_scores = None
                self.lenplan_median = 14.0
            # EXPERIMENTAL pronominalization target word ("it" — generic, no
            # gender/number modeling)
            self.pronom_word = self.stoi.get("it")
            self.ready = True
        except Exception as exc:                       # pragma: no cover
            import traceback; traceback.print_exc()
            self.err = f"{type(exc).__name__}: {exc}"
        finally:
            self.loading = False

    @staticmethod
    def _nparams(x):
        if isinstance(x, np.ndarray):
            return x.size
        if isinstance(x, (tuple, list)):
            return sum(Service._nparams(e) for e in x)
        if isinstance(x, dict):
            return sum(Service._nparams(e) for e in x.values())
        return 0

    @staticmethod
    def _nbytes(x):
        if isinstance(x, np.ndarray):
            return x.nbytes
        if isinstance(x, (tuple, list)):
            return sum(Service._nbytes(e) for e in x)
        if isinstance(x, dict):
            return sum(Service._nbytes(e) for e in x.values())
        return 0

    def _footprint(self):
        """Live param count + deploy footprint, measured from the loaded genomes/data
        (not hardcoded). heads = the evolved genomes; full = everything the pipeline
        needs at inference EXCEPT features derivable from the vocab at load."""
        rel_M = {k: v["M"] for k, v in self.rel.items()}          # evolved heads only
        rel_extra = {k: v["feat"] for k, v in self.rel.items()}   # crosswalk arrays (not "evolved")
        params = sum(self._nparams(v) for v in self.champs.values()) + self._nparams(rel_M)
        heads = self._nbytes(self.champs) + self._nbytes(rel_M)
        chunk_b = sum(rows.nbytes + p.nbytes
                      for d in self.chunks.values() for rows, p in d.values())
        table_b = sum(m.nbytes + p.nbytes for m, p in self.table.values())
        full = (heads + self.feat.nbytes + self.cents.nbytes + self.logfreq.nbytes
                + table_b + chunk_b + self._nbytes(rel_extra))
        return params, heads, full

    def status(self):
        s = {"ready": self.ready, "loading": self.loading, "err": self.err,
             "has_genomes": bool(self.champs),
             "trained": sorted(self.champs.keys()) + sorted("rel_" + k for k in self.rel.keys())
             if self.ready else [],
             "corpus_chars": int(len(self.cids)) if self.ready else 0,
             "vocab": len(self.vocab) if self.ready else 0,
             "n_classes": NCL,
             "rel_coverage_pct": self.rel_coverage if self.ready else 0,
             "chunk_phrases": (sum(len(v) for v in self.chunks.get(2, {}).values())
                               + sum(len(v) for v in self.chunks.get(3, {}).values()))
             if self.ready else 0}
        if self.ready:
            params, heads, full = self._footprint()
            s["params"] = int(params)
            s["heads_kb"] = round(heads / 1024)
            s["full_kb"] = round(full / 1024)
        return s

    def _bound_prob(self, en, cl, cur, prev_w, oblig_depth=0, len_factor=1.0):
        """P(sentence ends here) from the boundary genome, optionally reshaped by the
        closer genome so periods land after good ender-words (rate-preserving), by the
        EXPERIMENTAL obligation tracker (suppress ending while a prep/det/article opened
        a phrase that hasn't been closed by a content word yet), and by the EXPERIMENTAL
        length-plan genome (lean toward finishing before/after the planned median)."""
        pb = wp.boundary_prob(self.champs["bound"], cl, cur)
        if en.get("close") and self.close_scores is not None and prev_w is not None and 0 < pb < 1:
            pb = min(1.0, pb * np.exp(CLOSE_GAMMA * (self.close_scores[prev_w] - self.close_center)))
        if en.get("oblig") and oblig_depth > 0 and 0 < pb < 1:
            pb = pb * np.exp(-OBLIG_GAMMA * oblig_depth)
        if len_factor != 1.0 and 0 < pb < 1:
            pb = min(1.0, pb * len_factor)
        return pb

    def _punct(self, en, cl, cur, clause, rng, prev_w=None, oblig_depth=0, len_factor=1.0):
        """Decide end-of-word punctuation: 'period' (sentence end), 'comma'
        (clause break), or None. Period wins over comma."""
        if cur >= 55:
            return "period"
        if en.get("bound") and "bound" in self.champs and cur >= 4 \
                and rng.random() < self._bound_prob(en, cl, cur, prev_w, oblig_depth, len_factor):
            return "period"
        if en.get("commas") and "comma" in self.champs and clause >= 3 \
                and rng.random() < wp.boundary_prob(self.champs["comma"], cl, clause):
            return "comma"
        return None

    def generate(self, en, n=260, seed=0):
        if not self.ready:
            return ""
        rng = np.random.default_rng(seed)
        if en.get("order") and "order" in self.champs:
            order_reranks = []
            if en.get("altern") and "altern" in self.champs and ORDER_ALTERN_GAMMA > 0:
                order_reranks.append((self.altern_classfeat, self.champs["altern"], ORDER_ALTERN_GAMMA))
            if en.get("agree") and "agree" in self.champs and ORDER_AGREE_GAMMA > 0:
                order_reranks.append((self.agree_classfeat, self.champs["agree"], ORDER_AGREE_GAMMA))
            cls_seq = wp.gen_class_seq(self.champs["order"], C, n, self.cids[500:500 + C],
                                       rng, 0.8, reranks=order_reranks or None)
        else:
            cls_seq = list(rng.choice(self.nc, size=n, p=self.cprob))
        use_chunk = en.get("chunks") and self.chunks and en.get("vocab")
        reranks = []
        if en.get("altern") and "altern" in self.champs:
            reranks.append((self.altern_feats, self.champs["altern"], ALTERN_GAMMA))
        if en.get("agree") and "agree" in self.champs:
            reranks.append((self.agree_feats, self.champs["agree"], AGREE_GAMMA))
        if en.get("sem") and "sem" in self.champs:
            reranks.append((self.feat, self.champs["sem"], SEM_GAMMA))
        if en.get("hyper") and "hyper" in self.rel:
            reranks.append((self.rel["hyper"]["feat"], self.rel["hyper"]["M"], HYPER_GAMMA))
        if en.get("mero") and "mero" in self.rel:
            reranks.append((self.rel["mero"]["feat"], self.rel["mero"]["M"], MERO_GAMMA))
        if en.get("synant") and "synant" in self.rel:
            reranks.append((self.rel["synant"]["feat"], self.rel["synant"]["M"], SYNANT_GAMMA))
        reranks = reranks or None
        use_rep = en.get("rep") and "rep" in self.champs
        use_oblig = en.get("oblig")
        use_sent_type = en.get("sent_type") and self.sent_type_scores is not None
        use_lenplan = en.get("lenplan") and self.lenplan_scores is not None
        use_pronom = en.get("pronominal") and self.pronom_word is not None
        recent = []                                    # emitted word ids, for the rep genome
        parts, prev, cur, clause, j = [], None, 0, 0, 0
        oblig_depth = 0                                 # EXPERIMENTAL open-obligation tracker
        is_question = (use_sent_type and rng.random() < self.question_rate)  # EXPERIMENTAL sentence-type
        is_long = (use_lenplan and rng.random() < LENPLAN_LONG_RATE)  # EXPERIMENTAL length-plan
        def _len_factor(c):
            if not use_lenplan:
                return 1.0
            diff = float(np.clip(c - self.lenplan_median, -20, 20))
            f = np.exp(LENPLAN_LEN_GAMMA * diff) if is_long else np.exp(-LENPLAN_LEN_GAMMA * diff)
            return float(np.clip(f, 0.2, 5.0))
        def _track_oblig(w):
            nonlocal oblig_depth
            if not use_oblig:
                return
            if self.oblig_open[w]:
                oblig_depth = min(4, oblig_depth + 1)
            elif self.is_content[w]:
                oblig_depth = max(0, oblig_depth - 1)
        while j < len(cls_seq):
            cl = int(cls_seq[j])
            if cl not in self.table:
                # The reserved <unk> class isn't emitted as a word, but it carries most
                # of the sentence-boundary signal (rare sentence-final words fall into it).
                # Without this, that signal is dropped and sentences run to the length cap.
                if en.get("vocab") and en.get("bound") and "bound" in self.champs \
                        and cur >= 4 and rng.random() < self._bound_prob(en, cl, cur, prev, oblig_depth, _len_factor(cur)):
                    parts.append("?" if is_question else ".")
                    cur = 0; clause = 0; oblig_depth = 0
                    is_question = use_sent_type and rng.random() < self.question_rate
                    is_long = use_lenplan and rng.random() < LENPLAN_LONG_RATE
                j += 1; continue
            emitted = 0
            # try a real phrase whose class pattern matches the upcoming skeleton
            if use_chunk:
                for L in (3, 2):
                    if j + L <= len(cls_seq):
                        ct = tuple(int(x) for x in cls_seq[j:j + L])
                        b = self.chunks.get(L, {}).get(ct)
                        if b is not None and rng.random() < 0.6:
                            rows, p = b
                            ch = rows[rng.choice(len(rows), p=p)]
                            for w in ch:
                                parts.append(self.vocab[int(w)]); prev = int(w); recent.append(int(w))
                                _track_oblig(int(w))
                            emitted = len(ch); j += L; break
            if emitted == 0:
                if not en.get("vocab"):
                    Ln = int(rng.integers(2, 9))
                    parts.append("".join(chr(rng.integers(97, 123)) for _ in range(Ln)))
                else:
                    mem = self.table[cl][0]
                    bonus = None
                    if use_rep and prev is not None:
                        bonus = rp.penalty(self.champs["rep"], recent, mem, self.is_content)
                    # sentence-opener: bias the first word of each sentence (cur == 0)
                    if en.get("open") and self.open_scores is not None and prev is not None and cur == 0:
                        ob = OPEN_GAMMA * self.open_scores[mem]
                        bonus = ob if bonus is None else bonus + ob
                    # EXPERIMENTAL sentence-type: bias the first word toward a
                    # question-opener when this sentence was flagged a question
                    if is_question and prev is not None and cur == 0:
                        qb = SENT_TYPE_GAMMA * self.sent_type_scores[mem]
                        bonus = qb if bonus is None else bonus + qb
                    # EXPERIMENTAL length-plan: bias the first word toward/away from
                    # this genome's long-sentence-opener scores
                    if use_lenplan and prev is not None and cur == 0:
                        lb = LENPLAN_GAMMA * self.lenplan_scores[mem] * (1.0 if is_long else -1.0)
                        bonus = lb if bonus is None else bonus + lb
                    if en.get("sel") == "bi" and "bisel" in self.champs and prev is not None:
                        nxt = next((int(cls_seq[k]) for k in range(j + 1, len(cls_seq))
                                    if int(cls_seq[k]) in self.table), cl)
                        w = wp._fill_bisel(prev, cl, nxt, self.table, self.feat, self.logfreq,
                                           self.cents, self.champs["bisel"], rng,
                                           reranks=reranks, bonus=bonus)
                    elif en.get("sel") in ("uni", "bi") and "sel" in self.champs and prev is not None:
                        w = wp._fill_selected(prev, cl, self.table, self.feat, self.logfreq,
                                              self.champs["sel"], rng, reranks=reranks, bonus=bonus)
                    else:
                        w = int(rng.choice(mem, p=self.table[cl][1]))
                    # EXPERIMENTAL Passage-stage pronominalization: a content word
                    # that was already emitted recently gets replaced by a generic
                    # pronoun instead of repeating the literal noun. Reuses the
                    # No-repeat genome's `recent` buffer as the entity-recency
                    # signal — no new training, no gender/number (single generic
                    # pronoun only).
                    if (use_pronom and self.pronom_word is not None and prev is not None
                            and self.is_content[w] and w in recent[-PRONOM_WINDOW:]
                            and rng.random() < PRONOM_PROB):
                        w_out = self.pronom_word
                        parts.append(self.vocab[w_out]); prev = w_out; recent.append(w)
                    else:
                        parts.append(self.vocab[w]); prev = w; recent.append(w)
                    _track_oblig(w)
                emitted = 1; j += 1
            cur += emitted; clause += emitted
            if en.get("vocab"):
                mark = self._punct(en, cl, cur, clause, rng, prev_w=prev, oblig_depth=oblig_depth,
                                   len_factor=_len_factor(cur))
                if mark == "period":
                    parts.append("?" if is_question else ".")
                    cur = 0; clause = 0; oblig_depth = 0
                    is_question = use_sent_type and rng.random() < self.question_rate
                    is_long = use_lenplan and rng.random() < LENPLAN_LONG_RATE
                elif mark == "comma":
                    parts.append(","); clause = 0
        text = " ".join(parts).replace(" .", ".").replace(" ,", ",").replace(" ?", "?")
        return re.sub(r"(^|[.?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

    # ------------------------------------------------------------------
    # EXPERIMENTAL Revision stage (genreg_train roadmap: whole_sent, best_of_n).
    # Every genome above scores LOCALLY, word by word, during generation. This
    # reads a COMPLETE sentence after the fact and scores it as a whole, then
    # Best-of-N keeps the best of several independently-generated candidates.
    # No new training — this is a composed judgment over already-evolved
    # champions (Semantic, Closer, Opener, Alternation), hence abstraction-tier.
    # ------------------------------------------------------------------
    def _sentence_score(self, sent_text, en):
        """Composite whole-sentence fitness from already-evolved champions.
        Higher is better. Returns None if the sentence has no usable words."""
        words = [w for w in sent_text.strip(" .?").split(" ") if w]
        if not words:
            return None
        ids = [self.stoi.get(w.lower(), 0) for w in words]
        score = 0.0
        # semantic adjacency: consecutive content words should co-occur
        if en.get("sem") and "sem" in self.champs and self.is_content is not None:
            M = self.champs["sem"]
            pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)
                     if ids[i] and ids[i + 1] and self.is_content[ids[i]] and self.is_content[ids[i + 1]]]
            if pairs:
                score += SEM_GAMMA * np.mean([self.feat[a] @ M @ self.feat[b] for a, b in pairs])
        # opener fit
        if en.get("open") and self.open_scores is not None and ids[0]:
            score += OPEN_GAMMA * self.open_scores[ids[0]]
        # closer fit (rate-preserving, centred)
        if en.get("close") and self.close_scores is not None and ids[-1]:
            score += CLOSE_GAMMA * (self.close_scores[ids[-1]] - self.close_center)
        # alternation: penalize adjacent function-function runs
        if en.get("altern") and self.is_content is not None:
            ff_runs = sum(1 for i in range(len(ids) - 1)
                         if ids[i] and ids[i + 1] and not self.is_content[ids[i]]
                         and not self.is_content[ids[i + 1]])
            score -= ALTERN_GAMMA * 0.3 * ff_runs
        # no-repeat: penalize a content word reused within the sentence
        if en.get("rep") and self.is_content is not None:
            seen = set(); reps = 0
            for wid in ids:
                if wid and self.is_content[wid]:
                    if wid in seen:
                        reps += 1
                    seen.add(wid)
            score -= 0.5 * reps
        # degenerate-length guard (too short reads as a fragment, too long rambles)
        if len(words) < 3 or len(words) > 60:
            score -= 5.0
        return score

    def generate_revision(self, en, n_sentences=6, n_candidates=6, seed=0):
        """EXPERIMENTAL Best-of-N over the Whole-sentence scorer. For each
        sentence slot, generate several independent candidates with the
        UNCHANGED pipeline (same generate(), different seeds), score each
        whole sentence, keep the best. No new training."""
        if not self.ready:
            return ""
        rng = np.random.default_rng(seed)
        kept = []
        for s in range(n_sentences):
            best_text, best_score = None, None
            for c in range(n_candidates):
                cand_seed = int(rng.integers(0, 2**31 - 1))
                text = self.generate(en, n=48, seed=cand_seed)
                m = re.match(r"^(.*?[.?])", text)
                first = m.group(1) if m else text
                sc = self._sentence_score(first, en)
                if sc is None:
                    continue
                if best_score is None or sc > best_score:
                    best_text, best_score = first, sc
            if best_text:
                kept.append(best_text)
        return " ".join(kept)

    # ------------------------------------------------------------------
    # EXPERIMENTAL meaning-first generation (2026-07-08). Every mode above
    # is STRUCTURE-first: the Order genome picks a class skeleton blind, Fill
    # picks whatever word satisfies local constraints in each slot, and
    # "meaning" is bolted on afterward as a rerank bias. That's very likely
    # why Sentence coherence / Theme consistency (see genomes.txt) failed
    # near-chance — a linear rerank can't retrofit global coherence onto a
    # sequence that was never chosen for its content.
    #
    # This flips it: pick 3-5 semantically related CONTENT words FIRST
    # (using the already-evolved relation genomes — hyper/mero/synant/sem —
    # exactly the genomes built for "is this related to that"), THEN let the
    # already-trained Order skeleton run as normal and claim each reserved
    # word into the first slot whose class matches, instead of running
    # word-selection there. Structure still comes from the same evolved
    # genomes; it just accommodates chosen meaning instead of the reverse.
    # No new training — this recombines existing champions into a new flow.
    # ------------------------------------------------------------------
    def _select_content(self, en, rng, n_content=4, temp=0.35):
        """Pick n_content content-word ids that are mutually related, using
        whichever relation genomes are enabled (falls back to Semantic
        adjacency alone if none of hyper/mero/synant are on). Stochastic
        (softmax sample, not argmax) at every step — no ridge/argmax
        shortcuts, per project convention."""
        content_ids = np.where(self.is_content)[0]
        content_ids = content_ids[content_ids != 0]
        freq = self.logfreq[content_ids]
        p0 = freq / freq.sum()
        seed_w = int(rng.choice(content_ids, p=p0))
        selected = [seed_w]

        mats = []
        if en.get("sem", True) and "sem" in self.champs:
            mats.append((self.feat, self.champs["sem"]))
        if en.get("hyper") and "hyper" in self.rel:
            mats.append((self.rel["hyper"]["feat"], self.rel["hyper"]["M"]))
        if en.get("mero") and "mero" in self.rel:
            mats.append((self.rel["mero"]["feat"], self.rel["mero"]["M"]))
        if en.get("synant") and "synant" in self.rel:
            mats.append((self.rel["synant"]["feat"], self.rel["synant"]["M"]))
        if not mats and "sem" in self.champs:
            mats.append((self.feat, self.champs["sem"]))

        for _ in range(n_content - 1):
            remaining = np.array([w for w in content_ids if w not in selected])
            if len(remaining) == 0:
                break
            scores = np.zeros(len(remaining), np.float64)
            for feat, M in mats:
                for s in selected:
                    scores += feat[remaining] @ M @ feat[s]
            scores /= max(1, len(mats) * len(selected))
            z = scores - scores.max()
            probs = np.exp(z / temp)
            probs /= probs.sum()
            nxt = int(rng.choice(remaining, p=probs))
            selected.append(nxt)
        return selected

    def generate_meaning_first(self, en, n_content=4, n=140, seed=0):
        """EXPERIMENTAL. Pick content first (_select_content), then run the
        SAME evolved Order/Fill genomes as generate(), except each reserved
        content word claims the first upcoming slot whose class matches its
        own (self.w2c), instead of that slot running normal word-selection.
        Structural toggles (altern/agree/sem/rep/open/close/bound/commas/
        chunks) behave exactly as in generate() for every non-claimed slot.
        """
        if not self.ready:
            return ""
        rng = np.random.default_rng(seed)
        reserved = self._select_content(en, rng, n_content)
        pending = {int(w): int(self.w2c[w]) for w in reserved}   # word -> class, unplaced

        if en.get("order") and "order" in self.champs:
            order_reranks = []
            if en.get("altern") and "altern" in self.champs and ORDER_ALTERN_GAMMA > 0:
                order_reranks.append((self.altern_classfeat, self.champs["altern"], ORDER_ALTERN_GAMMA))
            if en.get("agree") and "agree" in self.champs and ORDER_AGREE_GAMMA > 0:
                order_reranks.append((self.agree_classfeat, self.champs["agree"], ORDER_AGREE_GAMMA))
            cls_seq = wp.gen_class_seq(self.champs["order"], C, n, self.cids[500:500 + C],
                                       rng, 0.8, reranks=order_reranks or None)
        else:
            cls_seq = list(rng.choice(self.nc, size=n, p=self.cprob))

        reranks = []
        if en.get("altern") and "altern" in self.champs:
            reranks.append((self.altern_feats, self.champs["altern"], ALTERN_GAMMA))
        if en.get("agree") and "agree" in self.champs:
            reranks.append((self.agree_feats, self.champs["agree"], AGREE_GAMMA))
        if en.get("sem") and "sem" in self.champs:
            reranks.append((self.feat, self.champs["sem"], SEM_GAMMA))
        reranks = reranks or None
        use_rep = en.get("rep") and "rep" in self.champs

        recent, parts, prev, cur, clause, j = [], [], None, 0, 0, 0
        placed_any = False
        while j < len(cls_seq):
            cl = int(cls_seq[j])
            if cl not in self.table:
                if en.get("vocab") and en.get("bound") and "bound" in self.champs \
                        and cur >= 4 and rng.random() < self._bound_prob(en, cl, cur, prev):
                    parts.append("."); cur = 0; clause = 0
                j += 1; continue
            claim_w = next((w for w, c in pending.items() if c == cl), None)
            if claim_w is not None and en.get("vocab"):
                w = claim_w
                del pending[claim_w]
                placed_any = True
            elif not en.get("vocab"):
                Ln = int(rng.integers(2, 9))
                parts.append("".join(chr(rng.integers(97, 123)) for _ in range(Ln)))
                cur += 1; clause += 1; j += 1; continue
            else:
                mem = self.table[cl][0]
                bonus = None
                if use_rep and prev is not None:
                    bonus = rp.penalty(self.champs["rep"], recent, mem, self.is_content)
                if en.get("open") and self.open_scores is not None and prev is not None and cur == 0:
                    ob = OPEN_GAMMA * self.open_scores[mem]
                    bonus = ob if bonus is None else bonus + ob
                if en.get("sel") == "bi" and "bisel" in self.champs and prev is not None:
                    nxt = next((int(cls_seq[k]) for k in range(j + 1, len(cls_seq))
                               if int(cls_seq[k]) in self.table), cl)
                    w = wp._fill_bisel(prev, cl, nxt, self.table, self.feat, self.logfreq,
                                       self.cents, self.champs["bisel"], rng,
                                       reranks=reranks, bonus=bonus)
                elif en.get("sel") in ("uni", "bi") and "sel" in self.champs and prev is not None:
                    w = wp._fill_selected(prev, cl, self.table, self.feat, self.logfreq,
                                          self.champs["sel"], rng, reranks=reranks, bonus=bonus)
                else:
                    w = int(rng.choice(mem, p=self.table[cl][1]))
            parts.append(self.vocab[w]); prev = w; recent.append(w)
            cur += 1; clause += 1; j += 1
            if en.get("vocab"):
                mark = self._punct(en, cl, cur, clause, rng, prev_w=prev)
                if mark == "period":
                    parts.append("."); cur = 0; clause = 0
                elif mark == "comma":
                    parts.append(","); clause = 0
        text = " ".join(parts).replace(" .", ".").replace(" ,", ",")
        text = re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        content_words = [self.vocab[w] for w in reserved]
        return {"text": text, "content_words": content_words,
                "placed": len(reserved) - len(pending), "requested": len(reserved)}


SERVICE = Service()
