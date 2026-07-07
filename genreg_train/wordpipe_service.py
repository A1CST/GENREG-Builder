"""Web backend for the /evolang page — the WordPipe specialist pipeline.

Lazy-loads the trained genomes (demo/genomes.pkl) + corpus caches in a background
thread on first request, then generates text with any subset of the evolved
specialist layers enabled. This is the current evolution-native language model:
a pipeline of tiny gradient-free genomes (order / selection / boundary / chunks),
composed — not one net trained on a loss. See documentation/WORDPIPE_FINDINGS.md.
"""
import os
import pickle
import threading

import numpy as np

from genreg_train import wordpipe as wp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "demo", "genomes.pkl")
NCL, C, D = 32, 4, 24


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
            self.ready = True
        except Exception as exc:                       # pragma: no cover
            import traceback; traceback.print_exc()
            self.err = f"{type(exc).__name__}: {exc}"
        finally:
            self.loading = False

    def status(self):
        return {"ready": self.ready, "loading": self.loading, "err": self.err,
                "has_genomes": bool(self.champs),
                "trained": sorted(self.champs.keys()),
                "corpus_chars": int(len(self.cids)) if self.ready else 0,
                "vocab": len(self.vocab) if self.ready else 0,
                "n_classes": NCL,
                "chunk_phrases": (sum(len(v) for v in self.chunks.get(2, {}).values())
                                  + sum(len(v) for v in self.chunks.get(3, {}).values()))
                if self.ready else 0}

    def _punct(self, en, cl, cur, clause, rng):
        """Decide end-of-word punctuation: 'period' (sentence end), 'comma'
        (clause break), or None. Period wins over comma."""
        if cur >= 55:
            return "period"
        if en.get("bound") and "bound" in self.champs and cur >= 4 \
                and rng.random() < wp.boundary_prob(self.champs["bound"], cl, cur):
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
            cls_seq = wp.gen_class_seq(self.champs["order"], C, n, self.cids[500:500 + C], rng, 0.8)
        else:
            cls_seq = list(rng.choice(self.nc, size=n, p=self.cprob))
        use_chunk = en.get("chunks") and self.chunks and en.get("vocab")
        parts, prev, cur, clause, j = [], None, 0, 0, 0
        while j < len(cls_seq):
            cl = int(cls_seq[j])
            if cl not in self.table:
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
                                parts.append(self.vocab[int(w)]); prev = int(w)
                            emitted = len(ch); j += L; break
            if emitted == 0:
                if not en.get("vocab"):
                    Ln = int(rng.integers(2, 9))
                    parts.append("".join(chr(rng.integers(97, 123)) for _ in range(Ln)))
                else:
                    mem = self.table[cl][0]
                    if en.get("sel") == "bi" and "bisel" in self.champs and prev is not None:
                        nxt = next((int(cls_seq[k]) for k in range(j + 1, len(cls_seq))
                                    if int(cls_seq[k]) in self.table), cl)
                        w = wp._fill_bisel(prev, cl, nxt, self.table, self.feat, self.logfreq,
                                           self.cents, self.champs["bisel"], rng)
                    elif en.get("sel") in ("uni", "bi") and "sel" in self.champs and prev is not None:
                        w = wp._fill_selected(prev, cl, self.table, self.feat, self.logfreq,
                                              self.champs["sel"], rng)
                    else:
                        w = int(rng.choice(mem, p=self.table[cl][1]))
                    parts.append(self.vocab[w]); prev = w
                emitted = 1; j += 1
            cur += emitted; clause += emitted
            if en.get("vocab"):
                mark = self._punct(en, cl, cur, clause, rng)
                if mark == "period":
                    parts.append("."); cur = 0; clause = 0
                elif mark == "comma":
                    parts.append(","); clause = 0
        import re
        text = " ".join(parts).replace(" .", ".").replace(" ,", ",")
        return re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


SERVICE = Service()
