"""Diagnostic: characterize wiki_corpus.txt's composition to explain why the
wiki-trained WordPipe genomes produce heavy "of the X of the Y" chains.
Hypothesis: the corpus is dominated by short, formulaic geography/biography
stub articles ("X is a town in... population census...") which are MORE
repetitive/templated than the Gutenberg novel prose was. Measures: article
length distribution (one article per line, per build_wiki_corpus.py),
frequency of template phrases ("is a town", "population census", "is a
municipality", "arrondissement"), and top bigrams by raw count. Runs on the
I2 primary -- read-only over a 316MB file, still dispatched remotely per
the no-local-heavy-compute constraint.
"""
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "wiki_corpus_diag.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

CORPUS = os.path.join(ROOT, "corpora", "wikipedia", "wiki_corpus.txt")
if not os.path.exists(CORPUS):
    log("FATAL: corpus missing"); sys.exit(1)

log(f"reading {CORPUS} ({os.path.getsize(CORPUS)/1e6:.0f} MB)...")

lengths = []
template_hits = Counter()
TEMPLATES = ["is a town", "is a village", "is a municipality", "population census",
            "arrondissement", "is a city", "is a species", "is a genus",
            "was an american", "was a british", "is a district", "national census"]
bigram_counts = Counter()
n_lines = 0
n_words_total = 0

with open(CORPUS, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        n_lines += 1
        words = line.split()
        lengths.append(len(words))
        n_words_total += len(words)
        low = line.lower()
        for t in TEMPLATES:
            if t in low:
                template_hits[t] += 1
        if n_lines <= 2_000_000:   # cap bigram sampling for speed
            toks = [w.strip(".,;:!?\"'()").lower() for w in words]
            for i in range(len(toks) - 1):
                if toks[i] and toks[i + 1]:
                    bigram_counts[(toks[i], toks[i + 1])] += 1
        if n_lines % 500000 == 0:
            log(f"  ...{n_lines} lines read")

lengths.sort()
n = len(lengths)
log(f"\ntotal articles/lines: {n}")
log(f"total words: {n_words_total}")
log(f"length percentiles (words): p10={lengths[n//10]} p25={lengths[n//4]} "
   f"p50={lengths[n//2]} p75={lengths[3*n//4]} p90={lengths[9*n//10]} max={lengths[-1]}")
short_thresh = 40
n_short = sum(1 for l in lengths if l < short_thresh)
log(f"articles under {short_thresh} words: {n_short} ({100*n_short/n:.1f}%)")

log("\ntemplate-phrase hits (articles containing this phrase):")
for t, c in template_hits.most_common():
    log(f"  {t!r}: {c} articles ({100*c/n:.1f}%)")

log("\ntop 30 bigrams by raw count (first 2M lines sampled):")
for (a, b), c in bigram_counts.most_common(30):
    log(f"  {a} {b}: {c}")

log("\nDONE")
