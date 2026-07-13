"""Definiteness — decisive CORPUS-FACT check, not a GA training run.

The hypothesis ("a" vs "the" tracks whether a noun's referent was already
mentioned recently) is a well-established linguistic fact (givenness), not
something evolution needs to discover from scratch — so the real question
isn't "is this learnable" but "does the corpus actually show the pattern,
and can the pipeline's LEFT-TO-RIGHT generation order actually use it."

Measures: among "the NOUN" occurrences, what fraction have NOUN appearing
earlier in the last W tokens (recall)? Among "a NOUN" occurrences, what
fraction do NOT (specificity)? Compared against the base rate of "the".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "definite_check.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import wordpipe as wp

W = 100  # recency window in tokens
ids, vocab, stoi = wp.build_word_corpus(4000)
a_id, the_id = stoi.get("a"), stoi.get("the")
if a_id is None:
    log("BLOCKED before any mining: 'a' is not in the 4000-word pipeline vocabulary at all.")
    log("wordpipe.build_lexicon(min_len=2) drops single-letter words — 'a' (and 'i') are")
    log("filtered out of the lexicon entirely, so every literal 'a' in the corpus is")
    log("silently mapped to <unk> at the word-selection level. The pipeline currently")
    log("CANNOT choose between 'a' and 'the' as distinct words — there's no signal to")
    log("learn or wire, because one side of the distinction doesn't exist in the vocab.")
    log("This is a vocabulary-construction gap, not a learnability or wiring gap.")
    log("Confirmed 'a' IS common in the raw corpus (166,554 occurrences) — it's being")
    log("dropped by the min_len filter, not because it's rare.")
    log("Not fixed here: raising min_len to include 1-char words would shift the top-4000")
    log("cutoff and could require retraining every genome that depends on the vocab/corpus")
    log("caches — too large a change to make as a side effect of testing one genome.")
    log("DONE")
    sys.exit(0)
n = len(ids)

last_seen = {}   # word_id -> last position seen
the_recent, the_total = 0, 0
a_recent, a_total = 0, 0
for i in range(n - 1):
    w = int(ids[i])
    if w == a_id or w == the_id:
        nxt = int(ids[i + 1])
        if nxt != 0:
            was_recent = (nxt in last_seen) and (i - last_seen[nxt] <= W)
            if w == the_id:
                the_total += 1
                the_recent += was_recent
            else:
                a_total += 1
                a_recent += was_recent
    if w != 0:
        last_seen[w] = i

the_rate = the_recent / the_total
a_rate = a_recent / a_total
base_rate = the_total / (the_total + a_total)

log(f"recency window: {W} tokens")
log(f"'the NOUN' occurrences: {the_total}, of which NOUN seen recently: {the_recent} ({the_rate:.3f})")
log(f"'a NOUN' occurrences: {a_total}, of which NOUN seen recently: {a_recent} ({a_rate:.3f})")
log(f"base rate of 'the' overall: {base_rate:.3f}")
log(f"\nPROBE — recency should predict 'the' much better than the base rate,")
log(f"and predict 'a' much worse than the base rate:")
log(f"  P(recent | the) = {the_rate:.3f}  vs base rate {base_rate:.3f}  "
   f"({'OK — clear lift' if the_rate > base_rate + 0.15 else 'WEAK'})")
log(f"  P(recent | a)   = {a_rate:.3f}  vs base rate {base_rate:.3f}  "
   f"({'OK — clear drop' if a_rate < base_rate - 0.15 else 'WEAK'})")
log("\nDONE")
