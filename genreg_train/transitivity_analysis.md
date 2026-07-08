# Transitivity — design analysis (no training run; redundant as specified)

Roadmap description: "If a verb requires an object, one must follow. Word-level
(not class-level), so it may dodge the redundancy wall that cut the earlier
class-level Verb argument genome."

## Why a training run isn't warranted here

There are no POS tags anywhere in this pipeline — only induced distributional
word classes. "Does this word (as a verb) require an object" can only be
approximated from position, and the only positional signals available are:

1. "Is a content word about to follow, or is this the last content word
   before a boundary?" — this is EXACTLY what the already-shipped **Closer**
   genome learns (a unary ender classifier: is this word a plausible
   sentence-final word, i.e. does nothing else need to follow it). A word
   that always "needs completion" (transitive-verb-like) is, by definition,
   a word Closer already learns to score LOW as an ender. Framing Transitivity
   at the word level collapses onto the same signal Closer already captures,
   not a new one.
2. "Is a content word about to follow, specifically because THIS word is a
   verb governing an object" — this is the actual target, and it's exactly
   the question the class-level **Verb argument** genome (already CUT, see
   `genomes.txt`) failed to answer, for the same underlying reason: without
   POS tags there's no way to isolate "verb-object" continuation from any
   other reason a content word might follow (compound nouns, adjective
   chains, appositives, etc). Moving from class-level to word-level features
   doesn't fix this — it's a labeling problem (what counts as a positive
   "verb requiring an object" example), not a resolution problem, and
   word-level features don't add label information.

## Verdict

**CUT as redundant**, not attempted. Word-level Transitivity would either
(a) re-derive a weaker copy of the already-shipped Closer genome, or
(b) re-hit the exact wall that already cut Verb argument, depending on how
loosely "requires an object" is mined. Neither justifies a training run.
A genuine Transitivity genome would need real POS/dependency information
this pipeline doesn't have — out of scope without a labeling source beyond
raw corpus statistics.
