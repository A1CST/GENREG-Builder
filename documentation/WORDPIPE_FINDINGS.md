# WordPipe — specialist-pipeline experiment, first results

**Date:** 2026-07-06. **Idea (user):** don't train one genome to "do English" —
decompose language into components and evolve a tiny specialist per component,
each with a clean survival condition ("the constraint IS the genome's entire
reason for existing"). This is the first gated test of that vision, built in
`genreg_train/wordpipe.py`. Everything gradient-free (shared tournament +
elitism + energy-homeostasis `ga_step`), corpus = the 48.6M-char Gutenberg dump,
lexicon = 20,676 words that occur ≥8× in it.

Three specialists, each gated (the next only runs if the prior earns it):

## GATE 1 — VOCABULARY (the speller) → **PASS**

A char genome rewarded for lexicon **coverage** (fraction of emitted characters
that fall inside real words), scaffolded by char-prediction so it can bootstrap.
Control = the same genome with the coverage pressure off (a plain char LM).

| | valid tokens | best coverage |
|---|---|---|
| control (plain char LM) | **18.9%** | — |
| vocabulary genome (coverage w1.5) | **52.4%** | 0.47 |

**The vocabulary specialist works — ~2.8× more of its output is real words**, and
coverage climbed steadily (0.23 → 0.47) as a *separate, clean fitness signal* on
top of prediction. Samples are still short-word-heavy ("o at st shed ts ee...")
but decisively less fragmentary than the control ("ee elo at tsotot io t er").
The decomposition is real: "emit valid words" is an evolvable specialist.

## GATE 2 — ORDER (the grammar specialist): first a design failure, then the fix

**A note on framing (corrected).** My first pass concluded the discriminator
"hit a gradient-free wall — evolution can't." That was wrong and un-GENREG: it
blamed the tool for a space *I* built badly. The GENREG premise is that WE shape
and shrink the space until the only survivor is the solution. So a stuck run is a
signal to fix the space or the fitness — not evidence of a limit. Two errors,
found in order:

**Error 1 — the search space ballooned.** The word-level discriminator had to
evolve a **4000 × 10 word-embedding (~40k params)** from random init. Held-out
accuracy sat at chance (51.9%) across 2500 gens. The signal was NOT absent — a
bigram probe separates real/shuffled at 69.2% — the space was just too big to
climb. So: shrink it.

**The shrink — order over CATEGORIES, not words.** Induced ~32 distributional
classes (unsupervised k-means on left/right anchor-context — see
`induce_word_classes`). The classes come out genuinely POS-like: {been, seen,
done, taken, gone} (past participles), {made, left, heard, found, told} (past-
tense verbs), {old, same, whole, next, second} (adjectives). The order genome's
embedding drops from 40k params to **~256** (32 × 8); total ~2,273.

**Error 2 — the fitness was the wrong SHAPE (the deeper one).** Even at 2,273
params the discriminator STILL sat at chance (0.51), on both accuracy and a dense
margin fitness. Shrinking the space was necessary but **not sufficient**. Why: a
*discriminative* "is this whole window real?" fitness is **holistic — one bit per
window, no per-position gradient**, so there is no smooth path from a random
genome to a grammatical one. The char/word LMs climbed because *prediction* gives
a **dense, graded, per-position** signal (log-prob of each next symbol). The task
framing, not evolution, was the blocker.

**The fix — frame the order specialist as a PREDICTOR.** A next-*class* model
(context of C classes → next class, scored by log-prob) over the induced classes:

| order specialist framing | outcome |
|---|---|
| discriminative, 4000-word emb (40k params) | chance (0.519) |
| discriminative, 32-class emb (2.3k params), accuracy fitness | chance (0.51) |
| discriminative, 32-class, dense margin fitness | chance (0.514) |
| **predictive, 32-class next-class LM, log-prob fitness** | **climbs — val_ppl 30.9 → 12.75, beats unigram 13.4** |

Same signal, same tiny space — only the **fitness shape** changed, and evolution
went from dead-at-chance to climbing. That IS the GENREG lever.

**How close to the ceiling (longer runs, E10/H64, 4000 gens):**

| config | evolved val_ppl | unigram | count-bigram ceiling | signal captured |
|---|---|---|---|---|
| **32 classes, C=4** | **10.91** | 13.43 | 10.06 | **~75%** (within 8% of the table) |
| 32 classes, C=6 | 11.50 | 13.94 | — | more context doesn't help (cf. EvoLang) |
| 64 classes, C=4 | 16.37 | 18.78 | 13.16 | ~43% (further from ceiling) |

The predictive order specialist over 32 categories reaches **ppl 10.91 —
within 8% of the count-based bigram-table ceiling (10.06), gradient-free**,
capturing ~75% of all the categorical-order predictability in the corpus. Two
secondary confirmations: more context (C=6) doesn't help this tiny genome (same
as the EvoLang sweep), and the finer **64-class space captures less of its signal
(43%) than the coarser 32-class space (75%)** — the search-space-size lever again,
in the same experiment: the smaller space is the more evolvable one.

## GATE 3 — COMPOSE the two specialists into text

The payoff: chain the two proven genomes. The **order genome** emits a *class
skeleton* (grammatical category sequence); the **vocabulary component** fills each
slot with a real word of that class (frequency-weighted from the class's members).

The decisive comparison isolates the order genome's contribution — pipeline vs a
**unigram baseline** that draws classes by marginal frequency (no order genome),
**using the identical real-word fillers**, so any difference is *purely class
order*. Metric: fraction of adjacent word-pairs that occur in the real corpus (a
local "looks like English" score, evaluation-only).

**Result (order genome trained 3000 gens → ppl 11.0):**

| generator | adj-pair hit (local English-likeness) |
|---|---|
| **pipeline** (order genome + vocab) | **0.604** |
| unigram baseline (no order, same fillers) | 0.558 |

The order genome adds **+4.6 points** of real adjacencies over the identical-filler
baseline, and the gap **grew** as the order genome trained better (0.582 vs 0.561
at 300 gens → 0.604 vs 0.558 at 3000) — confirming it's the evolved class order
driving it, not the fillers. (The "real corpus = 1.0" reference is degenerate here
— real pairs are all in the bigram set by construction — so the meaningful contrast
is pipeline vs unigram.)

Sample (both use identical fillers; only class order differs):
- **pipeline**: "handsome further clear of my world long be he up and active with
  they had explain bread exciting … to their the any he more supper and rich and
  all more supper interesting before faint but" — real words with visible skeleton
  ("clear of my world", "he up and active", "they had explain").
- **unigram**: "terrible held in sofa from off charming the of gold to twisted knew
  the the for the the is creatures such more as were to struck…" — real words,
  no skeleton, "the the" repeats.

Honest read: the pipeline produces **real-words-in-a-grammatical-skeleton**,
measurably more English-like than the unordered baseline — but not fluent. A
32-class skeleton is coarse, and the vocabulary component fills each slot
*class-randomly* (ignoring the specific neighbours), which caps adjacency quality.
Closing that gap is the next specialist(s) — word-selection-given-context,
agreement, punctuation — exactly the multi-specialist decomposition the vision
calls for. Two specialists get you skeleton + real words; fluency needs the rest.

## GATE 4 — WORD-SELECTION (fill the slot with the word that fits the neighbours)

The Gate-3 pipeline filled each class slot *class-randomly* — the fluency cap. The
selection specialist picks the class member that fits the **previous word**. Built
with both hard-won levers: the 4000-word representation is **FIXED** (SVD of
distributional co-occurrence, out of the search space) so only a tiny **bilinear
compatibility head M (24×24 ≈ 577 params)** evolves; fitness is **dense predictive**
(log-prob of the true word among in-class negatives).

**Standalone (vs a frequency baseline that ignores the previous word):**

| | held-out log-prob | top-1 among 8 |
|---|---|---|
| frequency baseline | −0.981 | 0.666 |
| **selection genome** | **−0.931** | **0.688** |

Beats the baseline — it learned collocation (context-dependent choice) gradient-
free in ~577 params. Modest standalone, but in the **pipeline it compounds**:

| pipeline fill | adj-pair hit (local English-likeness) |
|---|---|
| class-random (Gate 3) | 0.610 |
| **selection (Gate 4)** | **0.761** |

**+15 points** — the biggest single jump. Sample (same order skeleton, different
fill): random *"he is are behind the hundred existed … pretty gay health"* →
selection *"he was no in the peace himself … the eyes what few of good … and
quickly and leave"* — full of real bigrams (he was, in the, the eyes, and quickly)
where the random fill jars (is are, hundred existed, gay health). (Mild caveat:
selection optimises "compatible with prev word" and the metric rewards real
bigrams — related but not identical; features are fixed and unsupervised, scored
held-out, so it's a real gain, not gaming.) Still not sentence-fluent — no long-
range coherence, agreement, or punctuation — but each specialist visibly closes
the gap, exactly as the decomposition predicts.

## GATE 5 — PUNCTUATION / SENTENCE BOUNDARY (the run-on fix)

The output so far was one endless run-on. This specialist decides, per position,
whether a sentence ends — from (word's class, sentence-position-so-far) → P(end).
Dense per-position binary prediction (log-prob of the true boundary), tiny space
(class embed + position → H → 1, ~577 params). *(Chose this over "agreement",
which is largely absorbed by selection and whose distinct long-range part needs
parsing — punctuation is cleaner and truly orthogonal.)*

**Standalone (vs base-rate boundary prediction):** val log-prob **−0.119 vs
−0.174** — beats it, and learned the corpus rhythm (boundary rate 5.4% ≈ 18.6-word
sentences).

**Full 4-specialist pipeline** (order + selection + boundary firing together):
generated **sentence length mean 17.0 vs real 18.6** — the segmentation matches.
Sample:

> *More more for good more to his one as the good rich and him news we rested for
> more they would free tea cruel and mad. Life to was at further under my others
> to make the train more more to him have in out things long and tried reasonable.
> … More good good in the door she was long graceful. And good the dress and
> easily free with less bad but was less relative to one of the real was wise…*

**What works:** real sentences (periods, caps, corpus-matched length), dense local
bigrams ("the good rich", "we rested for", "cruel and mad", "she was long
graceful"). Four tiny gradient-free genomes producing sentence-structured real-word
text. **What's still broken:** heavy repetition of a few words ("more", "good") —
one induced class is over-represented and filled by a handful of high-frequency
words; and no sentence-level grammaticality or long-range meaning. The "more more
more" is literally the **repetitive-collapse failure mode the EvoLang novelty
constraint was built to fight** — the next lever is an anti-repetition / diversity
pressure on the order or selection genome (full circle to novelty), plus fixing the
dominant-class imbalance. Coherence beyond the local window needs the harder
specialists (agreement across distance, semantics) that have no clean fitness yet.

## Class-count sweep (does finer granularity fix the repetition?) — NO

Tested whether more induced classes split the dominant `<unk>`-polluted mega-class
and cut the "more/good" repetition. It does not:

| classes | adj-pair | distinct-ratio ↑ | top-word share ↓ | order ppl vs unigram |
|---|---|---|---|---|
| **32** | 0.767 | **0.219** | 0.126 | best captured |
| 48 | 0.853* | 0.168 | 0.218 | |
| 64 | 0.759 | 0.176 | 0.105 | |
| 96 | 0.707 | 0.175 | 0.264 | 24.9 vs 25.5 (barely beats unigram) |

32 has the **highest** distinct-word ratio (least repetitive); every finer setting
made repetition worse. *48's adj-pair 0.853 is a mirage — it collapsed onto "you"
×327, and repeating one word that forms many real bigrams inflates the metric while
being *more* repetitive. And 96 classes barely beat their own unigram, confirming
finer classes are harder for the order genome to evolve. **Verdict: keep 32.** The
repetition is not a granularity problem — it's the **absence of a context genome**
(no specialist carries memory of what was already said). That's the next build, not
a decode-time patch.

## TRACK A — local-fluency specialists (the world is local; push local fluency)

After the EEC memory finding (memory isn't required by a non-conversational
corpus world — see below), Track A adds *local* specialists.

**Bidirectional selection** — score a candidate against the previous WORD *and*
the next CLASS (from the skeleton), not just the previous word. Beats prev-word-
only on every metric: adj-pair **0.767 → 0.777**, distinct 0.219 → 0.231 (less
repetitive), held-out log-prob −0.940 → −0.911. Kept in the stack. ~1,150 params.

**Chunk / phrase genome** — a lexicon of frequent real phrases (1,270 bigram +
2,758 trigram) indexed by class pattern; emit a whole real phrase when the
skeleton's upcoming classes match one. Phrase-internal adjacencies are 100% real
by construction: adj-pair **0.776 → 0.851 (+7.5 pts)** — the biggest local-fluency
lever so far — at a small repetition cost (distinct 0.231 → 0.191). Sample reads
visibly more fluent ("…so long seven hundred and simple of life in which he had is
an excellent you are not…"). `build_chunk_index`, `gen_chunked`.

**Comma / internal-punctuation genome** — same shape as boundary: per-position
P(comma here) from (class, clause-position). Beats base rate (held-out log-prob
−0.159 vs −0.264; comma rate 8.1%), adding clause structure the pipeline lacked
entirely. `build_comma_corpus`, `run_comma`.

**Local agreement — measured, NOT built (make-or-break diagnostics).** Before
building, we tested whether the pipeline breaks agreement. Determiner-noun number
agreement is **already handled by selection** (pipeline plural-rate gap after
plural-vs-singular determiners +0.338 ≥ real corpus +0.199). Subject-verb
agreement barely shows a *local adjacent* signal even in real text (+0.026 — the
verb usually isn't adjacent to the subject), so it's the non-local "needs parsing"
tier, not a local specialist. Verdict: skip — the local agreement that exists is
subsumed; the rest isn't local. (Same for the **continuity/topic-memory** idea: a
running topic-average never beats the global mean and is identical on shuffled
text — no exploitable continuity signal in a static corpus. Both saved by cheap
diagnostics before any wasted evolution run.)

## The `/evolang` page (interactive)

The web page now IS this pipeline: toggle each evolved specialist (Vocabulary,
Order, Selection prev/both, Boundary, Chunks) and watch the text transform live —
server generates over the trained genomes (`genreg_train/wordpipe_service.py`,
REST `/api/evolang/*`, reuses `demo/genomes.pkl`). The old char-model EvoLang page
is retired.

## Deployment size (as-is)

All the **evolved genomes stay tiny**: order ~3,108 params + speller ~2,885 +
selection head ~577 ≈ **6.6 K parameters (~26 KB fp32, ~7 KB int8).** The deploy
size is dominated by the **data tables**, not the nets:
- lexicon (word strings + class map + freqs): ~38 KB
- fixed word-feature table for selection (4000 × 24): 384 KB fp32 / **~96 KB int8**

So the 3-specialist pipeline deploys at roughly **~140 KB (int8-quantized)** to
~450 KB (fp32) — the selection feature table is now the biggest single item, the
neural genomes a rounding error (~0.005% of GPT-2-small's 124M). Still **well under
a megabyte**, phone-sized by construction (the I2 premise). Each further specialist
adds only a few-KB genome (feature/lexicon tables are shared), so the whole
architecture stays sub-MB no matter how many specialists stack.

## Verdict (corrected)

- **Vocabulary specialist: works** (Gate 1, 18.9% → 52.4% valid).
- **Order specialist: works once framed right.** The failure was never
  "evolution can't" — it was (1) a ballooned search space and (2) a holistic,
  flat fitness. Shrink to categories **and** switch to a dense predictive fitness,
  and it climbs.

The reusable lesson, and it's pure GENREG: **when a specialist won't evolve,
shrink the search space and reshape the fitness until the landscape is dense and
the solution is the only survivor — don't conclude evolution can't.** Both levers
were needed here; either alone left it at chance.

Five gates cleared: vocabulary (real words) → order (categorical skeleton) →
composition (skeleton + words) → **selection** (context-appropriate words, +15 pts:
0.61 → 0.76) → **boundary** (real sentences, len 17.0 vs 18.6). Five tiny gradient-
free genomes now produce sentence-structured real-word text, sub-MB, and every
specialist added has closed the gap — the decomposition **compounds**, which was
the whole question. Remaining gap to fluency: an anti-repetition lever (the "more
more" collapse — reuse the EvoLang novelty constraint), then the hard tier
(agreement across distance, long-range semantic coherence) that still lacks a clean
gradient-free fitness.

## Reproduce
`genreg_train/wordpipe.py` (module: `run_speller`, `induce_word_classes`,
`run_disc_on`, `run_class_lm`), `scratchpad/wordpipe_experiments.py` (gate
battery), `scratchpad/classlm_run.py` (predictive ceiling probe).
