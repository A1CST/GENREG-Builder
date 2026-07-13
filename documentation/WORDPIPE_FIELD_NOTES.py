"""Generates documentation/WORDPIPE_FIELD_NOTES.pdf -- the complete findings
record for the WordPipe /evolang pipeline before it's archived for a
ground-up rebuild. Run once locally: python documentation/WORDPIPE_FIELD_NOTES.py
"""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, HRFlowable, ListFlowable, ListItem)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "WORDPIPE_FIELD_NOTES.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], fontSize=20, spaceAfter=14,
                          textColor=colors.HexColor("#1a2a3a")))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], fontSize=14, spaceBefore=18,
                          spaceAfter=8, textColor=colors.HexColor("#2a4a6a")))
styles.add(ParagraphStyle("H3c", parent=styles["Heading3"], fontSize=11.5, spaceBefore=12,
                          spaceAfter=6, textColor=colors.HexColor("#3a5a7a")))
styles.add(ParagraphStyle("Bodyc", parent=styles["BodyText"], fontSize=9.6, leading=13.5,
                          spaceAfter=8, alignment=TA_LEFT))
styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.3, leading=11.5,
                          textColor=colors.HexColor("#444444")))
styles.add(ParagraphStyle("Verdict", parent=styles["BodyText"], fontSize=9.6, leading=13.5,
                          spaceAfter=8, backColor=colors.HexColor("#f0f4f8"),
                          borderPadding=6, borderColor=colors.HexColor("#c0d0e0"),
                          borderWidth=0.5))
styles.add(ParagraphStyle("Title2", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER,
                          textColor=colors.HexColor("#555555")))
styles.add(ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=8, leading=10.5))
styles.add(ParagraphStyle("CellHead", parent=styles["BodyText"], fontSize=8.3, leading=10.5,
                          textColor=colors.white, fontName="Helvetica-Bold"))

STATUS_COLOR = {
    "shipped": colors.HexColor("#1e7d32"), "cut": colors.HexColor("#b71c1c"),
    "partial": colors.HexColor("#e08a00"), "experimental": colors.HexColor("#1565c0"),
    "defer": colors.HexColor("#6a4a9a"),
}


def cell(text, bold=False):
    style = styles["CellHead"] if bold else styles["Cell"]
    return Paragraph(text, style)


def table(rows, colWidths, header=True):
    data = [[cell(c, bold=header and i == 0) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=colWidths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4a6a")))
        style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#f4f7fa")]))
    t.setStyle(TableStyle(style))
    return t


def h1(t): return Paragraph(t, styles["H1c"])
def h2(t): return Paragraph(t, styles["H2c"])
def h3(t): return Paragraph(t, styles["H3c"])
def body(t): return Paragraph(t, styles["Bodyc"])
def small(t): return Paragraph(t, styles["Small"])
def verdict(t): return Paragraph(t, styles["Verdict"])
def rule(): return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#c8c8c8"),
                              spaceBefore=6, spaceAfter=10)
def bullets(items):
    return ListFlowable([ListItem(body(i), bulletColor=colors.HexColor("#2a4a6a")) for i in items],
                        bulletType="bullet", leftIndent=14)


story = []

# ---- Title page ----
story.append(Spacer(1, 1.6 * inch))
story.append(Paragraph("WordPipe Field Notes", styles["H1c"]))
story.append(Paragraph("The gradient-free specialist-genome language pipeline (/evolang)",
                       styles["Title2"]))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph("Complete findings record, archived 2026-07-08, before a ground-up rebuild.",
                       styles["Title2"]))
story.append(Spacer(1, 0.15 * inch))
story.append(Paragraph("GENREG project — genreg_train/wordpipe*.py and dependents",
                       styles["Title2"]))
story.append(PageBreak())

# ---- Executive summary ----
story.append(h1("Executive Summary"))
story.append(body(
    "WordPipe is the thesis that a language model can be built entirely gradient-free: not one "
    "network trained on a loss, but a pipeline of tiny evolved specialist genomes, each with a "
    "single falsifiable job, composed. The core architectural principle (the \"GA abstraction "
    "thesis\"): evolution does not discover an embedding space by searching it — that space is "
    "too large. Instead the space is BUILT from corpus statistics (induced word classes via "
    "k-means, SVD distributional features, mined relation pairs), and evolution learns one tiny "
    "relationship inside that pre-built space at a time. The features are the environment; "
    "evolution is the organism navigating it."))
story.append(body(
    "Over three corpus generations (Gutenberg novels → Wikipedia → Wikipedia + Cornell Movie "
    "Dialogs) and roughly 45 distinct genomes attempted, the pipeline reliably solved local, "
    "structural, single-word-pair problems (grammar rhythm, agreement, adjacent-word semantic "
    "fit, sentence-boundary placement) but never solved GLOBAL sentence coherence or clause "
    "completeness. That is the central, honest finding of this entire body of work, and it is "
    "the reason for archiving: the current architecture has been pushed hard on every axis "
    "available to it (better vocabulary, decomposed observability, meaning-first ordering, "
    "punctuation-first generation, backward growth, stateful obligation tracking) and the "
    "fluency ceiling has not moved. Whatever comes next needs a structurally different idea, "
    "not another genome bolted onto the same skeleton."))
story.append(verdict(
    "<b>What survived, in one sentence:</b> 13 shipped genomes (vocabulary, class-order skeleton, "
    "bidirectional word selection, boundary/comma placement, agreement, alternation, semantic "
    "adjacency, no-repeat, opener, closer, chunk lookup) plus a validated-but-experimental layer "
    "(sentence-type, pronominalization, three relation genomes, a full structural decomposition "
    "for observability, and a working intent-first/backward-generation mode) — none of it adds "
    "up to a sentence a person would write."))
story.append(PageBreak())

# ---- Architecture ----
story.append(h1("1. Architecture & Core Thesis"))
story.append(h2("1.1 The pipeline stages"))
story.append(body(
    "Generation was organized into pipeline STAGES, each a real point in the data flow: "
    "<b>Skeleton</b> (Order genome — next grammatical class from the last 4 classes), "
    "<b>Fill</b> (word selection into each class slot), <b>Boundary</b> (sentence/comma "
    "punctuation), plus three later, larger additions: <b>Content</b> (meaning-first — pick "
    "content words before structure), <b>Intent</b> (punctuation-first — pick the discourse "
    "skeleton before any words, grow backward), <b>Revision</b> (best-of-N whole-sentence "
    "scoring, post-hoc), and a partially-built <b>Passage</b> stage (cross-sentence, only "
    "pronominalization landed)."))
story.append(h2("1.2 The layer taxonomy"))
story.append(body(
    "Genomes were also tagged by LAYER — a metaphorical, not code-enforced, grouping: "
    "<b>Structural</b> (form: grammar, position, rhythm — never touches meaning), "
    "<b>Semantic</b> (meaning: distributional word-feature spaces built from corpus statistics), "
    "<b>Abstraction</b> (relations composed on top of the semantic space — the newest, "
    "least-tested tier). A genuine architectural lesson surfaced repeatedly: many attempted "
    "genomes asked a LAYER-3 question (\"do these two relations agree?\") using LAYER-2 machinery "
    "(raw distributional offsets) — Analogy is the clean example, chance-level "
    "no matter how long it trained, because A:B::C:D is a relationship between the OUTPUTS of "
    "relation genomes (hypernym, meronym, synonym/antonym), not a pattern in raw SVD offset "
    "vectors."))
story.append(h2("1.3 GA training machinery (shared, reused throughout)"))
story.append(bullets([
    "<b>BilinearPop / train_pairwise</b> — genome = a small bilinear head M; score(a,b) = "
    "f(a)ᵀ M f(b); trained as a corrupted-pair discriminator against hard negatives.",
    "<b>WindowPop / train_windowed</b> — genome = class embedding + one-hidden-layer validity "
    "net over a K-class window; tells a real window from one corrupted at a chosen slot.",
    "<b>UnaryPop / train_unary</b> — genome = a linear scorer over per-word features; used for "
    "positional questions (is this word a plausible opener/closer/question-starter).",
    "<b>OrderPop / run_class_lm</b> — genome = embedding + positional weight + tanh hidden layer "
    "+ softmax; a tiny autoregressive class-sequence (or, reused this session, punctuation-"
    "sequence) predictor.",
    "All trained by the same tournament-selection + elitism + self-adaptive-mutation + energy-"
    "homeostasis GA step (<code>wp.ga_step</code>) — no gradients anywhere in the model.",
]))
story.append(PageBreak())

# ---- Corpus history ----
story.append(h1("2. Corpus History — Three Generations"))
t = table([
    ["Corpus", "Size", "Why swapped", "What it fixed / didn't"],
    ["Gutenberg novels\n(project/EEC-main/\nengine/corpus.txt)",
     "49MB", "Original corpus.",
     "Baseline. 19th-century vocabulary (\"thou\", \"shalt\") baked into every generated sample."],
    ["Wikipedia\n(corpora/wikipedia/\nwiki_corpus.txt)",
     "316MB",
     "\"Modern corpus\" directive — kill the archaic vocabulary.",
     "Fixed vocabulary completely (zero archaic words in samples). Did NOT fix fluency — grammar "
     "was at least as broken, arguably worse (heavy \"of the X of the Y\" chains). Also revealed "
     "Wikipedia's register barely uses exclamation marks the way emotional language does — every "
     "probe word (wow/amazing/hooray) was out-of-vocabulary."],
    ["Combined\n(corpora/combined/\ncombined_corpus.txt)",
     "421MB\n(24.4% dialogue)",
     "Wikipedia + Cornell Movie Dialogs (17.1MB real dialogue, repeated 6x for real influence) — "
     "real questions, real exclamations, real turn-taking.",
     "Exclaim-affinity training samples rose 2,499 → 133,321 (53x). Corpus exclaim-rate in mark "
     "mining rose 0.11% → 2.34%; question-rate 0.07% → 6.69%. Modern/dialogue-specific words "
     "(\"youtube\", \"gimme\") now genuinely appear in output. Fluency STILL not fixed — this was "
     "explicitly not attempted as a fluency fix, only a vocabulary/intent-signal fix, paired with "
     "the intent-first architecture change (below) so it wasn't a repeat of the Wikipedia "
     "mistake."],
], colWidths=[1.15 * inch, 0.65 * inch, 1.7 * inch, 2.6 * inch])
story.append(t)
story.append(Spacer(1, 10))
story.append(verdict(
    "<b>Lesson, stated by the user and confirmed by data:</b> \"this is a living model, changing "
    "one thing isn't going to magically fix everything.\" A corpus swap changes vocabulary and "
    "available signal; it does not change what the Order/Selection genomes structurally can or "
    "cannot represent. Every corpus swap in this project's history fixed exactly what it targeted "
    "and nothing else."))
story.append(PageBreak())

# ---- Genome inventory ----
story.append(h1("3. Genome Inventory"))
story.append(h2("3.1 Shipped (13 genomes, default ON)"))
story.append(small("~9,554 params + chunk lookup. Deploy ~140-450 KB. All PASS on the original "
                   "generation-time battery (targeted metric + adj-hit/distinct guardrails)."))
t = table([
    ["Genome", "One thing", "Result"],
    ["Vocabulary", "Emit real English words", "19%→52% valid"],
    ["Order", "Next grammatical class from last 4 classes", "ppl 10.91, 75% of ceiling"],
    ["Selection", "Word that fits the previous word", "+15 pts adj-hit"],
    ["Bidirectional Selection", "...and the next class too", "adj 0.777, distinct 0.231"],
    ["Boundary", "Where sentences end", "learned 18.6-word rhythm"],
    ["Commas", "Where commas go inside sentences", "beats base rate"],
    ["Agreement", "Subject/verb, modal/aux agreement", "12/12 minimal pairs"],
    ["Alternation", "Content/function word rhythm", "func-func pairs -34%"],
    ["Semantic (adjacency)", "Adjacent content words co-occur", "content-adjacency 33%→40%"],
    ["No-repeat", "Don't reuse a recent content word", "repetition 2.2%→0%"],
    ["Opener", "First word is a plausible starter", "bad-openings 17%→8%"],
    ["Closer", "Sentences end on noun/verb, not \"of/the\"", "bad-endings 65%→33%, rate-preserving"],
    ["Chunks", "Emit whole real phrases on class match", "+7.5 pts adj-hit"],
], colWidths=[1.4 * inch, 2.6 * inch, 2.1 * inch])
story.append(t)

story.append(h2("3.2 Experimental (wired, OFF by default)"))
story.append(small("\"Tested\" for round-1 genomes originally meant probe + a light spot-check "
                   "(20-30 samples, 1-2 metrics) — NOT the full guardrail battery. When the full "
                   "battery (adj-hit, distinct, dangling-rate, mean-length, 60 samples) was "
                   "finally run, EVERY ONE of the round-1 experimental genomes regressed at least "
                   "one guardrail. None graduated to shipped. This was an important process "
                   "lesson: a spot-check systematically under-catches side effects."))
t = table([
    ["Genome", "Toggle", "Full-battery result"],
    ["Hypernym / Meronym / Synonym-Antonym", "hyper/mero/synant",
     "10/10, 9/10, 11/14 probes respectively — standalone-valid, never battery-tested in "
     "generation."],
    ["Sentence type", "sent_type",
     "Dangling-ending rate 20.8%→24.8%. Real question-rate effect confirmed, real fluency cost."],
    ["Sentence length plan", "lenplan",
     "Dangling-ending rate 20.8%→26.0% — worse than sent_type. (Corrects an earlier \"practically "
     "inert\" call that only checked sentence length, which didn't move, and missed this.)"],
    ["Pronominalization", "pronominal",
     "Dangling-ending rate 20.8%→24.0%. \"it\"-substitution effect real (0.77%→1.11% of words) "
     "but has a fluency cost the earlier spot-check didn't catch."],
    ["Revision (Best-of-N)", "generate_revision()",
     "Mean sentence length collapsed 14.7→7.8 words (-47%) — the whole-sentence scorer's "
     "penalties accumulate with word count, biasing toward short \"safe\" sentences."],
], colWidths=[1.5 * inch, 0.9 * inch, 3.7 * inch])
story.append(t)
story.append(PageBreak())

story.append(h2("3.3 Cut, with the specific reason (selected — the full list runs to ~30 genomes)"))
t = table([
    ["Genome", "Why cut"],
    ["Clause template / Verb argument / Pronoun reference",
     "\"Class-level redundancy wall\" — Order's 4-class context already subsumes any class-level "
     "windowed-validity signal. No headroom: baseline already emits ~0.2% rare class-trigrams."],
    ["Preposition completion / Determiner binding",
     "Redundant with Alternation for the illegal cases; determiner-binding had a strong signal "
     "(0.776) and real effect (orphan-det 11→0) but adj-hit dropped at EVERY gamma — 14 "
     "function-type features too coarse to pick the RIGHT noun."],
    ["Tense consistency", "Diffuse signal, subsumed by Agreement's local finiteness (54.9→56.4, "
     "no real effect)."],
    ["Sentiment (monolithic)", "Mining inverted — \"war\" scored +7.85, \"joy\" scored -7.38. "
     "Seed-propagation drifted onto generic frequent words, not true polarity."],
    ["Polysemy", "val_acc looked strong (0.88) but measured the wrong thing — Wikipedia's "
     "register skews classically polysemous words to one dominant sense, defeating the "
     "neighbor-spread proxy."],
    ["Analogy", "Chance-level (0.48-0.53) — a layer-3 question (do two pairs share a relation?) "
     "asked with layer-2 raw-offset machinery. Needs a genome over relation-genome OUTPUTS."],
    ["Register", "Quote-span mining conflated narration-about-feelings with real dialogue "
     "markers; incoherent probes."],
    ["Clause count", "Spearman correlation between opener score and TRUE empirical compound-rate "
     "= -0.009 — genuinely unlearnable from the opening word alone, not a compound question."],
    ["Sentence coherence / Theme consistency", "val_acc 0.525 / 0.531, barely above chance — "
     "mean-pooling several content words' features into one centroid likely destroys the "
     "information a linear head needs, not proof coherence itself is unlearnable."],
    ["Definiteness", "Blocked before training — \"a\" is not in the 4000-word pipeline "
     "vocabulary at all (min_len=2 drops single-letter words); a vocabulary-construction gap, "
     "not a learnability gap."],
    ["Transitivity", "Redundant — collapses onto the already-shipped Closer genome or the "
     "already-cut Verb-argument ambiguity; no POS tags to separate the two."],
    ["Open-obligation / Clause-obligation trackers", "Both a real, correctly-targeted idea "
     "(track unclosed prep/det or relative-clause structure, suppress boundary probability) "
     "whose implementation cost more than it fixed — flat suppression forced long run-ons; a "
     "decay variant showed no differentiation, traced to a suspected bug in the hand-rolled test "
     "harness (see §6)."],
], colWidths=[1.7 * inch, 4.4 * inch])
story.append(t)
story.append(PageBreak())

# ---- Key architectural experiments ----
story.append(h1("4. Major Architecture Experiments"))

story.append(h2("4.1 Meaning-first generation"))
story.append(body(
    "User diagnosis: the pipeline was structure-first the whole time — Order picks a class "
    "skeleton blind, meaning gets bolted on afterward as a rerank. Flipped: <code>_select_content()"
    "</code> picks 3-5 mutually-related content words FIRST (via Hypernym/Meronym/Synonym-Antonym/"
    "Semantic, stochastic sampling), then the SAME Order/Fill genomes place each reserved word "
    "into the first matching class slot instead of running selection there. No new training."))
story.append(verdict(
    "<b>Verified:</b> selected content sets score +0.58 higher on relatedness than random sets "
    "(t=8.25, 40 samples). Placement rate 66.5% (a third of chosen words never find a matching "
    "slot and get dropped). <b>Does not fix word-level fluency</b> — only WHICH words fill "
    "content slots changed, not how well the surrounding function words glue them together."))

story.append(h2("4.2 Structural genome decomposition (full observability)"))
story.append(body(
    "After four grammar-fix hypotheses failed (see §5), the direction changed from \"find the "
    "fix\" to \"build the ability to trace WHY output looks a specific way.\" Every shipped "
    "structural genome was assessed for a genuine internal compound question:"))
t = table([
    ["Parent", "Split into", "Why compound"],
    ["Selection", "Sel-backward / Sel-forward / Sel-frequency",
     "Already 3 separate terms (ML/MR/beta) summed into one score — split needed no retraining."],
    ["Order", "Order-bigram (K=1) / Order-context (K=4)",
     "Tight local transition vs wider 4-class context. Order-bigram trained (beats unigram "
     "baseline) but left unwired — its artifact is a full class-LM, not a bilinear rerank."],
    ["Alternation", "Altern-rhythm / Altern-func-chain",
     "Coarse content/function rhythm (val_acc 0.53, barely above chance) vs specific function→"
     "function legality (0.68, much stronger) — most of Alternation's real power lives in the "
     "subtype detail, not the coarse rhythm."],
    ["Agreement", "Agree-modal / Agree-number", "0.75 / 0.68 respectively — two genuinely "
     "distinct grammatical phenomena."],
    ["Semantic", "Sem-adjacent / Sem-window", "0.68 / 0.54 — loose topical fit is a genuinely "
     "harder signal than tight collocation."],
], colWidths=[1.1 * inch, 1.9 * inch, 3.1 * inch])
story.append(t)
story.append(verdict(
    "<b>Real, measured generation-time effect</b> (the only experiment this session with a clean "
    "multi-metric win): with decomposition active vs. disabled, func-func-adjacency -32%, "
    "distinct-ratio +14%, dangling-ending rate -67%. Visually confirmed in the actual text — the "
    "\"of the X of the Y\" pattern was measurably and visibly reduced. Traded for a different, "
    "smaller residual tic (\"whom\"/\"including\" used oddly as glue words)."))
story.append(PageBreak())

story.append(h2("4.3 Intent-first generation (punctuation as anchor, backward growth)"))
story.append(body(
    "User's idea: the punctuation mark IS the intent, chosen before any word exists — grammar "
    "grows backward to serve it instead of forward hoping to land on a valid ending. Implemented "
    "with zero new algorithm: Order and Selection are generic autoregressive predictors over "
    "whatever sequence they're trained on, so training them on the corpus read BACKWARD (a "
    "cache-reversal trick — same <code>wp.run_class_lm</code>/<code>wp.run_biselection</code>, "
    "same word→class mapping) gives real backward Order+Selection for free. "
    "<code>intent_punct.py</code> (new) is a tiny autoregressive model over the 6-symbol mark "
    "alphabet <code>{. , ; : ! ?}</code>, mined directly from the corpus with zero labeling. "
    "<code>sent_type_exclaim.py</code> generalizes the existing question/statement genome to "
    "exclaim via a second binary genome (same decompose-into-binary pattern as §4.2)."))
story.append(body(
    "<code>Service.generate_intent_first()</code>: generates the punctuation sequence for the "
    "whole response first, then grows each word-span BACKWARD from its mark toward the previous "
    "one, with the mark's TYPE biasing what grows toward it."))
story.append(verdict(
    "<b>Verified working, real output, corrected once (see §6):</b> real questions (\"who?\", "
    "\"whom?\") and a real exclamation (\"who!\") appeared tied to genuine discourse marks — the "
    "first time this session real intent-carrying punctuation appeared in generation at all. "
    "Does not fix fluency; sentences remain word salad. This changes WHAT anchors generation, "
    "not the underlying word-to-word mechanics."))

story.append(h2("4.4 Revision stage & crystallize"))
story.append(body(
    "<code>generate_revision()</code>: for each sentence slot, generate several candidates with "
    "the unchanged pipeline, score with <code>_sentence_score()</code> (composite of already-"
    "evolved champions), keep the best. Mechanically verified (correctly rank-orders 8 "
    "candidates) but has a severe length bias (mean length -47% in the full battery — see §3.2)."))
story.append(body(
    "<code>crystallize</code> (optional flag on intent-first): a forward polish sweep after the "
    "backward pass, re-picking each word against its LEFT neighbor too, using the shipped "
    "forward Selection genome. Real result once two implementation bugs were fixed: "
    "func-func-adjacency -0.026 (better), dangling-rate +0.068 (worse). Mixed — stays off by "
    "default; no visible readability gain in the actual samples for a second full pass's worth "
    "of compute."))
story.append(PageBreak())

# ---- Grammar investigation ----
story.append(h1("5. Grammar / Fluency Investigation — Every Hypothesis Tried"))
story.append(body(
    "The central open problem: generated text reads as word salad regardless of which genomes "
    "are active. Every hypothesis below was tested with real measurements, not assumed."))
t = table([
    ["#", "Hypothesis", "Verdict"],
    ["1", "Stub-article density in Wikipedia is the cause",
     "Disproven — template phrases (\"is a town\") appear in under 0.5% of articles; top "
     "bigrams look like ordinary English frequency stats, not a Wikipedia artifact."],
    ["2", "Alternation strength is too low",
     "Disproven, harmful — sweeping gamma up to 6x default made every metric WORSE "
     "(func-func-adjacency +19%, distinct -41%, dangling +200%). The chain is the DEGENERATE "
     "LIMIT of forced alternation, not a sign of too little."],
    ["3", "Suppress literal bigram repeats (\"of the\" recurring)",
     "Narrow win, no real quality gain — cut literal recurrence 62% but func-func-adjacency and "
     "distinct barely moved; dangling-rate rose 66%."],
    ["4", "Selection's frequency-bias term is the culprit",
     "Classic diversity-vs-plausibility trade-off, not a fix — zeroing the frequency term raised "
     "distinct +45% but adj-hit collapsed 23 points."],
    ["5", "Structural decomposition (§4.2)",
     "The one real, clean multi-metric win this session — but addresses function-word chaining, "
     "not sentence-level coherence."],
    ["6", "Clause-obligation tracker, flat gamma",
     "Small real effect (never-closed-relative-rate 13.1%→12.5%, plateaus immediately) at a "
     "real cost (dangling worse) — cut."],
    ["7", "Clause-obligation tracker, decay variant",
     "Inconclusive — all 9 gamma/tau combinations produced byte-identical results, traced to a "
     "suspected independent bug in the hand-rolled test harness (110+ word \"sentences\" even at "
     "baseline). Never resolved before archiving — see §7 recommendations."],
], colWidths=[0.3 * inch, 2.0 * inch, 3.9 * inch])
story.append(t)
story.append(verdict(
    "<b>User's conclusion, confirmed by every data point above:</b> no single knob fixes this, "
    "because Order and Selection are both fundamentally LOCAL — next class from the last 4 "
    "classes, word score against immediate neighbors — neither has any model of what the "
    "sentence has already said. This is the actual reason for archiving: seven independent "
    "hypotheses, seven honest non-wins, converging on the same structural ceiling."))
story.append(PageBreak())

# ---- Process lessons / bugs ----
story.append(h1("6. Process Lessons (worth carrying into the rebuild)"))
story.append(h3("6.1 Spot-checks systematically under-catch side effects"))
story.append(body(
    "Every round-1 experimental genome passed a lightweight spot-check (1-2 metrics, 20-30 "
    "samples) and then FAILED the full guardrail battery (4 metrics, 60 samples) once it was "
    "finally run. The fix adopted: run the full battery as a required last step before calling "
    "anything validated, not probe + one spot-check metric."))
story.append(h3("6.2 Silent no-op bugs are the most dangerous class"))
story.append(body(
    "Three separate incidents this session where code executed without crashing but silently "
    "did nothing: (a) <code>self.champs</code> empty on the I2 primary because "
    "<code>demo/genomes.pkl</code> is not in the deploy whitelist and a verification script "
    "never overrode the cache path — every \"X in self.champs\" check just quietly evaluated "
    "False, and reranks were inert without disclosure; (b) the crystallize forward-polish "
    "branch's own guard depended on the same empty-champs condition, so it silently never ran, "
    "producing byte-identical A/B output that was initially misreported as \"verified, no "
    "difference\"; (c) the decay-based clause-obligation sweep produced identical results across "
    "9 parameter combinations, later traced to a suspected bug in the underlying hand-rolled "
    "generation loop (110+ word \"sentences\" even at baseline) rather than the decay logic "
    "itself. <b>Lesson: byte-identical A/B results are the tell for a silent no-op, not evidence "
    "of \"no effect\" — always suspicious, always worth a diagnostic counter before trusting.</b>"))
story.append(h3("6.3 Hand-rolled parallel reimplementations of generate() are a liability"))
story.append(body(
    "Every standalone experiment script that reimplemented a simplified version of the "
    "generation loop (rather than adding a toggle to the shipped, proven "
    "<code>Service.generate()</code>) introduced a NEW implementation bug: missing Closer/Opener "
    "biases silently changing baseline behavior, cache-path mismatches, list-length "
    "misalignment when class sequences skip the reserved <unk> class. Recommendation for the "
    "rebuild: prefer adding toggles to one proven generation function over spinning up new "
    "parallel copies for every experiment."))
story.append(h3("6.4 Pushing to the I2 primary while a job runs kills it"))
story.append(body(
    "<code>push_to_primary.py</code> always triggers a node restart, which terminates any "
    "in-flight job (by design, to avoid orphaned processes — a deliberate earlier fix). This "
    "cost two jobs this session before the discipline was established: confirm the job queue is "
    "empty before every push."))
story.append(h3("6.5 The class-level / word-level redundancy wall (2026-07-07 finding, still true)"))
story.append(body(
    "The single most load-bearing lesson from the ORIGINAL battery, and the one that predicted "
    "most of this session's cuts: genomes that ask a validity question at the CLASS level "
    "(clause template, verb argument, pronoun reference) are always subsumed by Order's 4-class "
    "context — there's no headroom, Order already knows which class sequences are common. "
    "Genomes that ask a validity question at the word/function-type level (determiner-binding, "
    "preposition-completion) are always subsumed by Alternation for the same reason. <b>The "
    "genomes that survive are the ones enforcing a constraint Order/Selection/Alternation "
    "STRUCTURALLY cannot see</b> — semantic co-occurrence (content identity, not class) and "
    "exact-word recency (state, not a static rule)."))
story.append(PageBreak())

# ---- What to keep / archive ----
story.append(h1("7. What to Keep vs. Archive"))
story.append(h2("7.1 Keep — datasets and durable infrastructure"))
t = table([
    ["Path", "What it is", "Why keep it"],
    ["corpora/wikipedia/wiki_corpus.txt", "316MB Wikipedia text dump",
     "Real, cleaned, ready-to-tokenize modern-vocabulary corpus."],
    ["corpora/wikipedia/wiki_feats.npz", "30K-vocab, 128-dim SVD distributional features",
     "Expensive to rebuild; validated with real nearest-neighbor sanity checks (king→queen/"
     "prince/emperor)."],
    ["corpora/combined/combined_corpus.txt", "421MB Wikipedia + Cornell Movie Dialogs blend",
     "The best available corpus for real intent-carrying punctuation; took ~94 min to retrain "
     "everything against."],
    ["project/conversational/cornell movie-dialogs corpus/", "Raw Cornell dialogue dataset",
     "Source data for the combined corpus; genuinely modern, real turn-taking."],
    ["genreg_train/wordpipe.py", "Core GA training machinery (OrderPop, BilinearPop, "
     "WindowPop, UnaryPop, ga_step, gen_class_seq, boundary_prob, etc.)",
     "Generic, reusable, well-tested infrastructure independent of any specific genome."],
    ["genreg_train/genelib.py", "Shared pairwise/windowed/unary training scaffolding",
     "Same — infrastructure, not a specific model."],
    ["I2 job-dispatch system (i2_node.py + push_to_primary.py + run_job.py)",
     "Signed, whitelisted remote training dispatch to the I2 primary node",
     "Proven, reusable compute infrastructure independent of WordPipe specifically."],
], colWidths=[1.9 * inch, 2.1 * inch, 2.1 * inch])
story.append(t)
story.append(Spacer(1, 10))
story.append(h2("7.2 Archive — the pipeline itself"))
story.append(body(
    "Everything under the WordPipe genome umbrella: <code>genreg_train/wordpipe_service.py</code> "
    "(the Service class and every generate* method), all ~45 individual genome modules "
    "(agreement.py, altern.py, sem_compat.py, sent_type.py, intent_punct.py, and every cut "
    "module alongside them), the decomposition modules, the backward-generation experiment "
    "scripts, <code>templates/evolang*.html</code>, <code>static/evolang*.js</code>, and the "
    "<code>/evolang</code>-family routes in <code>app.py</code>. Archive as a snapshot (branch "
    "or dated directory), not delete — every cut reason and every real number in this document "
    "traces back to a specific file worth having on hand during the rebuild."))
story.append(PageBreak())

# ---- Recommendations ----
story.append(h1("8. Recommendations for the Rebuild"))
story.append(bullets([
    "<b>Start from the architecture, not the genome list.</b> The single clearest lesson: "
    "structure-first generation with meaning/intent bolted on afterward has been tried in every "
    "combination (forward, meaning-first, intent-first, backward, crystallized) and none reached "
    "sentence-level coherence. Consider whether the class-induction step itself (32 k-means "
    "clusters over left/right anchor context) is the real ceiling — it has no notion of "
    "subject/object/clause boundary by construction, and every genome that tried to add that "
    "notion at the class level got redundancy-walled by Order.",
    "<b>Rebuild the guardrail battery FIRST, before any genome ships default-ON again.</b> The "
    "spot-check-then-battery pattern cost real time and one public overclaim this session (§6.1, "
    "§6.2). Make the full battery (adj-hit, distinct, dangling-rate, mean-length, 60+ samples) "
    "step zero of the genome-acceptance protocol, not an afterthought.",
    "<b>One proven generate() function, toggles not forks.</b> Every hand-rolled parallel "
    "reimplementation introduced a new bug (§6.3). If the rebuild keeps a similar composed-"
    "genome architecture, invest early in making the real generation function flexible enough "
    "that experiments are toggles inside it, not new files.",
    "<b>Keep the decomposition discipline.</b> The full observability directive (§4.2) was this "
    "session's one clean, multi-metric win. Whatever architecture comes next, build in "
    "traceability (which sub-signal produced which word) from day one rather than retrofitting "
    "it after the fact.",
    "<b>Resolve the decay clause-obligation harness bug before concluding anything about "
    "clause-completeness.</b> The last open thread (§5, #7) was never actually resolved — the "
    "conclusion \"neither obligation-tracker mechanism works\" rests partly on a suspect test "
    "harness. Worth a clean re-test on the new architecture rather than importing the "
    "conclusion unverified.",
    "<b>Definiteness's vocabulary gap is a real, separate, cheap fix</b> (§3.3) — raising "
    "<code>min_len</code> in the lexicon builder to include \"a\"/\"I\" is independent of the "
    "architecture question and worth doing early in the rebuild regardless of what replaces "
    "the pipeline.",
]))
story.append(PageBreak())
story.append(h1("Appendix: Key File Index"))
t = table([
    ["File", "Role"],
    ["genomes.txt", "The full, continuously-updated running log this document was distilled "
     "from — every genome, every verdict, every number, in chronological order."],
    ["static/evolang_layers.js", "The live flow-map data — source of truth for CURRENT status "
     "of every genome (shipped/experimental/cut/deferred), exportable as JSON."],
    ["CHANGELOG.md", "Dated entries for every change this session, cross-referenced with "
     "genomes.txt for the reasoning behind each."],
    ["genreg_train/battery_round1.py", "The full guardrail battery script (adj-hit, distinct, "
     "dangling-rate, mean-length) — needs re-running on any future corpus before trusting a "
     "genome's default state."],
    ["genreg_train/run_retrain_combined.py", "The most recent full-pipeline retrain recipe (23 "
     "genomes) — the template for retraining everything after any future corpus change."],
    ["corpora/combined/build_combined_corpus.py", "How the combined corpus was built — reusable "
     "if more dialogue-style data needs mixing in later."],
], colWidths=[2.3 * inch, 3.8 * inch])
story.append(t)

doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        title="WordPipe Field Notes")
doc.build(story)
print(f"saved {OUT}")
