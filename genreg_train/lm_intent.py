"""LM rebuild — Genome group "punctuation": INTENT RECOGNITION, decomposed.

The single 6-way classifier (period vs exclaim vs question vs comma vs
semicolon vs colon) collapsed to mostly guessing two classes (recall:
question 63.7%, semicolon 67.5%, but exclaim 0.15%, colon 2.5%) — a 6-way
softmax head has to arbitrate all six at once, and the wildly different base
rates (periods 5.7M vs colons 99K in this corpus) meant two easy, frequent-
enough classes ate the capacity. Split into 5 small BINARY genomes, each with
one clean yes/no survival condition, all sub-labeled under the "punctuation"
group — the same decompose-into-specialists pattern that was the one clean
win in the archived /evolang pipeline (see archive/evolang_v1/README.md):

  punct_end        — does this end the thought (. ! ?) or continue it (, ; :)?
  punct_question   — among enders, is it a question (?) vs statement/exclaim?
  punct_exclaim    — among enders, is it an exclaim (!) vs statement/question?
  punct_semicolon  — among continuers, is it a semicolon (;) vs comma/colon?
  punct_colon      — among continuers, is it a colon (:) vs comma/semicolon?

Each binary problem is trained AND evaluated class-balanced by construction
(50/50 draw), so the earlier train/eval-distribution mismatch bug (see
CHANGELOG.md 2026-07-09) can't recur the same way. recognize() composes the
5 heads hierarchically to still answer "which of the 6 marks fits here?"

Deliberately self-contained — no dependency on the archived evolang/wordpipe
engine. The GA itself (tournament selection + elitism + self-adaptive
mutation, numpy-only, no gradients) is unchanged — that machinery was never
the problem.
"""
import collections
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(ROOT, "corpora", "combined", "combined_corpus.txt")

MARKS = [".", "!", "?", ",", ";", ":"]
MARK_ID = {m: i for i, m in enumerate(MARKS)}
MARK_INTENT = {
    ".": "end / statement complete",
    "!": "end / exclaim",
    "?": "end / question",
    ",": "more to say / light pause",
    ";": "more to say / related clause",
    ":": "more to say / elaboration follows",
}
N_CLASSES = len(MARKS)

_TOKEN_RE = re.compile(r"[A-Za-z']+|[.!?,;:]")

GROUP = "punctuation"

# Each split: positive marks -> label 1, negative marks -> label 0. "scope"
# restricts which examples are even IN this genome's problem (None = all).
SPLITS = [
    {"key": "punct_end", "group": GROUP,
     "desc": "does this end the thought (. ! ?) or continue it (, ; :)?",
     "positive": [".", "!", "?"], "positive_name": "end",
     "negative": [",", ";", ":"], "negative_name": "continue",
     "scope": None},
    {"key": "punct_question", "group": GROUP,
     "desc": "among enders, is it a question (?) vs statement/exclaim?",
     "positive": ["?"], "positive_name": "question",
     "negative": [".", "!"], "negative_name": "statement/exclaim",
     "scope": [".", "!", "?"]},
    {"key": "punct_exclaim", "group": GROUP,
     "desc": "among enders, is it an exclaim (!) vs statement/question?",
     "positive": ["!"], "positive_name": "exclaim",
     "negative": [".", "?"], "negative_name": "statement/question",
     "scope": [".", "!", "?"]},
    {"key": "punct_semicolon", "group": GROUP,
     "desc": "among continuers, is it a semicolon (;) vs comma/colon?",
     "positive": [";"], "positive_name": "semicolon",
     "negative": [",", ":"], "negative_name": "comma/colon",
     "scope": [",", ";", ":"]},
    {"key": "punct_colon", "group": GROUP,
     "desc": "among continuers, is it a colon (:) vs comma/semicolon?",
     "positive": [":"], "positive_name": "colon",
     "negative": [",", ";"], "negative_name": "comma/semicolon",
     "scope": [",", ";", ":"]},
]
SPLIT_BY_KEY = {s["key"]: s for s in SPLITS}

# --------------------------------------------------------------------------
# "opener" group — the mirror-image genome: instead of reading the words
# BEFORE a mark to recognize its intent, read the sentence's FIRST word to
# recognize (confirm) what intent the sentence is headed for. "who/what/
# where/how/can/don't..." at the very start is a strong, cheap signal for
# where the sentence is going to end up. ctx_k=1 by design — literally just
# the opening word, not a window — the same ClassifierPop/train_classifier
# code works unchanged since mean-pooling one embedding is a no-op.
# --------------------------------------------------------------------------
OPENER_GROUP = "opener"
END_MARK_IDS = {MARK_ID["."], MARK_ID["!"], MARK_ID["?"]}

OPENER_SPLITS = [
    {"key": "opener_question", "group": OPENER_GROUP,
     "desc": "does the first word predict the sentence will end in a question (?) "
            "vs statement/exclaim?",
     "positive": ["?"], "positive_name": "question",
     "negative": [".", "!"], "negative_name": "statement/exclaim",
     "scope": [".", "!", "?"]},
    {"key": "opener_exclaim", "group": OPENER_GROUP,
     "desc": "does the first word predict the sentence will end in an exclaim (!) "
            "vs statement/question?",
     "positive": ["!"], "positive_name": "exclaim",
     "negative": [".", "?"], "negative_name": "statement/question",
     "scope": [".", "!", "?"]},
]
ALL_SPLITS = SPLITS + OPENER_SPLITS
SPLIT_BY_KEY.update({s["key"]: s for s in OPENER_SPLITS})


def mine_opener_examples(tokens, stoi):
    """One example per SENTENCE: the first word's id (ctx of width 1),
    labeled with which end mark (. ! ?) that sentence terminates with.
    Sentences are delimited by . ! ? only — commas/semicolons/colons are
    mid-sentence and don't reset what counts as "the first word.\""""
    ctx, labels = [], []
    expecting_start = True
    first_word = None
    for tok in tokens:
        mid = MARK_ID.get(tok)
        if mid is not None:
            if mid in END_MARK_IDS:
                if first_word is not None:
                    ctx.append([first_word])
                    labels.append(mid)
                first_word = None
                expecting_start = True
            continue
        if expecting_start:
            first_word = stoi.get(tok, 0)
            expecting_start = False
    return np.asarray(ctx, dtype=np.int32), np.asarray(labels, dtype=np.int32)


# --------------------------------------------------------------------------
# "length" group — the growth/stop decision for hangman-style generation.
# Given a PARTIAL sentence prefix, is this already a complete, natural-
# sounding sentence, or does it need to keep growing? Unlike the punctuation/
# opener genomes, this has no fixed-position "scope" filter — it's evaluated
# at an arbitrary partial length, so it needs an explicit length feature
# (mean-pooling the context window alone can't tell a 3-word prefix from a
# 10-word one apart).
# --------------------------------------------------------------------------
LENGTH_GROUP = "length"
LENGTH_MAX_LEN_NORM = 30.0

LENGTH_SPLIT = {"key": "length_continue", "group": LENGTH_GROUP,
                "desc": "given a partial sentence, is it already complete, or does it "
                       "need to keep growing?",
                "positive_name": "end", "negative_name": "continue"}


def mine_length_examples(tokens, stoi, ctx_k=6, max_prefixes=4,
                         max_len_norm=LENGTH_MAX_LEN_NORM, seed=0):
    """One positive ('end') example per sentence — its FULL length — plus a
    bounded number of negative ('continue') examples per sentence (shorter
    prefixes of that SAME sentence). Bounding prefixes per sentence (instead
    of mining every possible prefix) keeps this roughly linear in corpus
    size instead of quadratic in sentence length — the exact kind of mistake
    already caught once this session (see mine_examples's docstring).

    Context = the last ctx_k words of the prefix (left-padded), plus an
    explicit normalized length scalar (prefix word count / max_len_norm,
    clipped to 1.0)."""
    rng = np.random.default_rng(seed)
    ctx, lengths, labels = [], [], []
    sentence_words = []
    for tok in tokens:
        mid = MARK_ID.get(tok)
        if mid is not None:
            if mid in END_MARK_IDS:
                K = len(sentence_words)
                if K >= 1:
                    window = ([0] * ctx_k + sentence_words)[-ctx_k:]
                    ctx.append(window)
                    lengths.append(K)
                    labels.append(1)
                    if K >= 2:
                        n_neg = min(max_prefixes, K - 1)
                        neg_lens = rng.choice(np.arange(1, K), size=n_neg, replace=False)
                        for prefix_len in neg_lens:
                            prefix = sentence_words[:int(prefix_len)]
                            window = ([0] * ctx_k + prefix)[-ctx_k:]
                            ctx.append(window)
                            lengths.append(int(prefix_len))
                            labels.append(0)
                sentence_words = []
            continue
        sentence_words.append(stoi.get(tok, 0))
    ctx = np.asarray(ctx, dtype=np.int32)
    extra = (np.asarray(lengths, dtype=np.float32) / max_len_norm)
    extra = np.clip(extra, 0.0, 1.0).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int32)
    return ctx, extra, labels


# --------------------------------------------------------------------------
# Corpus + vocabulary (self-contained — words are ONLY used as context
# features here, never as an output; these genomes never emit a word)
# --------------------------------------------------------------------------
def load_text(path=None, max_chars=None):
    path = path or CORPUS_PATH
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read(max_chars) if max_chars else fh.read()


def tokenize(text):
    """Words (lowercased) and the 6 marks as ONE stream, in original order."""
    return _TOKEN_RE.findall(text.lower())


def build_vocab(tokens, vocab_n=4000):
    counts = collections.Counter(t for t in tokens if t.isalpha())
    top = [w for w, _ in counts.most_common(vocab_n)]
    vocab = ["<unk>"] + top
    stoi = {w: i for i, w in enumerate(vocab)}
    return vocab, stoi


def mine_examples(tokens, stoi, ctx_k=6):
    """One example per mark occurrence: the up-to-ctx_k word-ids immediately
    before it (left-padded with <unk>=0), labeled with which of the 6 marks
    it is (0..5, order = MARKS). `recent` is a fixed-size ring (deque
    maxlen=ctx_k) — NOT the full word history — so this stays O(1) per token
    instead of O(corpus length) per mark."""
    ctx, labels = [], []
    recent = collections.deque([0] * ctx_k, maxlen=ctx_k)
    for tok in tokens:
        mid = MARK_ID.get(tok)
        if mid is not None:
            ctx.append(list(recent))
            labels.append(mid)
        else:
            recent.append(stoi.get(tok, 0))
    return np.asarray(ctx, dtype=np.int32), np.asarray(labels, dtype=np.int32)


def prepare_split(ctx, mark_labels, split):
    """Filter to `scope` marks (or all 6) and relabel to binary: 1 if the
    mark is in `positive`, 0 if in `negative`."""
    pos_ids = {MARK_ID[m] for m in split["positive"]}
    neg_ids = {MARK_ID[m] for m in split["negative"]}
    scope_ids = pos_ids | neg_ids if split["scope"] is None else \
        {MARK_ID[m] for m in split["scope"]}
    keep = np.isin(mark_labels, list(scope_ids))
    sub_ctx = ctx[keep]
    sub_mark = mark_labels[keep]
    bin_labels = np.isin(sub_mark, list(pos_ids)).astype(np.int32)
    return sub_ctx, bin_labels


def class_balanced_batch(ctx, labels, n_per_class, rng, n_classes, extra=None):
    """Draw an equal number of examples per class so the genome can't win by
    just always guessing the majority class. If `extra` (e.g. a length
    feature) is given, it's subsampled in lockstep with ctx/labels."""
    idx = []
    for c in range(n_classes):
        pool = np.flatnonzero(labels == c)
        if len(pool) == 0:
            continue
        take = rng.choice(pool, size=min(n_per_class, len(pool)),
                          replace=len(pool) < n_per_class)
        idx.append(take)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    if extra is None:
        return ctx[idx], labels[idx]
    return ctx[idx], labels[idx], extra[idx]


# --------------------------------------------------------------------------
# The genome: word-embedding -> mean-pool -> tanh hidden -> class logits.
# Generic over n_out so the same code trains the binary punctuation splits
# (n_out=2) or, if ever needed again, a wider classifier.
# --------------------------------------------------------------------------
class ClassifierPop:
    def __init__(self, pop, V, D, H, ctx_k, n_out, seed=0, extra_dim=0):
        self.pop, self.V, self.D, self.H, self.ctx_k, self.n_out = pop, V, D, H, ctx_k, n_out
        self.extra_dim = extra_dim
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(0, 0.3, (pop, V, D)).astype(np.float32)
        self.W1 = rng.normal(0, 0.3, (pop, D + extra_dim, H)).astype(np.float32)
        self.b1 = np.zeros((pop, H), dtype=np.float32)
        self.W2 = rng.normal(0, 0.3, (pop, H, n_out)).astype(np.float32)
        self.b2 = np.zeros((pop, n_out), dtype=np.float32)
        self.sigma = np.full(pop, 0.15, dtype=np.float32)

    def forward(self, i, ctx_ids, extra=None):
        e = self.emb[i][ctx_ids]                      # (batch, ctx_k, D)
        pooled = e.mean(axis=1)                        # (batch, D)
        if self.extra_dim:
            pooled = np.concatenate([pooled, extra.astype(np.float32)], axis=1)
        h = np.tanh(pooled @ self.W1[i] + self.b1[i])
        return h @ self.W2[i] + self.b2[i]

    def accuracy(self, i, ctx_ids, labels, extra=None):
        pred = self.forward(i, ctx_ids, extra).argmax(axis=1)
        return float((pred == labels).mean())

    def soft_fitness(self, i, ctx_ids, labels, extra=None):
        """Mean log-prob of the TRUE class — GENREG_RULES §IV.1: soft fitness
        only. Accuracy (argmax == target) is a step function with no climbing
        gradient for evolution; log-prob rewards every incremental sharpening
        of the right answer's probability, even before it flips the argmax."""
        logits = self.forward(i, ctx_ids, extra)
        logits = logits - logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(logp[np.arange(len(labels)), labels].mean())

    def balanced_accuracy(self, i, ctx_ids, labels, extra=None):
        """Mean per-class recall — the fair metric whenever classes aren't
        50/50 in the raw holdout. Plain accuracy on a skewed distribution
        actively PUNISHES learning to recognize the minority class."""
        pred = self.forward(i, ctx_ids, extra).argmax(axis=1)
        recalls = []
        for c in range(self.n_out):
            mask = labels == c
            if mask.sum() == 0:
                continue
            recalls.append(float((pred[mask] == c).mean()))
        return float(np.mean(recalls)) if recalls else 0.0

    def mutate(self, i, rng):
        s = self.sigma[i]
        self.emb[i] += rng.normal(0, s, self.emb[i].shape).astype(np.float32)
        self.W1[i] += rng.normal(0, s, self.W1[i].shape).astype(np.float32)
        self.b1[i] += rng.normal(0, s, self.b1[i].shape).astype(np.float32)
        self.W2[i] += rng.normal(0, s, self.W2[i].shape).astype(np.float32)
        self.b2[i] += rng.normal(0, s, self.b2[i].shape).astype(np.float32)
        self.sigma[i] = np.clip(s * float(rng.lognormal(0, 0.2)), 0.01, 1.0)

    def select_into(self, order):
        self.emb = self.emb[order].copy()
        self.W1 = self.W1[order].copy()
        self.b1 = self.b1[order].copy()
        self.W2 = self.W2[order].copy()
        self.b2 = self.b2[order].copy()
        self.sigma = self.sigma[order].copy()

    def export(self, i):
        return {"emb": self.emb[i].copy(), "W1": self.W1[i].copy(), "b1": self.b1[i].copy(),
                "W2": self.W2[i].copy(), "b2": self.b2[i].copy(), "ctx_k": self.ctx_k,
                "D": self.D, "H": self.H, "n_out": self.n_out, "extra_dim": self.extra_dim}


# Energy homeostasis (GENREG_RULES §III, mandatory): energy decides who is
# even eligible to survive/reproduce, independent of this generation's
# fitness rank. Rank-percentile delta (robust to the different fitness
# scales across genome types); equilibrium energy = 2 * percentile, so a
# genome persistently in the bottom ~10% decays below the floor and is
# culled even if a lucky batch spikes its fitness once. Fresh offspring
# start at 1.0. Target starved band: 3-15%/gen (§III) — logged by the
# training loops as `starved`.
ENERGY_DECAY = 0.9
ENERGY_GAIN = 0.2       # rank-percentile delta scale; equilibrium e = 2*pct
ENERGY_FLOOR = 0.2
E_MAX = 1.5


def ga_step(pop_obj, fits, rng, elite_frac=0.1, tourn_k=4):
    """Tournament selection + elitism + self-adaptive mutation + energy
    homeostasis, in place. Starved genomes (energy < floor) are removed from
    the survivor pool regardless of current fitness."""
    pop = pop_obj.pop
    if not hasattr(pop_obj, "energy"):
        pop_obj.energy = np.full(pop, 1.0, dtype=np.float32)
    ranks = np.empty(pop, dtype=np.float64)
    ranks[np.argsort(fits)] = np.arange(pop)
    pct = ranks / max(1, pop - 1)
    pop_obj.energy = np.clip(pop_obj.energy * ENERGY_DECAY + ENERGY_GAIN * pct,
                             0.0, E_MAX).astype(np.float32)
    starved = pop_obj.energy < ENERGY_FLOOR
    pop_obj.last_starved = int(starved.sum())
    alive = np.flatnonzero(~starved)
    if len(alive) < max(2, tourn_k):          # pathological all-starved guard
        alive = np.argsort(-fits)[:max(2, tourn_k)]
    alive_by_fit = alive[np.argsort(-fits[alive])]
    n_elite = max(1, int(pop * elite_frac))
    new_order = alive_by_fit[:n_elite].tolist()
    new_energy = [float(pop_obj.energy[i]) for i in new_order]
    while len(new_order) < pop:
        cand = rng.choice(alive, size=min(tourn_k, len(alive)), replace=False)
        new_order.append(int(cand[np.argmax(fits[cand])]))
        new_energy.append(1.0)                 # offspring start fresh
    pop_obj.select_into(new_order)
    pop_obj.energy = np.asarray(new_energy, dtype=np.float32)
    for i in range(n_elite, pop):
        pop_obj.mutate(i, rng)


def train_classifier(ctx, labels, n_out, gens=250, pop=120, D=16, H=24, ctx_k=6,
                     n_per_class=64, seed=0, log=print, extra=None, extra_dim=0):
    """Full training loop for one classifier (binary or wider). Returns
    (best_genome_dict, holdout_balanced_acc, holdout_raw_acc, confusion).

    Champion selection and the headline metric both use BALANCED accuracy
    (mean per-class recall) on held-out data, matching the class-balanced
    training batches — plain accuracy on the natural distribution is still
    reported for interpretability but must not drive which genome wins.

    `extra`/`extra_dim`: an optional extra feature array (e.g. normalized
    sentence-prefix length for `length_continue`, which mean-pooling alone
    can't see) concatenated onto the pooled context embedding."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    n_holdout = max(400, n // 20)
    perm = rng.permutation(n)
    hold_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    ctx_tr, lab_tr = ctx[train_idx], labels[train_idx]
    ctx_ho, lab_ho = ctx[hold_idx], labels[hold_idx]
    extra_tr = extra_ho = None
    if extra is not None:
        extra_tr, extra_ho = extra[train_idx], extra[hold_idx]

    V = int(ctx.max()) + 1
    popn = ClassifierPop(pop, V, D, H, ctx_k, n_out, seed=seed, extra_dim=extra_dim)
    best_bal_acc = -1.0
    best_export = None
    for g in range(gens):
        if extra is not None:
            bctx, blab, bextra = class_balanced_batch(ctx_tr, lab_tr, n_per_class, rng, n_out, extra_tr)
        else:
            bctx, blab = class_balanced_batch(ctx_tr, lab_tr, n_per_class, rng, n_out)
            bextra = None
        # soft fitness (mean log-prob of true class), GENREG_RULES §IV.1 —
        # accuracy is still what we REPORT, but log-prob is what selection
        # climbs; argmax fitness is a step function evolution can't ascend.
        fits = np.array([popn.soft_fitness(i, bctx, blab, bextra) for i in range(pop)])
        champ = int(np.argmax(fits))
        if g % 25 == 0 or g == gens - 1:
            ho_bal = popn.balanced_accuracy(champ, ctx_ho, lab_ho, extra_ho)
            ho_raw = popn.accuracy(champ, ctx_ho, lab_ho, extra_ho)
            starved = getattr(popn, "last_starved", 0)
            log(f"    gen {g:4d}  soft-fit={fits[champ]:.4f}  starved={starved}  "
               f"holdout-balanced-acc={ho_bal:.3f}  holdout-raw-acc={ho_raw:.3f}")
            if ho_bal > best_bal_acc:
                best_bal_acc = ho_bal
                best_export = popn.export(champ)
        ga_step(popn, fits, rng)

    final = ClassifierPop(1, V, D, H, ctx_k, n_out, seed=seed, extra_dim=extra_dim)
    final.emb[0], final.W1[0], final.b1[0] = best_export["emb"], best_export["W1"], best_export["b1"]
    final.W2[0], final.b2[0] = best_export["W2"], best_export["b2"]
    pred = final.forward(0, ctx_ho, extra_ho).argmax(axis=1)
    cm = np.zeros((n_out, n_out), dtype=int)
    for t, p in zip(lab_ho, pred):
        cm[t, p] += 1
    raw_acc = float((pred == lab_ho).mean())
    return best_export, best_bal_acc, raw_acc, cm


def forward_export(export, ctx_ids, extra=None):
    """Run a saved genome export on raw context word-ids -> class logits."""
    e = export["emb"][ctx_ids]
    pooled = e.mean(axis=1) if e.ndim == 3 else e.mean(axis=0, keepdims=True)
    if export.get("extra_dim"):
        pooled = np.concatenate([pooled, extra.astype(np.float32)], axis=1)
    h = np.tanh(pooled @ export["W1"] + export["b1"])
    return h @ export["W2"] + export["b2"]


# --------------------------------------------------------------------------
# "fill" group — the word-choice genome for hangman-style generation. Given
# the LEFT-filled and RIGHT-filled words around a blank, score how well a
# candidate word fits. Deliberately NOT a softmax classifier over the whole
# ~4000-word vocabulary — word frequency is even more Zipfian than mark
# frequency was, and a monolithic multi-way head would reintroduce exactly
# the collapse the punctuation group hit (see CHANGELOG.md 2026-07-09).
# Instead: a contrastive/discriminator design — score the TRUE word from the
# corpus against random corrupted negatives (real-vs-shuffled discrimination,
# this project's original thesis). That's structurally a binary "does this
# word beat a random one" decision regardless of vocabulary size, so it
# can't collapse toward a handful of frequent words the same way.
# --------------------------------------------------------------------------
FILL_GROUP = "fill"
FILL_SPLIT = {"key": "fill_word", "group": FILL_GROUP,
             "desc": "given the words around a blank, does the TRUE word score higher "
                    "than a random corrupted candidate?",
             "positive_name": "true-word-wins", "negative_name": "n/a"}


def mine_fill_examples(tokens, stoi, ctx_k=6, n_samples=1_000_000, n_neg=5, seed=0):
    """Build a flat word-only id array (marks stripped) in ONE O(n) pass,
    then sample N positions and vectorized-slice fixed-width left/right
    windows around each — never rebuilds a growing list per example (the
    exact quadratic-mining mistake already caught and fixed once this
    session, see mine_examples's docstring).

    Returns (left_ctx, right_ctx, candidates) where candidates[:, 0] is
    always the TRUE word and candidates[:, 1:] are n_neg random negatives
    (drawn from real corpus positions, so they're frequency-natural, not
    uniform-over-vocab)."""
    rng = np.random.default_rng(seed)
    words = np.asarray([stoi.get(t, 0) for t in tokens if MARK_ID.get(t) is None],
                       dtype=np.int32)
    n = len(words)
    n_samples = min(n_samples, max(0, n - 2 * ctx_k))
    positions = rng.integers(ctx_k, n - ctx_k, size=n_samples)

    left = np.stack([words[p - ctx_k:p] for p in positions])
    right = np.stack([words[p + 1:p + 1 + ctx_k] for p in positions])
    true_word = words[positions]
    neg_idx = rng.integers(0, n, size=(n_samples, n_neg))
    negatives = words[neg_idx]
    candidates = np.concatenate([true_word[:, None], negatives], axis=1)
    return left, right, candidates


class FillPop:
    """genome = word embedding (V x D) + a query projection that combines
    mean-pooled LEFT and RIGHT context into a D-dim query vector. A
    candidate word's score = query . embedding(candidate) (bilinear form,
    no separate classifier head — works for any vocab size unchanged)."""
    def __init__(self, pop, V, D, ctx_k, seed=0):
        self.pop, self.V, self.D, self.ctx_k = pop, V, D, ctx_k
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(0, 0.3, (pop, V, D)).astype(np.float32)
        self.Wq = rng.normal(0, 0.3, (pop, 2 * D, D)).astype(np.float32)
        self.bq = np.zeros((pop, D), dtype=np.float32)
        self.sigma = np.full(pop, 0.15, dtype=np.float32)

    def query(self, i, left_ids, right_ids):
        el = self.emb[i][left_ids].mean(axis=1)     # (batch, D)
        er = self.emb[i][right_ids].mean(axis=1)     # (batch, D)
        combo = np.concatenate([el, er], axis=1)      # (batch, 2D)
        return np.tanh(combo @ self.Wq[i] + self.bq[i])   # (batch, D)

    def score(self, i, q, cand_ids):
        w = self.emb[i][cand_ids]                     # (batch, K, D)
        return np.einsum("bd,bkd->bk", q, w)           # (batch, K)

    def accuracy(self, i, left_ids, right_ids, cand_ids):
        """Fraction of examples where the TRUE word (candidates[:,0]) scores
        highest among itself + its negatives."""
        q = self.query(i, left_ids, right_ids)
        s = self.score(i, q, cand_ids)
        return float((s.argmax(axis=1) == 0).mean())

    def soft_fitness(self, i, left_ids, right_ids, cand_ids):
        """Mean log-prob of the TRUE candidate under a softmax over the
        candidate set — soft fitness per GENREG_RULES §IV.1."""
        q = self.query(i, left_ids, right_ids)
        s = self.score(i, q, cand_ids)
        s = s - s.max(axis=1, keepdims=True)
        logp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
        return float(logp[:, 0].mean())

    def mutate(self, i, rng):
        s = self.sigma[i]
        self.emb[i] += rng.normal(0, s, self.emb[i].shape).astype(np.float32)
        self.Wq[i] += rng.normal(0, s, self.Wq[i].shape).astype(np.float32)
        self.bq[i] += rng.normal(0, s, self.bq[i].shape).astype(np.float32)
        self.sigma[i] = np.clip(s * float(rng.lognormal(0, 0.2)), 0.01, 1.0)

    def select_into(self, order):
        self.emb = self.emb[order].copy()
        self.Wq = self.Wq[order].copy()
        self.bq = self.bq[order].copy()
        self.sigma = self.sigma[order].copy()

    def export(self, i):
        return {"emb": self.emb[i].copy(), "Wq": self.Wq[i].copy(), "bq": self.bq[i].copy(),
                "ctx_k": self.ctx_k, "D": self.D}


def train_fill(left, right, cand, gens=250, pop=120, D=24, ctx_k=6, batch_size=512, seed=0, log=print):
    """Full training loop for the fill_word contrastive genome. Returns
    (best_genome_dict, holdout_accuracy) — accuracy = fraction of held-out
    examples where the true word outranks its corrupted negatives (no
    balanced/raw split needed here, the contrastive setup is inherently
    class-balanced regardless of raw word frequency)."""
    rng = np.random.default_rng(seed)
    n = len(cand)
    n_holdout = max(400, n // 20)
    perm = rng.permutation(n)
    hold_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    left_tr, right_tr, cand_tr = left[train_idx], right[train_idx], cand[train_idx]
    left_ho, right_ho, cand_ho = left[hold_idx], right[hold_idx], cand[hold_idx]

    V = int(max(left.max(), right.max(), cand.max())) + 1
    popn = FillPop(pop, V, D, ctx_k, seed=seed)
    best_acc = -1.0
    best_export = None
    for g in range(gens):
        bidx = rng.integers(0, len(cand_tr), size=min(batch_size, len(cand_tr)))
        bleft, bright, bcand = left_tr[bidx], right_tr[bidx], cand_tr[bidx]
        # soft fitness (mean log-prob of the true candidate), §IV.1
        fits = np.array([popn.soft_fitness(i, bleft, bright, bcand) for i in range(pop)])
        champ = int(np.argmax(fits))
        if g % 25 == 0 or g == gens - 1:
            ho_acc = popn.accuracy(champ, left_ho, right_ho, cand_ho)
            starved = getattr(popn, "last_starved", 0)
            log(f"    gen {g:4d}  soft-fit={fits[champ]:.4f}  starved={starved}  holdout-acc={ho_acc:.3f}")
            if ho_acc > best_acc:
                best_acc = ho_acc
                best_export = popn.export(champ)
        ga_step(popn, fits, rng)
    return best_export, best_acc


def fill_score_export(export, left_ids, right_ids, cand_ids):
    """Run a saved fill_word genome export: score each candidate word for
    one blank. Returns (K,) scores, higher = better fit."""
    emb = export["emb"]
    el = emb[left_ids].mean(axis=0)
    er = emb[right_ids].mean(axis=0)
    combo = np.concatenate([el, er])
    q = np.tanh(combo @ export["Wq"] + export["bq"])
    w = emb[cand_ids]                                  # (K, D)
    return w @ q


# --------------------------------------------------------------------------
# "next" group — intent-conditioned NEXT-WORD prediction. Revives the
# proven core idea from the archived pipeline's Selection/Bidirectional-
# Selection genomes (bilinear word-pair scoring, contrastive real-vs-
# corrupted training — this project's original thesis) but rebuilt fresh
# with two real differences from fill_word:
#   1. Properly AUTOREGRESSIVE — left context only, matching how generate()
#      actually runs. fill_word was trained bidirectionally (left+right)
#      but used with zero-padded right context at inference, a real
#      train/inference mismatch; this genome never sees a right context at
#      all, so there's nothing to mismatch.
#   2. INTENT-conditioned — the sentence's target end-mark (period/exclaim/
#      question) is fed in as an explicit small embedding alongside the
#      context, a signal fill_word never had. "What word comes next" is a
#      different question when the sentence is headed for "?" vs "!".
# Negative sampling: HARD negatives, not random corpus positions. The first
# pass (random negatives, same as fill_word) produced a near-tie with
# fill_word (24.25% vs 24.08%) despite fixing the autoregressive mismatch
# and adding intent-conditioning — an honest negative result pointing
# straight at random negatives being too easy to beat via a handful of
# generically "safe" embedding directions, regardless of architecture. Fix:
# for a training example whose immediately-preceding word is W, draw
# negatives from words that ACTUALLY FOLLOWED W somewhere else in the
# corpus — genuinely confusable candidates (other things that plausibly
# come after "the", not a random word like "purple" or "quickly"), not
# noise the genome can shortcut past.
# --------------------------------------------------------------------------
NEXT_GROUP = "next"
NEXT_SPLIT = {"key": "next_word", "group": NEXT_GROUP,
             "desc": "given the words before this one AND the sentence's target intent, "
                    "does the TRUE next word score higher than a HARD corrupted candidate "
                    "(a word that also followed the same preceding word elsewhere)?",
             "positive_name": "true-word-wins", "negative_name": "n/a"}
N_INTENTS = 3   # period / exclaim / question — MARKS[0:3], the only sentence-final marks


def _build_followers_index(words, bucket_cap, rng):
    """words[i] -> words[i+1] pairs, grouped by the FIRST word into a
    per-vocab-word array of words that followed it somewhere in the corpus
    (capped per bucket so a common word like "the" doesn't blow up memory).
    Built with one argsort over the whole corpus, not a python-level
    per-pair loop — the same vectorized-first discipline that avoided the
    earlier quadratic-mining mistake."""
    V = int(words.max()) + 1
    prev_ids, next_ids = words[:-1], words[1:]
    order = np.argsort(prev_ids, kind="stable")
    sorted_prev, sorted_next = prev_ids[order], next_ids[order]
    boundaries = np.searchsorted(sorted_prev, np.arange(V + 1))
    buckets = []
    for v in range(V):
        seg = sorted_next[boundaries[v]:boundaries[v + 1]]
        if len(seg) > bucket_cap:
            seg = rng.choice(seg, size=bucket_cap, replace=False)
        buckets.append(seg)
    return buckets


def mine_next_word_examples(tokens, stoi, ctx_k=6, n_samples=1_000_000, n_neg=5,
                            bucket_cap=2000, seed=0):
    """Build a flat word-only id array (marks stripped) PLUS a parallel
    intent array (which end mark . ! ? that word's SENTENCE terminates
    with) in one O(n) pass, then sample N positions and vectorized-slice
    fixed-width LEFT-only windows around each (no right context — the
    autoregressive fix over mine_fill_examples). Sentence end-mark is only
    known once the mark is hit, so ranges are backfilled with a vectorized
    slice assign per sentence.

    Negatives are HARD: drawn from _build_followers_index's per-preceding-
    word buckets (see that function's docstring), falling back to a random
    corpus word only for the rare preceding word with too few followers to
    fill n_neg."""
    rng = np.random.default_rng(seed)
    words = []
    sentence_ranges = []   # (start_idx, end_idx_exclusive, mark_id)
    start = 0
    for tok in tokens:
        mid = MARK_ID.get(tok)
        if mid is not None:
            if mid in END_MARK_IDS:
                if len(words) > start:
                    sentence_ranges.append((start, len(words), mid))
                start = len(words)
            continue
        words.append(stoi.get(tok, 0))
    words = np.asarray(words, dtype=np.int32)
    intents = np.zeros(len(words), dtype=np.int32)   # default 0 (period) for any trailing partial sentence
    for s, e, mid in sentence_ranges:
        intents[s:e] = mid

    n = len(words)
    n_samples = min(n_samples, max(0, n - ctx_k))
    positions = rng.integers(ctx_k, n, size=n_samples)

    left = np.stack([words[p - ctx_k:p] for p in positions])
    true_word = words[positions]
    intent_feat = intents[positions]

    followers = _build_followers_index(words, bucket_cap, rng)
    preceding_word = left[:, -1]
    negatives = np.zeros((n_samples, n_neg), dtype=np.int32)
    for row in range(n_samples):
        pool = followers[preceding_word[row]]
        if len(pool) >= n_neg:
            negatives[row] = rng.choice(pool, size=n_neg, replace=False)
        elif len(pool) > 0:
            negatives[row] = rng.choice(pool, size=n_neg, replace=True)
        else:
            negatives[row] = rng.integers(0, n, size=n_neg)   # rare fallback

    candidates = np.concatenate([true_word[:, None], negatives], axis=1)
    return left, intent_feat, candidates


class NextWordPop:
    """genome = word embedding (V x D) + a small intent embedding (3 x Di)
    + a query projection combining mean-pooled LEFT context and the intent
    embedding into a D-dim query. Score(candidate) = query . embedding
    (candidate) — same bilinear contrastive scoring as FillPop, but
    properly autoregressive and intent-conditioned (see module docstring)."""
    def __init__(self, pop, V, D, ctx_k, n_intents=N_INTENTS, intent_dim=4, seed=0):
        self.pop, self.V, self.D, self.ctx_k = pop, V, D, ctx_k
        self.n_intents, self.intent_dim = n_intents, intent_dim
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(0, 0.3, (pop, V, D)).astype(np.float32)
        self.intent_emb = rng.normal(0, 0.3, (pop, n_intents, intent_dim)).astype(np.float32)
        self.Wq = rng.normal(0, 0.3, (pop, D + intent_dim, D)).astype(np.float32)
        self.bq = np.zeros((pop, D), dtype=np.float32)
        self.sigma = np.full(pop, 0.15, dtype=np.float32)

    def query(self, i, left_ids, intent_ids):
        el = self.emb[i][left_ids].mean(axis=1)             # (batch, D)
        ie = self.intent_emb[i][intent_ids]                  # (batch, intent_dim)
        combo = np.concatenate([el, ie], axis=1)
        return np.tanh(combo @ self.Wq[i] + self.bq[i])       # (batch, D)

    def score(self, i, q, cand_ids):
        w = self.emb[i][cand_ids]                            # (batch, K, D)
        return np.einsum("bd,bkd->bk", q, w)

    def accuracy(self, i, left_ids, intent_ids, cand_ids):
        q = self.query(i, left_ids, intent_ids)
        s = self.score(i, q, cand_ids)
        return float((s.argmax(axis=1) == 0).mean())

    def soft_fitness(self, i, left_ids, intent_ids, cand_ids):
        """Mean log-prob of the TRUE candidate under a softmax over the
        candidate set — soft fitness per GENREG_RULES §IV.1."""
        q = self.query(i, left_ids, intent_ids)
        s = self.score(i, q, cand_ids)
        s = s - s.max(axis=1, keepdims=True)
        logp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
        return float(logp[:, 0].mean())

    def mutate(self, i, rng):
        s = self.sigma[i]
        self.emb[i] += rng.normal(0, s, self.emb[i].shape).astype(np.float32)
        self.intent_emb[i] += rng.normal(0, s, self.intent_emb[i].shape).astype(np.float32)
        self.Wq[i] += rng.normal(0, s, self.Wq[i].shape).astype(np.float32)
        self.bq[i] += rng.normal(0, s, self.bq[i].shape).astype(np.float32)
        self.sigma[i] = np.clip(s * float(rng.lognormal(0, 0.2)), 0.01, 1.0)

    def select_into(self, order):
        self.emb = self.emb[order].copy()
        self.intent_emb = self.intent_emb[order].copy()
        self.Wq = self.Wq[order].copy()
        self.bq = self.bq[order].copy()
        self.sigma = self.sigma[order].copy()

    def export(self, i):
        return {"emb": self.emb[i].copy(), "intent_emb": self.intent_emb[i].copy(),
                "Wq": self.Wq[i].copy(), "bq": self.bq[i].copy(),
                "ctx_k": self.ctx_k, "D": self.D, "intent_dim": self.intent_dim}


def train_next_word(left, intent_feat, cand, gens=250, pop=120, D=24, ctx_k=6,
                    batch_size=512, seed=0, log=print):
    """Full training loop for the next_word contrastive genome. Same shape
    as train_fill, plus intent conditioning threaded through every call."""
    rng = np.random.default_rng(seed)
    n = len(cand)
    n_holdout = max(400, n // 20)
    perm = rng.permutation(n)
    hold_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    left_tr, intent_tr, cand_tr = left[train_idx], intent_feat[train_idx], cand[train_idx]
    left_ho, intent_ho, cand_ho = left[hold_idx], intent_feat[hold_idx], cand[hold_idx]

    V = int(max(left.max(), cand.max())) + 1
    popn = NextWordPop(pop, V, D, ctx_k, seed=seed)
    best_acc = -1.0
    best_export = None
    for g in range(gens):
        bidx = rng.integers(0, len(cand_tr), size=min(batch_size, len(cand_tr)))
        bleft, bintent, bcand = left_tr[bidx], intent_tr[bidx], cand_tr[bidx]
        # soft fitness (mean log-prob of the true candidate), §IV.1
        fits = np.array([popn.soft_fitness(i, bleft, bintent, bcand) for i in range(pop)])
        champ = int(np.argmax(fits))
        if g % 25 == 0 or g == gens - 1:
            ho_acc = popn.accuracy(champ, left_ho, intent_ho, cand_ho)
            starved = getattr(popn, "last_starved", 0)
            log(f"    gen {g:4d}  soft-fit={fits[champ]:.4f}  starved={starved}  holdout-acc={ho_acc:.3f}")
            if ho_acc > best_acc:
                best_acc = ho_acc
                best_export = popn.export(champ)
        ga_step(popn, fits, rng)
    return best_export, best_acc


def next_word_score_export(export, left_ids, intent_id, cand_ids):
    """Run a saved next_word genome export: score each candidate word for
    the next position, given left context + target intent. Returns (K,)
    scores, higher = better fit."""
    emb = export["emb"]
    el = emb[left_ids].mean(axis=0)
    ie = export["intent_emb"][intent_id]
    combo = np.concatenate([el, ie])
    q = np.tanh(combo @ export["Wq"] + export["bq"])
    w = emb[cand_ids]                                  # (K, D)
    return w @ q


def build_generation_followers(tokens, stoi, top_n=200):
    """Per-word candidate pools for RERANK generation (GENREG_RULES §VI:
    propose candidates, then let the evolved genome rerank them — never
    softmax over the whole vocabulary). For each vocab word, the top_n most
    frequent words that followed it in the corpus, plus a global top_n
    fallback for words with no followers on record.

    This is the same distribution the hard negatives were mined from, so
    training and generation finally see the SAME candidate universe — the
    fix for the norm-domination failure (§XI: a handful of large-norm
    embeddings dominating a full-vocab softmax they were never trained to
    compete in). One vectorized pass (unique over packed bigram keys), no
    per-pair python loop."""
    words = np.asarray([stoi.get(t, 0) for t in tokens if MARK_ID.get(t) is None],
                       dtype=np.int64)
    V = int(words.max()) + 1
    keys = words[:-1] * V + words[1:]
    uniq, counts = np.unique(keys, return_counts=True)
    prev = (uniq // V).astype(np.int32)
    nxt = (uniq % V).astype(np.int32)
    order = np.lexsort((-counts, prev))    # group by prev, most frequent first
    prev_s, nxt_s = prev[order], nxt[order]
    boundaries = np.searchsorted(prev_s, np.arange(V + 1))
    followers = {}
    for v in range(V):
        seg = nxt_s[boundaries[v]:boundaries[v + 1]][:top_n]
        seg = seg[seg != 0]                # never propose <unk>
        if len(seg):
            followers[int(v)] = seg.astype(np.int32)
    word_counts = np.bincount(words, minlength=V)
    word_counts[0] = 0                     # never propose <unk>
    global_top = np.argsort(-word_counts)[:top_n].astype(np.int32)
    return followers, global_top


def recognize_mark(splits_export, ctx_ids):
    """Compose the 5 binary genomes hierarchically into a single ranked list
    over all 6 marks, given one context window. splits_export: {key: export
    dict}. Returns [(mark, prob), ...] sorted by probability, using the
    product of the path probabilities that lead to each mark (end-vs-continue
    times which-end-mark or which-continue-mark)."""
    def probs(key):
        logits = forward_export(splits_export[key], ctx_ids)[0]
        e = np.exp(logits - logits.max())
        return e / e.sum()

    p_end = probs("punct_end")            # [p(continue), p(end)]
    p_q = probs("punct_question")         # [p(not-question), p(question)]
    p_ex = probs("punct_exclaim")         # [p(not-exclaim), p(exclaim)]
    p_semi = probs("punct_semicolon")     # [p(not-semi), p(semi)]
    p_col = probs("punct_colon")          # [p(not-colon), p(colon)]

    p_question = float(p_end[1] * p_q[1])
    p_exclaim = float(p_end[1] * p_ex[1])
    p_period = max(0.0, float(p_end[1]) - p_question - p_exclaim)
    p_semicolon = float(p_end[0] * p_semi[1])
    p_colon = float(p_end[0] * p_col[1])
    p_comma = max(0.0, float(p_end[0]) - p_semicolon - p_colon)

    raw = {".": p_period, "!": p_exclaim, "?": p_question,
          ",": p_comma, ";": p_semicolon, ":": p_colon}
    total = sum(raw.values()) or 1.0
    return sorted(((m, v / total) for m, v in raw.items()), key=lambda t: -t[1])


def recognize_opener(opener_splits_export, first_word_id):
    """Compose the 2 opener genomes into a ranked (statement, question,
    exclaim) guess for how a sentence will end, from ONLY its first word.
    opener_splits_export: {key: export dict} for opener_question/opener_exclaim."""
    ctx = np.asarray([[first_word_id]], dtype=np.int32)

    def probs(key):
        logits = forward_export(opener_splits_export[key], ctx)[0]
        e = np.exp(logits - logits.max())
        return e / e.sum()

    p_q = probs("opener_question")   # [p(not-question), p(question)]
    p_ex = probs("opener_exclaim")   # [p(not-exclaim), p(exclaim)]

    p_question = float(p_q[1])
    p_exclaim = float(p_ex[1])
    p_period = max(0.0, 1.0 - p_question - p_exclaim)
    raw = {".": p_period, "!": p_exclaim, "?": p_question}
    total = sum(raw.values()) or 1.0
    return sorted(((m, v / total) for m, v in raw.items()), key=lambda t: -t[1])
