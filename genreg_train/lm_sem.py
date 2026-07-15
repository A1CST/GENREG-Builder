"""LM rebuild round 3 — the "features are the environment" round.

Two new genomes, both operating INSIDE a pre-built distributional feature
space instead of evolving their own embedding tables from random init:

  "sem"     (sem_next)     — intent-conditioned next-word ranking. The genome
                             evolves a small query map into the fixed feature
                             space; score(candidate) = feat[candidate] . q.
  "grammar" (grammar_real) — real-vs-shuffled word-order discriminator. The
                             genome reads a window's feature vectors IN ORDER
                             and answers "is this real English order?"

Why the space is built and not evolved (the ga-abstraction thesis, already
validated twice in this repo — the wiki semantic-relation genomes and the
MNIST diversity-first features): evolving a V x D embedding table means the
genome must invent word similarity AND the relationship at the same time,
in a ~500K-parameter space. Round 2's next_word did exactly that and lost
to the majority-frequency baseline (21.16% vs 26.92%). Here the environment
already knows "road" and "street" are neighbors (PPMI + eigendecomposition
of the SAME corpus the genomes train on — corpus statistics, no external
model, no gradients); evolution only has to learn ONE tiny relationship
inside it (~20K parameters).

NO LOOKUP TABLES IN THE MODEL (user rule, this round's contract): the
feature matrix is environment — it is never consulted as a predictor, never
stores "what followed what", and generation never proposes candidates from
frequency pools. Every decision is an evolved genome's forward pass:
sem_next scores the ENTIRE vocabulary through the feature space, grammar_real
reranks, punctuation/opener/length genomes (round 2 artifact) drive intent
and stopping. Hard negatives at MINING time still come from corpus
statistics — that's training-data construction (the environment pushing
back), not model machinery; nothing mined is available at inference.

--------------------------------------------------------------------------
GENREG_RULES SS II model template — sem_next
  Name        sem_next
  Purpose     rank the true next word above confusable alternatives, given
              left context + sentence intent, inside the fixed feature
              space. Without it generation has no semantic word choice.
  Interface   in: left ctx word-ids (k=6), intent id (0..2); out: score per
              candidate word (higher = better). Stateless.
  Evolved     pw (k,) position weights over context slots, init N(0,0.3);
              I (3, DI=16) intent embedding, init N(0,0.3);
              Wq (F+DI, F=128) query map, init N(0, 1/sqrt(F+DI));
              bq (F,) zeros;
              bias (V,) per-word score bias, init ZEROS — the unit-norm
              feature space deliberately carries no frequency information,
              so the frequency prior must be DISCOVERED by evolution (it is
              never initialized or updated from corpus counts — that would
              be the lookup table this round prohibits);
              E1, E2 (F, R=32) low-rank bilinear transition maps (SS VI
              primitive): score += (feat[last] @ E1) . (feat[cand] @ E2) —
              "what kind of word follows what kind of word", learned as a
              map in feature space, init N(0, 1/sqrt(F)).
              Total ~= 6 + 48 + 18432 + 128 + 4001 + 8192 ~= 30.8K params.
  Fitness     mean log softmax-prob of the true candidate over its
              candidate set (soft, SS IV.1). Range (-inf, 0].
  Energy      DECAY=0.95, GAIN=0.05 * clip(robust-z(EMA-fit - median), -1, 1),
              FLOOR=0.2, E_MAX=1.5; offspring inherit parent energy minus
              BIRTH_COST=0.1 (floored at 0.3 so newborns get ~3 gens to
              prove themselves). Tuned across two smokes: no birth cost ->
              starved=0 (decorative, SS XI); cost 0.2 + gain 0.1 -> 33-41%
              starved (near-genocidal). All selection + energy decisions
              act on a per-lineage fitness EMA (SS IV.7), not one batch.
  Selection   POP=120, elite 10%, tournament k=4 among non-starved.
  Mutation    sigma self-adapted per genome, lognormal walk, clip [.01, 1].
  Hyperparams N_GENS=300, BATCH=512, LOG_EVERY=25.
  Success     local: holdout candidate-set acc > majority-frequency
              baseline on the SAME candidate sets (the bar round 2 failed);
              downstream: full-vocab top-1 > always-most-frequent-word AND
              generation output visibly on-topic.
  Failure     watch: train climbs / holdout flat (mismatch); all genomes
              identical output (collapse); starved=0 (energy decorative)
              or starved>50% (genocidal).
  Baselines   majority-frequency candidate pick; most-frequent-word top-1;
              round 2 next_word (21.16%).
  Artifacts   corpora/combined/lm_sem.pkl, lm_sem.log, runs/lm/<ts>-lm-*.

GENREG_RULES SS II model template — grammar_real
  Name        grammar_real
  Purpose     tell real English word order from shuffled. Without it
              generation strings plausible-per-step words into unordered
              soup ("near-grammatical" is round 2's visible failure).
  Interface   in: m=6 word-ids (one clause-internal window); out: one
              logit (>0 = real order). Stateless.
  Evolved     W1 (m*F, H=32) init N(0, 1/sqrt(m*F)); b1 zeros; acts (H,)
              per-neuron activation ids from the 8-catalog (SS VI signature
              primitive); W2 (H,) init N(0,0.3); b2 scalar.
              Total ~= 24576 + 32 + 32 + 33 ~= 24.7K params.
  Fitness     mean log sigmoid(label * logit) (soft binary, SS IV.1).
  Energy      same scheme as sem_next.
  Selection   POP=120, elite 10%, tournament k=4 among non-starved.
  Mutation    sigma self-adapted; activation ids flip with p=0.02/neuron.
  Hyperparams N_GENS=300, BATCH=512 (256 real + 256 shuffled), LOG_EVERY=25.
  Success     local: holdout balanced acc meaningfully > 50% chance;
              downstream: reranking by grammar logit visibly improves word
              order in generated sentences.
  Failure     same watch-list as sem_next; plus "shuffle too easy" (window
              with repeated words -> shuffle == original; mining skips those).
  Baselines   50% chance; always-predict-real.
  Artifacts   shared with sem_next.
--------------------------------------------------------------------------
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

F_DIM = 128          # feature-space dimensionality (matches wiki_feats convention)
COOC_WIN = 5         # co-occurrence window for the PPMI matrix
SEM_DI = 16          # intent embedding width inside sem_next
GRAM_M = 6           # grammar window length (words)
GRAM_H = 32          # grammar hidden width

SEM_GROUP = "sem"
SEM_SPLIT = {"key": "sem_next", "group": SEM_GROUP,
             "desc": "inside the pre-built PPMI/eig feature space, does an evolved "
                     "query (left context + intent) rank the TRUE next word above "
                     "hard corrupted candidates?",
             "positive_name": "true-word-wins", "negative_name": "n/a"}

GRAM_GROUP = "grammar"
GRAM_SPLIT = {"key": "grammar_real", "group": GRAM_GROUP,
              "desc": "is this window of words real English order, or the same "
                      "words shuffled?",
              "positive_name": "real", "negative_name": "shuffled"}


# ==========================================================================
# The environment: distributional features built from the training corpus
# itself (PPMI of windowed co-occurrence -> symmetric eigendecomposition).
# Built once, frozen, unit-normalized. Row 0 (<unk>) is zeroed so <unk> can
# neither be predicted nor contribute context signal.
# ==========================================================================
def build_feature_space(words, V, dim=F_DIM, win=COOC_WIN, log=print):
    """words: flat int array of word ids (marks stripped). Returns (V, dim)
    float32, rows unit-norm (zero rows for words with no usable counts).
    One bincount per offset — O(win * n), no python-level pair loop."""
    C = np.zeros(V * V, dtype=np.float64)
    n = len(words)
    a64 = words.astype(np.int64)
    for off in range(1, win + 1):
        keys = a64[:-off] * V + a64[off:]
        C += np.bincount(keys, minlength=V * V)
    C = C.reshape(V, V)
    C = C + C.T                                   # symmetric context counts
    C[0, :] = 0.0
    C[:, 0] = 0.0
    total = C.sum()
    row = C.sum(axis=1)
    log(f"  co-occurrence built: {int(total):,} pair counts over {n:,} words")
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C * total) / (row[:, None] * row[None, :]))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
    log("  PPMI done; eigendecomposition...")
    vals, vecs = np.linalg.eigh(ppmi)             # ascending; PPMI is symmetric
    top = np.argsort(-vals)[:dim]
    feats = (vecs[:, top] * np.sqrt(np.maximum(vals[top], 0.0))).astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = np.where(norms > 1e-8, feats / (norms + 1e-12), 0.0).astype(np.float32)
    feats[0] = 0.0
    log(f"  feature space: {feats.shape}, kept eigenvalues "
        f"{vals[top].max():.1f}..{vals[top].min():.1f}")
    return feats


def nn_probe(feats, vocab, stoi, probes=("king", "france", "hot", "good", "car"),
             topn=5, log=print):
    """SS VII.3 inspect-the-space check: nearest neighbors of a few probe
    words. A broken space fails here before wasting a training run."""
    for w in probes:
        i = stoi.get(w)
        if i is None or not feats[i].any():
            log(f"    nn[{w}]: (not in vocab / no features)")
            continue
        sims = feats @ feats[i]
        sims[i] = -2.0
        best = np.argsort(-sims)[:topn]
        log(f"    nn[{w}]: " + ", ".join(f"{vocab[j]}({sims[j]:.2f})" for j in best))


# ==========================================================================
# Energy homeostasis, SS III-compliant this time. Round 2's version produced
# starved=0 for the entire run (decorative — a named failure mode, SS XI)
# because tournament offspring reset to energy 1.0 every generation, so
# nothing persisted long enough to starve. Two fixes: (1) delta is a
# robust-z of (fitness - median) per SS III, not a rank percentile; (2)
# offspring INHERIT their parent's energy, so persistently-weak LINEAGES
# ride toward the floor across generations even as individuals get copied.
# Equilibrium: e* = z (clipped to +-1); a genome must hold above the median
# long-term to sustain e above the 0.2 floor (decay from 1.0 crosses the
# floor in ~15 generations of exactly-median performance).
# ==========================================================================
ENERGY_DECAY = 0.95
ENERGY_GAIN = 0.05
ENERGY_FLOOR = 0.2
E_MAX = 1.5
BIRTH_COST = 0.1     # child energy = parent energy - this; see SS II template
BIRTH_MIN = 0.3      # newborns get at least ~3 gens to prove themselves
FIT_EMA = 0.7        # SS IV.7: noise-driven culling destroys ratchets — all
                     # selection/energy decisions act on a per-lineage EMA of
                     # fitness, not one batch's number (smoke 2 showed flat
                     # soft-fit for 60 gens: batch noise >> genome differences)


def ga_step_energy(pop_obj, fits, rng, elite_frac=0.1, tourn_k=2):
    pop = pop_obj.pop
    if not hasattr(pop_obj, "energy"):
        pop_obj.energy = np.full(pop, 1.0, dtype=np.float32)
        pop_obj.fit_ema = fits.astype(np.float64).copy()
    pop_obj.fit_ema = FIT_EMA * pop_obj.fit_ema + (1.0 - FIT_EMA) * fits
    ema = pop_obj.fit_ema
    med = float(np.median(ema))
    mad = float(np.median(np.abs(ema - med)))
    z = np.clip((ema - med) / (1.4826 * mad + 1e-9), -1.0, 1.0)
    pop_obj.energy = np.clip(pop_obj.energy * ENERGY_DECAY + ENERGY_GAIN * z,
                             0.0, E_MAX).astype(np.float32)
    starved = pop_obj.energy < ENERGY_FLOOR
    pop_obj.last_starved = int(starved.sum())
    pop_obj.mean_energy = float(pop_obj.energy.mean())
    alive = np.flatnonzero(~starved)
    if len(alive) < max(2, tourn_k):              # pathological all-starved guard
        alive = np.argsort(-ema)[:max(2, tourn_k)]
    alive_by_fit = alive[np.argsort(-ema[alive])]
    n_elite = max(1, int(pop * elite_frac))
    new_order = alive_by_fit[:n_elite].tolist()
    new_energy = [float(pop_obj.energy[i]) for i in new_order]
    new_ema = [float(ema[i]) for i in new_order]
    while len(new_order) < pop:
        cand = rng.choice(alive, size=min(tourn_k, len(alive)), replace=False)
        parent = int(cand[np.argmax(ema[cand])])
        new_order.append(parent)
        new_energy.append(max(BIRTH_MIN, float(pop_obj.energy[parent]) - BIRTH_COST))
        new_ema.append(float(ema[parent]))        # lineage EMA carries over
    pop_obj.select_into(new_order)
    pop_obj.energy = np.asarray(new_energy, dtype=np.float32)
    pop_obj.fit_ema = np.asarray(new_ema, dtype=np.float64)
    for i in range(n_elite, pop):
        pop_obj.mutate(i, rng)


# ==========================================================================
# sem_next — evolved query into the fixed feature space
# ==========================================================================
SEM_R = 32           # rank of the bilinear transition maps


class SemNextPop:
    def __init__(self, pop, feats, ctx_k, logfreq=None, seed=0):
        self.pop, self.ctx_k = pop, ctx_k
        self.feats = feats                        # FIXED environment, not evolved
        F = feats.shape[1]
        V = feats.shape[0]
        self.F, self.V = F, V
        # log-frequency environment feature (normalized, per word). Like
        # the SVD features it's a corpus statistic the genome READS — the
        # evolved scalar wf decides whether/how much it matters (init 0).
        # Making evolution rediscover the frequency prior across 4001
        # independent bias genes instead is an unreachable fitness.
        self.lf = (np.zeros(V, np.float32) if logfreq is None
                   else logfreq.astype(np.float32))
        rng = np.random.default_rng(seed)
        self.pw = rng.normal(0, 0.3, (pop, ctx_k)).astype(np.float32)
        self.I = rng.normal(0, 0.3, (pop, 3, SEM_DI)).astype(np.float32)
        # BOOTSTRAP init (SS VI: cascades cannot be evolved from random
        # init): Wq starts as the identity on the context block — the gen-0
        # query IS the pooled context vector (pure distributional
        # similarity), and evolution deforms from sense, not from noise.
        eye = np.zeros((F + SEM_DI, F), dtype=np.float32)
        eye[:F, :] = np.eye(F, dtype=np.float32)
        self.Wq = (eye[None, :, :]
                   + rng.normal(0, 0.02 / np.sqrt(F + SEM_DI),
                                (pop, F + SEM_DI, F))).astype(np.float32)
        self.bq = np.zeros((pop, F), dtype=np.float32)
        # per-word fine-correction bias, init ZEROS (evolved, not counted)
        self.bias = np.zeros((pop, V), dtype=np.float32)
        # low-rank bilinear transition (SS VI): what-follows-what as a MAP.
        # Near-no-op init (x0.05) — fades IN if selected for, instead of
        # masking the query signal with random bilinear noise at gen 0.
        self.E1 = (rng.normal(0, 1.0 / np.sqrt(F), (pop, F, SEM_R)) * 0.05).astype(np.float32)
        self.E2 = (rng.normal(0, 1.0 / np.sqrt(F), (pop, F, SEM_R)) * 0.05).astype(np.float32)
        self.wf = np.zeros(pop, dtype=np.float32)  # evolved logfreq weight
        self.sigma = np.full(pop, 0.3, dtype=np.float32)

    def query(self, i, left_ids, intent_ids):
        cf = self.feats[left_ids]                             # (B, k, F)
        w = np.exp(self.pw[i] - self.pw[i].max())
        w = (w / w.sum()).astype(np.float32)
        c = np.einsum("bkf,k->bf", cf, w)                     # (B, F)
        ie = self.I[i][intent_ids]                            # (B, DI)
        q = np.tanh(np.concatenate([c, ie], axis=1) @ self.Wq[i] + self.bq[i])
        return q                                              # (B, F)

    def cand_scores(self, i, left_ids, intent_ids, cand_ids):
        q = self.query(i, left_ids, intent_ids)
        cf = self.feats[cand_ids]                             # (B, C, F)
        s = np.einsum("bcf,bf->bc", cf, q)                    # semantic match
        s = s + self.bias[i][cand_ids]                        # evolved fine bias
        s = s + self.wf[i] * self.lf[cand_ids]                # evolved freq weight
        t1 = self.feats[left_ids[:, -1]] @ self.E1[i]         # (B, R)
        t2 = np.einsum("bcf,fr->bcr", cf, self.E2[i])         # (B, C, R)
        return s + np.einsum("bcr,br->bc", t2, t1)            # + transition

    def soft_fitness(self, i, left_ids, intent_ids, cand_ids):
        s = self.cand_scores(i, left_ids, intent_ids, cand_ids)
        s = s - s.max(axis=1, keepdims=True)
        logp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
        return float(logp[:, 0].mean())

    def accuracy(self, i, left_ids, intent_ids, cand_ids):
        s = self.cand_scores(i, left_ids, intent_ids, cand_ids)
        return float((s.argmax(axis=1) == 0).mean())

    def vocab_topk(self, i, left_ids, intent_ids, true_ids, ks=(1, 5)):
        """Full-vocabulary ranking — the genome scoring EVERY word through
        the feature space (this is also exactly what generation does)."""
        q = self.query(i, left_ids, intent_ids)               # (B, F)
        scores = q @ self.feats.T + self.bias[i] + self.wf[i] * self.lf
        t1 = self.feats[left_ids[:, -1]] @ self.E1[i]         # (B, R)
        scores = scores + t1 @ (self.feats @ self.E2[i]).T    # transition term
        scores[:, 0] = -1e9
        out = {}
        order = np.argsort(-scores, axis=1)
        for k in ks:
            out[k] = float((order[:, :k] == true_ids[:, None]).any(axis=1).mean())
        return out

    def mutate(self, i, rng):
        # RELATIVE mutation (SS V per-tensor scaling): each tensor is
        # perturbed by sigma * its own init scale, so sigma is a
        # dimensionless knob. Smokes 1-3 used absolute sigma ~= the init
        # scale of Wq — every mutation re-randomized the genome and no
        # improvement could accumulate (flat soft-fit for 60 gens).
        s = self.sigma[i]
        F = self.F
        self.pw[i] += rng.normal(0, s * 0.3, self.pw[i].shape).astype(np.float32)
        self.I[i] += rng.normal(0, s * 0.3, self.I[i].shape).astype(np.float32)
        self.Wq[i] += rng.normal(0, s / np.sqrt(F + SEM_DI),
                                 self.Wq[i].shape).astype(np.float32)
        self.bq[i] += rng.normal(0, s * 0.05, self.bq[i].shape).astype(np.float32)
        # sparse bias mutation: perturb a random ~2% of words per event —
        # dense V-wide noise every generation would swamp the small tensors
        nb = max(1, self.V // 50)
        j = rng.integers(0, self.V, nb)
        self.bias[i, j] += rng.normal(0, s * 0.2, nb).astype(np.float32)
        self.E1[i] += rng.normal(0, s * 0.05 / np.sqrt(F), self.E1[i].shape).astype(np.float32)
        self.E2[i] += rng.normal(0, s * 0.05 / np.sqrt(F), self.E2[i].shape).astype(np.float32)
        self.wf[i] += float(rng.normal(0, s * 0.3))
        self.sigma[i] = np.clip(s * float(rng.lognormal(0, 0.2)), 0.02, 2.0)

    def select_into(self, order):
        self.pw = self.pw[order].copy()
        self.I = self.I[order].copy()
        self.Wq = self.Wq[order].copy()
        self.bq = self.bq[order].copy()
        self.bias = self.bias[order].copy()
        self.E1 = self.E1[order].copy()
        self.E2 = self.E2[order].copy()
        self.wf = self.wf[order].copy()
        self.sigma = self.sigma[order].copy()

    def export(self, i):
        return {"pw": self.pw[i].copy(), "I": self.I[i].copy(),
                "Wq": self.Wq[i].copy(), "bq": self.bq[i].copy(),
                "bias": self.bias[i].copy(), "E1": self.E1[i].copy(),
                "E2": self.E2[i].copy(), "wf": float(self.wf[i]),
                "ctx_k": self.ctx_k, "F": self.F, "DI": SEM_DI, "R": SEM_R}


def sem_query_export(export, feats, left_ids, intent_id):
    """Saved-genome query for a single context. left_ids (k,), intent_id int."""
    w = np.exp(export["pw"] - export["pw"].max())
    w = w / w.sum()
    c = (feats[left_ids] * w[:, None]).sum(axis=0)
    ie = export["I"][intent_id]
    return np.tanh(np.concatenate([c, ie]) @ export["Wq"] + export["bq"])


def sem_vocab_scores_export(export, feats, logfreq, left_ids, intent_id):
    """Saved-genome score over the ENTIRE vocabulary for one context —
    the generation-time path. All terms are the genome's own evolved
    parameters acting on the fixed environment (features + logfreq)."""
    q = sem_query_export(export, feats, left_ids, intent_id)
    scores = feats @ q + export["bias"] + export["wf"] * logfreq
    t1 = feats[left_ids[-1]] @ export["E1"]                   # (R,)
    scores = scores + (feats @ export["E2"]) @ t1
    scores[0] = -1e9
    return scores


def train_sem_next(feats, left, intent_feat, cand, logfreq=None, gens=300,
                   pop=120, batch_size=2048, seed=0, log=print):
    rng = np.random.default_rng(seed)
    n = len(cand)
    n_holdout = max(2000, n // 20)
    perm = rng.permutation(n)
    ho, tr = perm[:n_holdout], perm[n_holdout:]
    left_tr, intent_tr, cand_tr = left[tr], intent_feat[tr], cand[tr]
    left_ho, intent_ho, cand_ho = left[ho], intent_feat[ho], cand[ho]

    popn = SemNextPop(pop, feats, ctx_k=left.shape[1], logfreq=logfreq, seed=seed)
    best_acc, best_export = -1.0, None
    bidx = None
    for g in range(gens):
        # FIXED probe batch, resampled every 25 gens: on a shared fixed
        # batch, genome comparisons are exact, so differences far smaller
        # than one batch's sampling noise are still selectable (smokes 1-3:
        # noise std ~0.045 vs genome differences ~0.005 — selection was a
        # coin flip). Champion selection still uses the true holdout.
        if g % 25 == 0:
            bidx = rng.integers(0, len(cand_tr), size=min(batch_size, len(cand_tr)))
        fits = np.array([popn.soft_fitness(i, left_tr[bidx], intent_tr[bidx],
                                           cand_tr[bidx]) for i in range(pop)])
        champ = int(np.argmax(fits))
        if g % 25 == 0 or g == gens - 1:
            ho_acc = popn.accuracy(champ, left_ho, intent_ho, cand_ho)
            log(f"    gen {g:4d}  soft-fit={fits[champ]:.4f}  "
                f"starved={getattr(popn, 'last_starved', 0)}  "
                f"mean-e={getattr(popn, 'mean_energy', 1.0):.2f}  "
                f"holdout-acc={ho_acc:.3f}")
            if ho_acc > best_acc:
                best_acc = ho_acc
                best_export = popn.export(champ)
        ga_step_energy(popn, fits, rng)

    final = SemNextPop(1, feats, ctx_k=left.shape[1], logfreq=logfreq, seed=seed)
    final.pw[0], final.I[0] = best_export["pw"], best_export["I"]
    final.Wq[0], final.bq[0] = best_export["Wq"], best_export["bq"]
    final.bias[0], final.E1[0], final.E2[0] = (best_export["bias"],
                                               best_export["E1"], best_export["E2"])
    final.wf[0] = best_export["wf"]
    topk = final.vocab_topk(0, left_ho, intent_ho, cand_ho[:, 0])
    return best_export, best_acc, topk, (left_ho, intent_ho, cand_ho)


# ==========================================================================
# grammar_real — real-vs-shuffled word order discriminator
# ==========================================================================
# The 8-function activation catalog (SS VI signature primitive — each neuron
# literally sees through a different mathematical lens).
def _act(x, a):
    if a == 0:
        return np.tanh(x)
    if a == 1:
        return 1.0 / (1.0 + np.exp(-x))
    if a == 2:
        return np.maximum(x, 0.0)
    if a == 3:
        return x
    if a == 4:
        return np.sin(x)
    if a == 5:
        return np.exp(-x * x)
    if a == 6:
        return np.abs(x)
    return np.sign(x) * np.sqrt(np.abs(x))        # a == 7


def mine_grammar_examples(tokens, stoi, mark_id, m=GRAM_M, n_samples=1_000_000,
                          seed=0):
    """Clause-internal windows of m consecutive words (all 6 marks are
    boundaries — grammar signal shouldn't straddle a comma). Each window
    yields one REAL example and one SHUFFLED negative (a permutation
    verified to differ; windows of near-identical words that can't shuffle
    into a different order are skipped). Balanced 50/50 by construction."""
    rng = np.random.default_rng(seed)
    clause = []
    windows = []
    budget = n_samples // 2
    step = max(1, m // 2)
    for tok in tokens:
        if mark_id.get(tok) is not None:
            if len(clause) >= m:
                arr = np.asarray(clause, dtype=np.int32)
                for s in range(0, len(arr) - m + 1, step):
                    windows.append(arr[s:s + m])
            clause = []
            if len(windows) >= budget:
                break
            continue
        clause.append(stoi.get(tok, 0))
    windows = np.stack(windows[:budget])
    real = windows
    shuf = np.empty_like(real)
    keep = np.ones(len(real), dtype=bool)
    for r in range(len(real)):
        w = real[r]
        ok = False
        for _ in range(5):
            p = rng.permutation(m)
            if not np.array_equal(w[p], w):
                shuf[r] = w[p]
                ok = True
                break
        keep[r] = ok
    real, shuf = real[keep], shuf[keep]
    X = np.concatenate([real, shuf])
    y = np.concatenate([np.ones(len(real), np.int32), np.zeros(len(shuf), np.int32)])
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


class GrammarPop:
    def __init__(self, pop, feats, m=GRAM_M, H=GRAM_H, seed=0):
        self.pop, self.m, self.H = pop, m, H
        self.feats = feats                        # FIXED environment
        F = feats.shape[1]
        self.F = F
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(m * F), (pop, m * F, H)).astype(np.float32)
        self.b1 = np.zeros((pop, H), dtype=np.float32)
        self.acts = rng.integers(0, 8, (pop, H)).astype(np.int8)
        self.W2 = rng.normal(0, 0.3, (pop, H)).astype(np.float32)
        self.b2 = np.zeros(pop, dtype=np.float32)
        self.sigma = np.full(pop, 0.3, dtype=np.float32)

    def logits(self, i, win_ids):
        x = self.feats[win_ids].reshape(len(win_ids), self.m * self.F)
        pre = x @ self.W1[i] + self.b1[i]                     # (B, H)
        h = np.empty_like(pre)
        for a in range(8):
            cols = np.flatnonzero(self.acts[i] == a)
            if len(cols):
                h[:, cols] = _act(pre[:, cols], a)
        return h @ self.W2[i] + self.b2[i]                    # (B,)

    def soft_fitness(self, i, win_ids, labels):
        s = self.logits(i, win_ids)
        sign = np.where(labels == 1, 1.0, -1.0)
        z = sign * s
        # log sigmoid(z), numerically stable
        return float(np.where(z > 0, -np.log1p(np.exp(-z)),
                              z - np.log1p(np.exp(z))).mean())

    def balanced_accuracy(self, i, win_ids, labels):
        pred = (self.logits(i, win_ids) > 0).astype(np.int32)
        accs = [float((pred[labels == c] == c).mean()) for c in (0, 1)
                if (labels == c).any()]
        return float(np.mean(accs))

    def mutate(self, i, rng):
        # relative mutation, same rationale as SemNextPop.mutate
        s = self.sigma[i]
        self.W1[i] += rng.normal(0, s / np.sqrt(self.m * self.F),
                                 self.W1[i].shape).astype(np.float32)
        self.b1[i] += rng.normal(0, s * 0.05, self.b1[i].shape).astype(np.float32)
        self.W2[i] += rng.normal(0, s * 0.3, self.W2[i].shape).astype(np.float32)
        self.b2[i] += float(rng.normal(0, s * 0.05))
        flip = rng.random(self.H) < 0.02
        if flip.any():
            self.acts[i, flip] = rng.integers(0, 8, int(flip.sum()))
        self.sigma[i] = np.clip(s * float(rng.lognormal(0, 0.2)), 0.02, 2.0)

    def select_into(self, order):
        self.W1 = self.W1[order].copy()
        self.b1 = self.b1[order].copy()
        self.acts = self.acts[order].copy()
        self.W2 = self.W2[order].copy()
        self.b2 = self.b2[order].copy()
        self.sigma = self.sigma[order].copy()

    def export(self, i):
        return {"W1": self.W1[i].copy(), "b1": self.b1[i].copy(),
                "acts": self.acts[i].copy(), "W2": self.W2[i].copy(),
                "b2": float(self.b2[i]), "m": self.m, "F": self.F, "H": self.H}


def grammar_logit_export(export, feats, win_ids):
    """Saved-genome logit(s). win_ids (m,) or (B, m)."""
    win_ids = np.atleast_2d(win_ids)
    x = feats[win_ids].reshape(len(win_ids), export["m"] * export["F"])
    pre = x @ export["W1"] + export["b1"]
    h = np.empty_like(pre)
    for a in range(8):
        cols = np.flatnonzero(export["acts"] == a)
        if len(cols):
            h[:, cols] = _act(pre[:, cols], a)
    return h @ export["W2"] + export["b2"]


def train_grammar(feats, X, y, gens=300, pop=120, batch_size=2048, seed=0,
                  log=print):
    rng = np.random.default_rng(seed)
    n = len(y)
    n_holdout = max(2000, n // 20)
    perm = rng.permutation(n)
    ho, tr = perm[:n_holdout], perm[n_holdout:]
    X_tr, y_tr = X[tr], y[tr]
    X_ho, y_ho = X[ho], y[ho]

    popn = GrammarPop(pop, feats, seed=seed)
    best_acc, best_export = -1.0, None
    bidx = None
    for g in range(gens):
        # fixed probe batch, resampled every 25 gens — see train_sem_next
        if g % 25 == 0:
            bidx = rng.integers(0, len(y_tr), size=min(batch_size, len(y_tr)))
        fits = np.array([popn.soft_fitness(i, X_tr[bidx], y_tr[bidx])
                         for i in range(pop)])
        champ = int(np.argmax(fits))
        if g % 25 == 0 or g == gens - 1:
            ho_acc = popn.balanced_accuracy(champ, X_ho, y_ho)
            log(f"    gen {g:4d}  soft-fit={fits[champ]:.4f}  "
                f"starved={getattr(popn, 'last_starved', 0)}  "
                f"mean-e={getattr(popn, 'mean_energy', 1.0):.2f}  "
                f"holdout-balanced-acc={ho_acc:.3f}")
            if ho_acc > best_acc:
                best_acc = ho_acc
                best_export = popn.export(champ)
        ga_step_energy(popn, fits, rng)
    return best_export, best_acc
