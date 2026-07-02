"""Energy-gated evolutionary next-word prediction.

Each genome is a tiny single-hidden-layer network mapping a context window of
word IDs to a distribution over the vocabulary. Genomes evolve independently by
mutation only (no crossover). The defining feature is an *energy economy*:
correct predictions are free, wrong ones burn energy, so accurate genomes stay
alive longer in a round, see more of the corpus, and accumulate more correct
predictions -- a triple-compounding advantage.

Mutation is *relative*: each weight is perturbed proportional to its own
magnitude, so large weights never calcify and small weights are never blown out.
"""
import os
import pickle
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus.txt")
BEST_DIR = os.path.join(HERE, "best")

# ---- hyperparameters --------------------------------------------------------
CONTEXT = 3            # context window: N previous words -> next word
HIDDEN = 64            # MAX hidden units; each genome uses an evolvable subset
MIN_WIDTH = 2          # floor on a genome's active hidden width
EMBED = 16             # per-word embedding dimension
WIDTH_MUT_RATE = 0.30  # prob of mutating the active width per reproduction
WIDTH_MUT_SCALE = 0.15 # relative step size for width mutation
POP_SIZE = 64
START_ENERGY = 500     # energy each genome gets per round (large vocab needs
                       # enough runway for low-accuracy genomes to differentiate)
HARD_CAP = 3000        # safety ceiling on generations (don't grind forever)
# Plateau detection: stop once the smoothed best fitness stops improving.
PLATEAU_WINDOW = 50    # gens to average best-fitness over (cancels round noise)
PLATEAU_PATIENCE = 300 # stop after this many gens with no smoothed improvement
PLATEAU_EPS = 0.5      # min absolute gain in the smoothed metric to count

# Time pressure: genomes that evaluate faster than the population's average
# wall-clock time get a small fitness bump (rewards computational efficiency /
# Occam's razor). Applied to the fitness used for selection.
TIME_BONUS = 2.0       # small but noticeable bump for below-average eval time
# Optional warm-up gate (EEC_GATE=1): withhold the time bonus until accuracy
# ignites, so Occam pressure can't strangle the model before it has signal.
GATE_THRESHOLD = 8.0   # latch time pressure on once smoothed raw best clears this
GATE_WINDOW = 10       # gens of best_raw to smooth over for the gate
WIDTH_COST = 0.5       # EEC_WCOST: max fraction of fitness a full-width genome
                       # forfeits (relative compute penalty, scales with width)
TOURNAMENT_K = 3       # tournament selection size
MUT_RATE = 0.15        # fraction of weights perturbed per mutation
MUT_SCALE = 0.20       # relative perturbation magnitude
MAX_VOCAB = int(os.environ.get("EEC_VOCAB", "8000"))   # cap vocab to most frequent words
SEED = 0

WORD_RE = None  # set after import to avoid re cost at top if unused


def tokenize(text):
    import re
    return re.findall(r"[a-z]+|[0-9]+|[^\sa-z0-9]", text.lower())


def build_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    tokens = tokenize(text)
    # frequency-rank vocab, cap size; rare words map to <unk> (id 0)
    from collections import Counter
    counts = Counter(tokens)
    vocab = ["<unk>"] + [w for w, _ in counts.most_common(MAX_VOCAB - 1)]
    word2id = {w: i for i, w in enumerate(vocab)}
    ids = np.array([word2id.get(t, 0) for t in tokens], dtype=np.int32)
    return ids, vocab, word2id


def make_sequences(ids):
    """Sliding windows: CONTEXT ids -> next id."""
    n = len(ids) - CONTEXT
    X = np.empty((n, CONTEXT), dtype=np.int32)
    for i in range(CONTEXT):
        X[:, i] = ids[i:i + n]
    y = ids[CONTEXT:CONTEXT + n]
    return X, y


class Genome:
    """Embedding table + single hidden layer + output projection."""

    def __init__(self, vocab_size, rng):
        s = 1.0 / np.sqrt(EMBED)
        self.E = rng.normal(0, 0.1, (vocab_size, EMBED)).astype(np.float32)
        self.W1 = (rng.normal(0, s, (CONTEXT * EMBED, HIDDEN)) ).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = (rng.normal(0, 1.0 / np.sqrt(HIDDEN), (HIDDEN, vocab_size))).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)
        # evolvable active hidden width: only the first `width` units are used,
        # so compute (and wall-clock time) scales with width. Seed a spread so
        # the time signal is real from gen 1.
        self.width = int(rng.integers(MIN_WIDTH, HIDDEN + 1))

    def params(self):
        return [self.E, self.W1, self.b1, self.W2, self.b2]

    @classmethod
    def from_params(cls, plist, width=HIDDEN):
        g = cls.__new__(cls)
        g.E, g.W1, g.b1, g.W2, g.b2 = [np.array(p, dtype=np.float32) for p in plist]
        g.width = int(width)
        return g

    def predict(self, ctx_rows):
        """ctx_rows: (B, CONTEXT) int ids -> (B,) argmax predicted ids.

        Only the genome's active hidden width participates, so a leaner genome
        does proportionally fewer FLOPs in both matmuls.
        """
        w = self.width
        emb = self.E[ctx_rows].reshape(ctx_rows.shape[0], -1)        # (B, CONTEXT*EMBED)
        h = np.tanh(emb @ self.W1[:, :w] + self.b1[:w])             # (B, w)
        logits = h @ self.W2[:w, :] + self.b2                       # (B, vocab)
        return np.argmax(logits, axis=1)

    def copy(self):
        g = Genome.__new__(Genome)
        g.E = self.E.copy(); g.W1 = self.W1.copy(); g.b1 = self.b1.copy()
        g.W2 = self.W2.copy(); g.b2 = self.b2.copy()
        g.width = self.width
        return g

    def mutate(self, rng):
        """RELATIVE mutation: perturbation scales with each weight's magnitude.

        A weight near 0 gets a near-zero nudge; a large weight gets a
        proportionally large one. A small absolute floor keeps dead-zero
        weights from being permanently frozen at zero. The active hidden width
        mutates relatively too, so structure can grow or shrink.
        """
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            # relative step + tiny absolute floor so zeros can still move
            step = rng.normal(0, 1, p.shape).astype(np.float32)
            scale = MUT_SCALE * (np.abs(p) + 1e-3)
            p += mask * step * scale
        # relative width mutation (step proportional to current width)
        if rng.random() < WIDTH_MUT_RATE:
            delta = rng.normal(0, WIDTH_MUT_SCALE * self.width)
            self.width = int(np.clip(round(self.width + delta), MIN_WIDTH, HIDDEN))


def evaluate(genome, X, y, order):
    """Run one energy-gated round. Returns (fitness, energy_left, words_seen).

    Start with START_ENERGY. Each prediction costs 1 energy; correct refunds it
    (net 0), wrong loses it (net -1). Stop when energy hits 0. Fitness = number
    of correct predictions made before running out.
    """
    energy = START_ENERGY
    correct = 0
    seen = 0
    BATCH = 1024
    pos = 0
    # Stream the shuffled corpus in batches. Within a batch we account energy
    # vectorized: correct refunds (net 0), wrong costs 1. The genome dies the
    # instant cumulative wrongs reach its remaining energy.
    while energy > 0 and pos < len(order):
        idx = order[pos:pos + BATCH]
        hits = genome.predict(X[idx]) == y[idx]
        cum_wrongs = np.cumsum(~hits)
        if cum_wrongs[-1] < energy:
            correct += int(hits.sum())
            seen += len(idx)
            energy -= int(cum_wrongs[-1])
            pos += BATCH
        else:
            # energy drains at the `energy`-th wrong inside this batch
            cut = int(np.searchsorted(cum_wrongs, energy))   # first idx hitting it
            correct += int(hits[:cut + 1].sum())
            seen += cut + 1
            energy = 0
    return correct, energy, seen


def main():
    os.makedirs(BEST_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading corpus...")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences(ids)
    vocab_size = len(vocab)
    print(f"Corpus: {len(ids):,} tokens | vocab: {vocab_size:,} | sequences: {len(X):,}")

    id2word = vocab

    # ---- resume from a saved best genome, or start cold-random -------------
    ckpt = os.path.join(BEST_DIR, "best.pkl")
    start_gen = 1
    if os.environ.get("EEC_RESUME") and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            saved = pickle.load(f)
        seed_genome = Genome.from_params(saved["genome"], saved.get("width", HIDDEN))
        start_gen = saved["gen"] + 1
        # re-seed the population from the saved elite: one exact copy + mutants
        population = [seed_genome.copy()]
        while len(population) < POP_SIZE:
            child = seed_genome.copy()
            child.mutate(rng)
            population.append(child)
        print(f"Resumed from gen {saved['gen']} (fitness {saved['fitness']}); "
              f"continuing at gen {start_gen}")
    else:
        population = [Genome(vocab_size, rng) for _ in range(POP_SIZE)]

    # ---- plateau tracking --------------------------------------------------
    best_hist = []          # best fitness per generation
    smoothed_best = -1.0    # best smoothed value seen so far
    stall = 0               # gens since smoothed value last improved

    # ---- time-pressure warm-up gate ---------------------------------------
    gated = bool(os.environ.get("EEC_GATE"))
    time_on = not gated     # if not gated, time pressure is on from gen 1
    raw_hist = []           # best raw fitness per gen, for the ignition gate

    # ---- width-cost mode (EEC_WCOST) --------------------------------------
    # Deterministic compute cost modeled as a RELATIVE penalty on fitness:
    #   fitness = raw * (1 - WIDTH_COST * width/HIDDEN)
    # Self-gating (penalty ~0 while raw~0) and noise-free, so parsimony and
    # accuracy stay live pressures simultaneously instead of one crowding out
    # the other. Supersedes the flat wall-clock bonus when enabled.
    wcost = bool(os.environ.get("EEC_WCOST"))

    for gen in range(start_gen, HARD_CAP + 1):
        # fresh shuffled view of the data each generation
        order = rng.permutation(len(X))
        results = []
        times = np.empty(POP_SIZE)
        for gi, g in enumerate(population):
            t0 = time.perf_counter()
            results.append(evaluate(g, X, y, order))
            times[gi] = time.perf_counter() - t0
        raw_fits = np.array([r[0] for r in results], dtype=np.float64)
        energies = np.array([r[1] for r in results])
        seens = np.array([r[2] for r in results])

        avg_t = float(times.mean())
        faster = times < avg_t
        widths = np.array([g.width for g in population])

        if wcost:
            # relative, deterministic width-cost (self-gating, noise-free)
            fits = raw_fits * (1.0 - WIDTH_COST * widths / HIDDEN)
        else:
            # ignition gate: latch time pressure on once raw accuracy clears noise
            raw_hist.append(float(raw_fits.max()))
            if not time_on and len(raw_hist) >= GATE_WINDOW and \
                    np.mean(raw_hist[-GATE_WINDOW:]) >= GATE_THRESHOLD:
                time_on = True
                print(f"  >> ignition gate opened at gen {gen}: time pressure ON")
            # time pressure: below-average eval time -> small fitness bump
            fits = raw_fits + (faster * TIME_BONUS if time_on else 0.0)
        best_i = int(np.argmax(fits))
        best_fit = float(fits[best_i])
        avg_fit = float(fits.mean())

        print(f"gen {gen:>3} | best {best_fit:>6.1f} | avg {avg_fit:>7.1f} | "
              f"best_raw {int(raw_fits[best_i]):>4} | best_w {int(widths[best_i]):>2} | "
              f"avg_w {widths.mean():>4.1f} | n_fast {int(faster.sum()):>2} | "
              f"avg_t {avg_t * 1e3:>5.2f}ms | "
              f"{'wcost' if wcost else ('T on' if time_on else 'T off')}")

        # save best genome each generation
        with open(os.path.join(BEST_DIR, "best.pkl"), "wb") as f:
            pickle.dump({"genome": population[best_i].params(),
                         "width": int(widths[best_i]),
                         "gen": gen, "fitness": best_fit,
                         "vocab": vocab}, f)

        # every 10 gens, show sample predictions from the best genome
        if gen % 10 == 0:
            sample_idx = order[:5]
            preds = population[best_i].predict(X[sample_idx])
            print("  --- sample predictions (best genome) ---")
            for j, si in enumerate(sample_idx):
                ctx = " ".join(id2word[w] for w in X[si])
                pred_w = id2word[preds[j]]
                true_w = id2word[y[si]]
                mark = "OK " if preds[j] == y[si] else "  x"
                print(f"  {mark} [{ctx}] -> pred '{pred_w}' | actual '{true_w}'")

        # ---- plateau check: smooth best fitness, stop when it stops rising
        best_hist.append(best_fit)
        if len(best_hist) >= PLATEAU_WINDOW:
            recent = float(np.mean(best_hist[-PLATEAU_WINDOW:]))
            if recent > smoothed_best + PLATEAU_EPS:
                smoothed_best = recent
                stall = 0
            else:
                stall += 1
            if stall >= PLATEAU_PATIENCE:
                print(f"\nPlateau reached at gen {gen}: smoothed best fitness "
                      f"stuck at ~{smoothed_best:.1f} for {PLATEAU_PATIENCE} gens.")
                break

        # ---- reproduction: tournament selection, mutation only, no crossover
        new_pop = [population[best_i].copy()]   # elitism: keep the best as-is
        while len(new_pop) < POP_SIZE:
            contestants = rng.integers(0, POP_SIZE, TOURNAMENT_K)
            winner = contestants[np.argmax(fits[contestants])]
            child = population[winner].copy()
            child.mutate(rng)
            new_pop.append(child)
        population = new_pop

    print("\nDone. Best genome saved to", os.path.join(BEST_DIR, "best.pkl"))


if __name__ == "__main__":
    main()
