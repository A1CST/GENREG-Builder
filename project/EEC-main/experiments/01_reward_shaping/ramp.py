"""RAMP_A: ramped rarity-gated energy to break the <unk> unigram ceiling.

The ceiling is a property of the FITNESS LANDSCAPE, not the architecture: a
constant predictor of the single most-frequent token is a stable attractor.
This removes the attractor progressively. A correct prediction only refunds
energy / scores fitness if the target word is NOT in a "trivial" set. That set
starts empty (so it ignites to the <unk> ceiling exactly like the baseline),
and every time the population plateaus we add the current crutch -- the most
frequent token still earning credit -- to the trivial set. Each removal raises
the energy floor one notch and forces the population onto the next tier of
structure. Gradient-free; pure environment construction.
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import pickle
import numpy as np

import evolve as E
from evolve import (POP_SIZE, TOURNAMENT_K, START_ENERGY, HARD_CAP, MUT_RATE,
                    MUT_SCALE, Genome, build_corpus, make_sequences)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_DIR = os.path.join(HERE, "best")
SEED = 0
# Fixed-cadence ramp (robust to per-gen shuffle noise): after an ignition
# window, strip the next crutch every RAMP_EVERY gens, but only while the
# population is still scoring (starvation guard) so we don't ramp into collapse.
RAMP_START = 80        # gens to ignite before any ramping
RAMP_EVERY = 50        # gens between crutch removals
STARVE_GUARD = 3       # hold the ramp if best credit drops below this
MAX_TRIVIAL = int(os.environ.get("EEC_MAXTRIV", "15"))


def evaluate_ramp(genome, X, y, order, trivial_mask):
    """Energy round where only NON-trivial correct predictions earn credit.

    Returns (credit, raw_correct, seen). Energy decrements on every prediction
    that isn't a non-trivial hit (wrong OR trivial-correct), so a constant
    predictor of a now-trivial token starves. Fitness = credit.
    """
    energy = START_ENERGY
    credit = 0
    raw_correct = 0
    seen = 0
    pos = 0
    BATCH = 1024
    while energy > 0 and pos < len(order):
        idx = order[pos:pos + BATCH]
        ty = y[idx]
        hits = genome.predict(X[idx]) == ty
        cred = hits & ~trivial_mask[ty]          # correct AND target non-trivial
        cum = np.cumsum(~cred)                    # energy burns on non-credit
        if cum[-1] < energy:
            raw_correct += int(hits.sum())
            credit += int(cred.sum())
            seen += len(idx)
            energy -= int(cum[-1])
            pos += BATCH
        else:
            cut = int(np.searchsorted(cum, energy))
            raw_correct += int(hits[:cut + 1].sum())
            credit += int(cred[:cut + 1].sum())
            seen += cut + 1
            energy = 0
    return credit, raw_correct, seen


def main():
    rng = np.random.default_rng(SEED)
    print("Loading corpus...")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences(ids)
    vocab_size = len(vocab)

    # frequency rank over the actual training targets (most frequent first)
    counts = np.bincount(y, minlength=vocab_size)
    freq_order = np.argsort(counts)[::-1]         # token ids, most frequent first
    trivial_mask = np.zeros(vocab_size, dtype=bool)
    T = 0                                         # how many crutches removed
    print(f"vocab {vocab_size} | top tokens: "
          + ", ".join(f"{vocab[freq_order[i]]!r}" for i in range(6)))

    population = [Genome(vocab_size, rng) for _ in range(POP_SIZE)]

    since_ramp = 0
    for gen in range(1, HARD_CAP + 1):
        order = rng.permutation(len(X))
        res = [evaluate_ramp(g, X, y, order, trivial_mask) for g in population]
        creds = np.array([r[0] for r in res])
        raws = np.array([r[1] for r in res])
        bi = int(np.argmax(creds))
        bc = int(creds[bi]); braw = int(raws[bi])
        # diversity of the best genome's predictions (detect constant-collapse)
        uniq = int(np.unique(population[bi].predict(X[order[:512]])).size)
        nextcrutch = vocab[freq_order[T]] if T < MAX_TRIVIAL else "-"
        print(f"gen {gen:>4} | credit {bc:>5} | avg {creds.mean():>7.1f} | "
              f"raw {braw:>5} | uniq {uniq:>3} | T {T:>2} (next {nextcrutch!r})")

        with open(os.path.join(BEST_DIR, "ramp_best.pkl"), "wb") as f:
            pickle.dump({"genome": population[bi].params(), "gen": gen,
                         "credit": bc, "raw": braw, "T": T, "vocab": vocab}, f)

        if gen % 20 == 0:
            si = order[:6]
            preds = population[bi].predict(X[si])
            print("  --- best: ctx -> pred | actual  (* = non-trivial hit) ---")
            for j, s in enumerate(si):
                ctx = " ".join(vocab[w] for w in X[s])
                hit = preds[j] == y[s]
                star = "*" if hit and not trivial_mask[y[s]] else (" " if hit else "x")
                print(f"  {star} [{ctx}] -> {vocab[preds[j]]!r} | {vocab[y[s]]!r}")

        # fixed-cadence ramp with starvation guard
        since_ramp += 1
        if gen >= RAMP_START and since_ramp >= RAMP_EVERY and T < MAX_TRIVIAL:
            if bc >= STARVE_GUARD:
                trivial_mask[freq_order[T]] = True
                T += 1
                since_ramp = 0
                print(f"  >> RAMP gen {gen}: stripped {vocab[freq_order[T-1]]!r} "
                      f"from credit (T={T}); floor raised.")
            else:
                print(f"  .. ramp held at gen {gen}: starved (best credit {bc}).")

        new_pop = [population[bi].copy()]
        while len(new_pop) < POP_SIZE:
            c = rng.integers(0, POP_SIZE, TOURNAMENT_K)
            w = c[np.argmax(creds[c])]
            child = population[w].copy(); child.mutate(rng)
            new_pop.append(child)
        population = new_pop

    print("\nDone. Best saved to", os.path.join(BEST_DIR, "ramp_best.pkl"))


if __name__ == "__main__":
    main()
