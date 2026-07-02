"""NICHE_A: fitness sharing to prevent monoculture and reward context.

Every prior run monocultured onto one constant, after which reward tweaks just
moved the monoculture. Fitness sharing attacks that: the reward for a correctly
predicted token instance is SPLIT among all genomes that get it. So 30 genomes
all predicting <unk> each receive 1/30 of each <unk> reward -- crowding is
near-worthless -- while a genome that UNIQUELY predicts a word in its context
keeps the full reward. This (a) preserves diversity and (b) rewards covering
under-served contexts, which requires actually using the context.

Energy still gates lifespan (accurate genomes process more of the shared stream).
Gradient-free. EEC_NICHE_W=surp weights the shared reward by surprisal.
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import pickle
import numpy as np

from evolve import (POP_SIZE, TOURNAMENT_K, START_ENERGY, HARD_CAP,
                    Genome, build_corpus, make_sequences)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_DIR = os.path.join(HERE, "best")
SEED = 0
SHARE_S = 2000         # length of the shared evaluation stream per generation
USE_SURP = os.environ.get("EEC_NICHE_W") == "surp"


def lived_correctness(genome, Xs, ys):
    """Correctness over the shared stream, zeroed after energy death
    (energy = START_ENERGY, dies at START_ENERGY wrongs)."""
    hits = genome.predict(Xs) == ys                       # (S,)
    cum = np.cumsum(~hits)
    if cum[-1] >= START_ENERGY:                            # died within the stream
        d = int(np.searchsorted(cum, START_ENERGY)) + 1
        hits[d:] = False
    return hits


def main():
    rng = np.random.default_rng(SEED)
    print("Loading corpus...")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences(ids)
    vocab_size = len(vocab)

    counts = np.bincount(y, minlength=vocab_size).astype(np.float64)
    p = counts / counts.sum()
    surprisal = -np.log2(np.where(p > 0, p, 1.0))
    print(f"vocab {vocab_size} | fitness sharing | weight={'surprisal' if USE_SURP else '1'}")

    population = [Genome(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(1, HARD_CAP + 1):
        order = rng.permutation(len(X))[:SHARE_S]
        Xs, ys = X[order], y[order]
        w = surprisal[ys] if USE_SURP else np.ones(SHARE_S)

        C = np.stack([lived_correctness(g, Xs, ys) for g in population])  # (G,S)
        k = C.sum(0)                                       # genomes correct per window
        share = np.where(k > 0, w / np.maximum(k, 1), 0.0)
        fits = (C * share).sum(1)                          # shared fitness per genome
        raws = C.sum(1)                                    # raw correct (lived)

        bi = int(np.argmax(fits))
        # population diversity: how many DISTINCT windows the pop covers at all
        covered = int((k > 0).sum())
        uniq = int(np.unique(population[bi].predict(Xs[:512])).size)
        print(f"gen {gen:>4} | shFit {fits[bi]:>7.1f} | raw {int(raws[bi]):>4} | "
              f"covered {covered:>4}/{SHARE_S} | uniq {uniq:>3} | avgRaw {raws.mean():>6.1f}")

        with open(os.path.join(BEST_DIR, "niche_best.pkl"), "wb") as f:
            pickle.dump({"genome": population[bi].params(), "gen": gen,
                         "shfit": float(fits[bi]), "raw": int(raws[bi]),
                         "vocab": vocab}, f)

        if gen % 20 == 0:
            preds = population[bi].predict(Xs[:6])
            print("  --- best: ctx -> pred | actual ---")
            for j in range(6):
                ctx = " ".join(vocab[wd] for wd in Xs[j])
                mark = "OK" if preds[j] == ys[j] else " x"
                print(f"  {mark} [{ctx}] -> {vocab[preds[j]]!r} | {vocab[ys[j]]!r}")

        new_pop = [population[bi].copy()]
        while len(new_pop) < POP_SIZE:
            c = rng.integers(0, POP_SIZE, TOURNAMENT_K)
            wn = c[np.argmax(fits[c])]
            child = population[wn].copy(); child.mutate(rng)
            new_pop.append(child)
        population = new_pop

    print("\nDone.", os.path.join(BEST_DIR, "niche_best.pkl"))


if __name__ == "__main__":
    main()
