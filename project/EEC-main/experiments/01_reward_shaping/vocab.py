"""VOCAB_A: shrink the output space and kill the OOV crutch.

Every reward-shaping gambit failed on the same wall: from a constant attractor,
mutation can't reach context-specific outputs in an 8000-way space. This attacks
the SEARCH problem directly -- small vocab (tractable output) AND drop every
window whose target is <unk> (no dominant OOV constant to hide behind). The
ceiling to beat is the single most-frequent REAL-word target rate; exceeding it
means context is actually being used. Base energy economy, gradient-free.

EEC_VOCAB sets vocab size (default 1000 here).
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import pickle
import numpy as np

os.environ.setdefault("EEC_VOCAB", "1000")
from evolve import (POP_SIZE, TOURNAMENT_K, HARD_CAP, Genome, evaluate,
                    build_corpus, make_sequences)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_DIR = os.path.join(HERE, "best")
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    print(f"Loading corpus (vocab={os.environ['EEC_VOCAB']})...")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences(ids)
    vocab_size = len(vocab)

    keep = y != 0                       # drop OOV (<unk>, id 0) targets
    X, y = X[keep], y[keep]
    counts = np.bincount(y, minlength=vocab_size)
    top_id = int(np.argmax(counts))
    ceiling = counts[top_id] / len(y)
    print(f"vocab {vocab_size} | non-OOV windows {len(y):,} | "
          f"best constant = {vocab[top_id]!r} @ {100*ceiling:.2f}%  (the ceiling to beat)")

    population = [Genome(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(1, HARD_CAP + 1):
        order = rng.permutation(len(X))
        res = [evaluate(g, X, y, order) for g in population]
        fits = np.array([r[0] for r in res]); seens = np.array([r[2] for r in res])
        bi = int(np.argmax(fits))
        bf = int(fits[bi]); bseen = int(seens[bi])
        acc = bf / bseen if bseen else 0.0
        uniq = int(np.unique(population[bi].predict(X[order[:512]])).size)
        print(f"gen {gen:>4} | best {bf:>5} | acc {100*acc:>5.2f}% | "
              f"vs_ceil {100*(acc-ceiling):>+5.2f}pp | uniq {uniq:>3} | "
              f"seen {bseen:>5} | avg {fits.mean():>6.1f}")

        with open(os.path.join(BEST_DIR, "vocab_best.pkl"), "wb") as f:
            pickle.dump({"genome": population[bi].params(), "gen": gen,
                         "acc": acc, "vocab": vocab}, f)

        if gen % 20 == 0:
            si = order[:8]; preds = population[bi].predict(X[si])
            print("  --- best: ctx -> pred | actual ---")
            for j, s in enumerate(si):
                ctx = " ".join(vocab[w] for w in X[s])
                mark = "OK" if preds[j] == y[s] else " x"
                print(f"  {mark} [{ctx}] -> {vocab[preds[j]]!r} | {vocab[y[s]]!r}")

        new_pop = [population[bi].copy()]
        while len(new_pop) < POP_SIZE:
            c = rng.integers(0, POP_SIZE, TOURNAMENT_K)
            wn = c[np.argmax(fits[c])]
            child = population[wn].copy(); child.mutate(rng)
            new_pop.append(child)
        population = new_pop
    print("\nDone.", os.path.join(BEST_DIR, "vocab_best.pkl"))


if __name__ == "__main__":
    main()
