"""SURP_A: surprisal-weighted fitness (smooth, no extinction cliffs).

RAMP_B showed that hard-stripping a crutch token from a *converged* population
is a mass-extinction event -> everyone hits credit 0 at once -> no gradient ->
no recovery. The cure is to remove the cliffs: weight every correct prediction
by its surprisal -log2(p(word)). Predicting a frequent token right is worth
little; a rare content word, a lot. The energy economy stays normal (so it
ignites to the <unk> level fine), but SELECTION rewards informative correctness,
applying continuous pressure toward context without any extinction events.

Gradient-free. EEC_SURP_POW scales the surprisal exponent (sharpness).
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
SURP_POW = float(os.environ.get("EEC_SURP_POW", "1.0"))   # surprisal sharpness


def evaluate_surp(genome, X, y, order, weight):
    """Normal energy economy (refund on any correct), but fitness =
    sum of weight[token] over correct predictions. Returns
    (weighted_fitness, raw_correct, seen)."""
    energy = START_ENERGY
    wfit = 0.0
    raw = 0
    seen = 0
    pos = 0
    BATCH = 1024
    while energy > 0 and pos < len(order):
        idx = order[pos:pos + BATCH]
        ty = y[idx]
        hits = genome.predict(X[idx]) == ty
        cum = np.cumsum(~hits)
        if cum[-1] < energy:
            raw += int(hits.sum())
            wfit += float(weight[ty[hits]].sum())
            seen += len(idx)
            energy -= int(cum[-1])
            pos += BATCH
        else:
            cut = int(np.searchsorted(cum, energy))
            h = hits[:cut + 1]
            raw += int(h.sum())
            wfit += float(weight[ty[:cut + 1][h]].sum())
            seen += cut + 1
            energy = 0
    return wfit, raw, seen


def main():
    rng = np.random.default_rng(SEED)
    print("Loading corpus...")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences(ids)
    vocab_size = len(vocab)

    counts = np.bincount(y, minlength=vocab_size).astype(np.float64)
    p = counts / counts.sum()
    surprisal = -np.log2(np.where(p > 0, p, 1.0))          # bits; 0 for unseen
    weight = (surprisal ** SURP_POW).astype(np.float64)
    print(f"vocab {vocab_size} | surprisal range {surprisal[counts>0].min():.1f}"
          f"..{surprisal.max():.1f} bits | weight pow {SURP_POW}")
    print("  weight of common tokens: " + ", ".join(
        f"{vocab[i]!r}={weight[i]:.1f}" for i in np.argsort(counts)[::-1][:5]))

    population = [Genome(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(1, HARD_CAP + 1):
        order = rng.permutation(len(X))
        res = [evaluate_surp(g, X, y, order, weight) for g in population]
        wfits = np.array([r[0] for r in res])
        raws = np.array([r[1] for r in res])
        bi = int(np.argmax(wfits))
        # best genome's avg surprisal per correct = how informative its hits are
        bw = wfits[bi]; braw = int(raws[bi])
        bits = bw / braw if braw else 0.0
        uniq = int(np.unique(population[bi].predict(X[order[:512]])).size)
        print(f"gen {gen:>4} | wfit {bw:>8.1f} | raw {braw:>4} | "
              f"bits/hit {bits:>5.2f} | uniq {uniq:>3} | avg_wfit {wfits.mean():>7.1f}")

        with open(os.path.join(BEST_DIR, "surp_best.pkl"), "wb") as f:
            pickle.dump({"genome": population[bi].params(), "gen": gen,
                         "wfit": bw, "raw": braw, "vocab": vocab}, f)

        if gen % 20 == 0:
            si = order[:6]
            preds = population[bi].predict(X[si])
            print("  --- best: ctx -> pred | actual ---")
            for j, s in enumerate(si):
                ctx = " ".join(vocab[w] for w in X[s])
                mark = "OK" if preds[j] == y[s] else " x"
                print(f"  {mark} [{ctx}] -> {vocab[preds[j]]!r} | {vocab[y[s]]!r}")

        new_pop = [population[bi].copy()]
        while len(new_pop) < POP_SIZE:
            c = rng.integers(0, POP_SIZE, TOURNAMENT_K)
            w = c[np.argmax(wfits[c])]
            child = population[w].copy(); child.mutate(rng)
            new_pop.append(child)
        population = new_pop

    print("\nDone.", os.path.join(BEST_DIR, "surp_best.pkl"))


if __name__ == "__main__":
    main()
