"""PMI_A: conditional-information fitness that a constant cannot game.

Every prior gambit failed because its reward was maximized by SOME reachable
constant. The fix is a CONDITIONAL reward: value a correct prediction by how
much the context made it predictable beyond the marginal --
    weight(prev, word) = p(word | prev_word) * surprisal(word)
A constant predictor gets ~0 (its token has no conditional advantage tied to
context); a genome that learns context-determined transitions ("New"->"York",
"don" "'" -> "t") scores hugely. The bigram statistics are pure counts
(gradient-free) and shape ONLY the fitness landscape, never the genome's input.

Energy economy unchanged. Gradient-free.
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


def build_pmi_weight(X, y, vocab_size):
    """Per-window weight = p(y | prev_word) * surprisal(y), from bigram counts."""
    prev = X[:, -1].astype(np.int64)
    nxt = y.astype(np.int64)
    # marginal surprisal
    counts = np.bincount(nxt, minlength=vocab_size).astype(np.float64)
    p = counts / counts.sum()
    surpr = -np.log2(np.where(p > 0, p, 1.0))
    # conditional p(next|prev) via sparse pair counts
    prev_tot = np.bincount(prev, minlength=vocab_size).astype(np.float64)
    pair = prev * vocab_size + nxt
    upair, inv, pcount = np.unique(pair, return_inverse=True, return_counts=True)
    paircount = pcount[inv].astype(np.float64)        # count of each window's pair
    p_cond = paircount / np.maximum(prev_tot[prev], 1.0)
    w = (p_cond * surpr[nxt]).astype(np.float64)      # per-window fitness weight
    return w, surpr


def evaluate_pmi(genome, X, y, order, weight):
    energy = START_ENERGY
    wfit = 0.0; raw = 0; seen = 0; pos = 0; BATCH = 1024
    while energy > 0 and pos < len(order):
        idx = order[pos:pos + BATCH]
        hits = genome.predict(X[idx]) == y[idx]
        cum = np.cumsum(~hits)
        if cum[-1] < energy:
            raw += int(hits.sum())
            wfit += float(weight[idx][hits].sum())
            seen += len(idx); energy -= int(cum[-1]); pos += BATCH
        else:
            cut = int(np.searchsorted(cum, energy)); h = hits[:cut + 1]
            raw += int(h.sum()); wfit += float(weight[idx[:cut + 1]][h].sum())
            seen += cut + 1; energy = 0
    return wfit, raw, seen


def main():
    rng = np.random.default_rng(SEED)
    print("Loading corpus...")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences(ids)
    vocab_size = len(vocab)
    weight, surpr = build_pmi_weight(X, y, vocab_size)
    print(f"vocab {vocab_size} | PMI weight range {weight.min():.2f}..{weight.max():.2f} "
          f"| mean {weight.mean():.3f}")

    population = [Genome(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(1, HARD_CAP + 1):
        order = rng.permutation(len(X))
        res = [evaluate_pmi(g, X, y, order, weight) for g in population]
        wfits = np.array([r[0] for r in res]); raws = np.array([r[1] for r in res])
        bi = int(np.argmax(wfits))
        uniq = int(np.unique(population[bi].predict(X[order[:512]])).size)
        print(f"gen {gen:>4} | pmiFit {wfits[bi]:>8.1f} | raw {int(raws[bi]):>4} | "
              f"uniq {uniq:>3} | avgFit {wfits.mean():>7.1f}")

        with open(os.path.join(BEST_DIR, "pmi_best.pkl"), "wb") as f:
            pickle.dump({"genome": population[bi].params(), "gen": gen,
                         "pmifit": float(wfits[bi]), "raw": int(raws[bi]),
                         "vocab": vocab}, f)

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
            wn = c[np.argmax(wfits[c])]
            child = population[wn].copy(); child.mutate(rng)
            new_pop.append(child)
        population = new_pop
    print("\nDone.", os.path.join(BEST_DIR, "pmi_best.pkl"))


if __name__ == "__main__":
    main()
