"""PERC_A: perception cost as a law of existence (not reward, not architecture).

The organism does NOT get the context for free. It has an evolvable perception
gate p_i = sigmoid(g_i) per context position -- how hard it looks there. The MLP
only sees gated embeddings p_i * E[word_i]; a position at p_i~0 is unseen. Every
prediction drains  kappa * sum_i p_i  energy from the SAME pool, on top of the
existing economy (wrong -1, correct refunds). Death at 0, fitness = correct.

Nothing references the next word. Attention is never rewarded -- it must EMERGE,
because the blind default (p=0) can't cut its error rate and dies, while seeing
everything is too expensive. Survival forces SELECTIVE context use.

Pairs with the existing energy/time constraints; memory-rent, resource-depletion
and metabolic-upkeep are the next laws in the pocket.

EEC_L = context length, EEC_KAPPA = perception price.
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import pickle
import numpy as np

from evolve import (EMBED, HIDDEN, MUT_RATE, MUT_SCALE, POP_SIZE, TOURNAMENT_K,
                    START_ENERGY, HARD_CAP, build_corpus)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_DIR = os.path.join(HERE, "best")
SEED = 0
L = int(os.environ.get("EEC_L", "8"))            # context window length
KAPPA = float(os.environ.get("EEC_KAPPA", "0.05"))   # energy per unit perception


def make_sequences_L(ids):
    n = len(ids) - L
    X = np.empty((n, L), dtype=np.int32)
    for i in range(L):
        X[:, i] = ids[i:i + n]
    return X, ids[L:L + n]


class PercGenome:
    def __init__(self, vocab_size, rng):
        s = 1.0 / np.sqrt(EMBED)
        self.E = rng.normal(0, 0.1, (vocab_size, EMBED)).astype(np.float32)
        # perception logits -> p=sigmoid(g). Seed with spread: relative mutation
        # can't move a zero-init gate (calcification), so start them differentiated.
        self.g = rng.normal(0, 1.5, L).astype(np.float32)
        self.W1 = rng.normal(0, s, (L * EMBED, HIDDEN)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = rng.normal(0, 1.0/np.sqrt(HIDDEN), (HIDDEN, vocab_size)).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)

    def perception(self):
        return 1.0 / (1.0 + np.exp(-self.g))                   # p_i in [0,1]

    def predict(self, ctx_rows):
        p = self.perception().astype(np.float32)               # (L,)
        emb = self.E[ctx_rows] * p[None, :, None]              # (B, L, EMBED) gated
        emb = emb.reshape(ctx_rows.shape[0], -1)
        h = np.tanh(emb @ self.W1 + self.b1)
        return np.argmax(h @ self.W2 + self.b2, axis=1)

    def params(self):
        return [self.E, self.g, self.W1, self.b1, self.W2, self.b2]

    def copy(self):
        g = PercGenome.__new__(PercGenome)
        g.E = self.E.copy(); g.g = self.g.copy(); g.W1 = self.W1.copy()
        g.b1 = self.b1.copy(); g.W2 = self.W2.copy(); g.b2 = self.b2.copy()
        return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            step = rng.normal(0, 1, p.shape).astype(np.float32)
            p += mask * step * (MUT_SCALE * (np.abs(p) + 1e-3))


def evaluate_perc(genome, X, y, order):
    """Energy economy + perception drain. Each prediction costs kappa*sum(p)
    always, plus 1 if wrong. Death when cumulative cost reaches START_ENERGY."""
    load = float(genome.perception().sum())            # total perception per look
    per_look = KAPPA * load
    correct = 0; seen = 0; pos = 0; BATCH = 1024
    budget = float(START_ENERGY)
    spent = 0.0
    while spent < budget and pos < len(order):
        idx = order[pos:pos + BATCH]
        hits = genome.predict(X[idx]) == y[idx]
        # cost per prediction: per_look always, +1 if wrong
        step_cost = per_look + (~hits)
        cum = np.cumsum(step_cost) + spent
        if cum[-1] < budget:
            correct += int(hits.sum()); seen += len(idx); spent = float(cum[-1]); pos += BATCH
        else:
            cut = int(np.searchsorted(cum, budget))    # first prediction that exhausts budget
            correct += int(hits[:cut + 1].sum()); seen += cut + 1; spent = budget
    return correct, seen, load


def main():
    rng = np.random.default_rng(SEED)
    print(f"Loading corpus... (L={L}, kappa={KAPPA})")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences_L(ids)
    vocab_size = len(vocab)
    unigram = np.bincount(y, minlength=vocab_size).max() / len(y)
    print(f"vocab {vocab_size} | windows {len(y):,} | unigram ceiling {100*unigram:.2f}% "
          f"| full-perception drain = {KAPPA*L:.2f}/pred")

    population = [PercGenome(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(1, HARD_CAP + 1):
        order = rng.permutation(len(X))
        res = [evaluate_perc(g, X, y, order) for g in population]
        fits = np.array([r[0] for r in res]); seens = np.array([r[1] for r in res])
        loads = np.array([r[2] for r in res])
        bi = int(np.argmax(fits)); bf = int(fits[bi]); bseen = int(seens[bi])
        acc = bf / bseen if bseen else 0.0
        uniq = int(np.unique(population[bi].predict(X[order[:512]])).size)
        pbest = population[bi].perception()
        print(f"gen {gen:>4} | best {bf:>5} | acc {100*acc:>5.2f}% | "
              f"vs_uni {100*(acc-unigram):>+5.2f}pp | load {loads[bi]:>4.1f}/{L} | "
              f"uniq {uniq:>3} | perc[" + " ".join(f"{v:.1f}" for v in pbest) + "]")

        with open(os.path.join(BEST_DIR, "perc_best.pkl"), "wb") as f:
            pickle.dump({"genome": population[bi].params(), "gen": gen, "acc": acc,
                         "perception": pbest, "vocab": vocab}, f)

        if gen % 20 == 0:
            si = order[:6]; preds = population[bi].predict(X[si])
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
    print("\nDone.", os.path.join(BEST_DIR, "perc_best.pkl"))


if __name__ == "__main__":
    main()
