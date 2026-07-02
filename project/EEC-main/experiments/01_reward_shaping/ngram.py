"""NGRAM_A: make context REACHABLE by feeding an n-gram feature.

11 experiments showed the wall is search, not reward: a randomly-mutated
NN->softmax always converges to the best reachable CONSTANT and never bridges to
context. The constructive fix (matching the whole working LM lineage: frozen
n-gram channels + evolved mixing) is to give the genome a context-informative
feature it can learn to USE. Here: append the bigram's predicted next-token as a
4th input token. Evolution can learn to trust/refine it -> instantly above the
unigram ceiling. The bigram table is pure counts (gradient-free).

Ceiling to beat: the unigram constant. Reference: raw bigram-copy accuracy.
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import pickle
import numpy as np

from evolve import (CONTEXT, EMBED, HIDDEN, MIN_WIDTH, MUT_RATE, MUT_SCALE,
                    POP_SIZE, TOURNAMENT_K, HARD_CAP, evaluate,
                    build_corpus, make_sequences)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_DIR = os.path.join(HERE, "best")
SEED = 0
NTOK = CONTEXT + 1            # context tokens + 1 bigram-feature token
# EEC_CHANNEL=1 (NGRAM_B): add the bigram prediction as a DIRECT additive logit
# channel with an evolved scalar mix, instead of as an input feature the MLP must
# learn to route. The mix is one searchable parameter -> breaks the ceiling fast.
CHANNEL = bool(os.environ.get("EEC_CHANNEL"))


def bigram_best_next(prev, nxt, vocab_size):
    """best_next[p] = most frequent token following p (argmax bigram)."""
    pair = prev.astype(np.int64) * vocab_size + nxt.astype(np.int64)
    upair, pc = np.unique(pair, return_counts=True)
    pp = (upair // vocab_size).astype(np.int64)
    nn = (upair % vocab_size).astype(np.int64)
    order = np.lexsort((pc, pp))               # by prev, then count asc
    best = np.zeros(vocab_size, dtype=np.int32)
    best[pp[order]] = nn[order]                # last write per prev = max count
    return best


class NGGenome:
    """Single hidden layer over NTOK input tokens (context + bigram feature)."""

    def __init__(self, vocab_size, rng):
        s = 1.0 / np.sqrt(EMBED)
        mlp_tokens = CONTEXT if CHANNEL else NTOK     # channel mode: MLP sees only context
        self.E = rng.normal(0, 0.1, (vocab_size, EMBED)).astype(np.float32)
        self.W1 = rng.normal(0, s, (mlp_tokens * EMBED, HIDDEN)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = rng.normal(0, 1.0/np.sqrt(HIDDEN), (HIDDEN, vocab_size)).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)
        self.width = HIDDEN
        # evolved scalar mix for the bigram logit channel (NGRAM_B)
        self.mix = float(abs(rng.normal(0, 1.0))) if CHANNEL else 0.0

    def predict(self, ctx_rows):
        # CHANNEL mode: last column is the bigram prediction (direct channel);
        # otherwise it's just another input token to the MLP.
        if CHANNEL:
            bg = ctx_rows[:, -1]
            ctx = ctx_rows[:, :CONTEXT]
        else:
            ctx = ctx_rows
        emb = self.E[ctx].reshape(ctx.shape[0], -1)
        h = np.tanh(emb @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        if CHANNEL:
            logits[np.arange(ctx.shape[0]), bg] += self.mix     # additive n-gram channel
        return np.argmax(logits, axis=1)

    def params(self):
        return [self.E, self.W1, self.b1, self.W2, self.b2]

    def copy(self):
        g = NGGenome.__new__(NGGenome)
        g.E = self.E.copy(); g.W1 = self.W1.copy(); g.b1 = self.b1.copy()
        g.W2 = self.W2.copy(); g.b2 = self.b2.copy(); g.width = self.width
        g.mix = self.mix
        return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            step = rng.normal(0, 1, p.shape).astype(np.float32)
            p += mask * step * (MUT_SCALE * (np.abs(p) + 1e-3))
        if CHANNEL:                                              # relative mix mutation
            self.mix = max(0.0, self.mix + rng.normal(0, MUT_SCALE * (abs(self.mix) + 0.1)))


def main():
    rng = np.random.default_rng(SEED)
    print("Loading corpus...")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences(ids)
    vocab_size = len(vocab)

    best_next = bigram_best_next(X[:, -1], y, vocab_size)
    bfeat = best_next[X[:, -1]]                      # bigram prediction per window
    Xaug = np.concatenate([X, bfeat[:, None]], axis=1).astype(np.int32)
    bigram_acc = (bfeat == y).mean()
    unigram_acc = np.bincount(y, minlength=vocab_size).max() / len(y)
    print(f"vocab {vocab_size} | unigram ceiling {100*unigram_acc:.2f}% | "
          f"raw bigram-copy {100*bigram_acc:.2f}% (the signal now available)")

    population = [NGGenome(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(1, HARD_CAP + 1):
        order = rng.permutation(len(Xaug))
        res = [evaluate(g, Xaug, y, order) for g in population]
        fits = np.array([r[0] for r in res]); seens = np.array([r[2] for r in res])
        bi = int(np.argmax(fits)); bf = int(fits[bi]); bseen = int(seens[bi])
        acc = bf / bseen if bseen else 0.0
        uniq = int(np.unique(population[bi].predict(Xaug[order[:512]])).size)
        print(f"gen {gen:>4} | best {bf:>5} | acc {100*acc:>5.2f}% | "
              f"vs_unigram {100*(acc-unigram_acc):>+5.2f}pp | uniq {uniq:>3} | "
              f"mix {population[bi].mix:>5.2f} | seen {bseen:>5}")

        with open(os.path.join(BEST_DIR, "ngram_best.pkl"), "wb") as f:
            pickle.dump({"genome": population[bi].params(), "gen": gen,
                         "acc": acc, "vocab": vocab}, f)

        if gen % 20 == 0:
            si = order[:8]; preds = population[bi].predict(Xaug[si])
            print("  --- best: ctx +bigram -> pred | actual ---")
            for j, s in enumerate(si):
                ctx = " ".join(vocab[w] for w in Xaug[s][:CONTEXT])
                bg = vocab[Xaug[s][-1]]
                mark = "OK" if preds[j] == y[s] else " x"
                print(f"  {mark} [{ctx} |bg:{bg!r}] -> {vocab[preds[j]]!r} | {vocab[y[s]]!r}")

        new_pop = [population[bi].copy()]
        while len(new_pop) < POP_SIZE:
            c = rng.integers(0, POP_SIZE, TOURNAMENT_K)
            wn = c[np.argmax(fits[c])]
            child = population[wn].copy(); child.mutate(rng)
            new_pop.append(child)
        population = new_pop
    print("\nDone.", os.path.join(BEST_DIR, "ngram_best.pkl"))


if __name__ == "__main__":
    main()
