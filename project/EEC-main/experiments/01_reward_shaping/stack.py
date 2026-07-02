"""Residual depth-stacking to break the <unk> unigram ceiling.

Freeze a <unk>-plateaued genome as layer-1. Evolve a layer-2 that sees layer-1's
frozen features ([context-embedding ; h1]) and emits a RESIDUAL added to
layer-1's logits: final = logits1 + logits2. Layer-2's output weights start near
zero, so the stack begins exactly at layer-1's accuracy (the energy floor) and
can only gain energy by fixing layer-1's misses -- which requires using context.

Gradient-free: relative mutation + the energy economy, reused from evolve.py.
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import pickle
import numpy as np

import evolve as E
from evolve import (CONTEXT, EMBED, HIDDEN, MUT_RATE, MUT_SCALE, POP_SIZE,
                    TOURNAMENT_K, START_ENERGY, HARD_CAP, PLATEAU_WINDOW,
                    PLATEAU_PATIENCE, PLATEAU_EPS, Genome, evaluate,
                    build_corpus, make_sequences)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_DIR = os.path.join(HERE, "best")
LAYER1_CKPT = os.path.join(BEST_DIR, "best_noTime_plateau.pkl")
HIDDEN2 = 64           # layer-2 hidden size
# layer-2 output init scale (relative to 1/sqrt(HIDDEN2)). Near-zero (STACK_A)
# made L2 inert; a real magnitude makes it selectable at the cost of starting
# below the floor. EEC_L2SCALE overrides.
L2_OUT_SCALE = float(os.environ.get("EEC_L2SCALE", "1.0"))
# EEC_NOSKIP=1: layer-2 produces final logits from frozen features [emb; h1]
# WITHOUT adding layer-1's saturated logits. Feature-reuse depth (the L1 output
# spike of ~1900 can't be residual-corrected; its hidden h1 is still context-rich).
NOSKIP = bool(os.environ.get("EEC_NOSKIP"))
SEED = 0


def load_frozen_layer1():
    with open(LAYER1_CKPT, "rb") as f:
        saved = pickle.load(f)
    g = Genome.from_params(saved["genome"], saved.get("width", HIDDEN))
    return g, saved.get("fitness", "?")


class StackGenome:
    """Frozen layer-1 (shared, never mutated) + evolvable residual layer-2."""

    def __init__(self, frozen, vocab_size, rng):
        self.f = frozen                      # frozen Genome (read-only, shared)
        d_in = CONTEXT * EMBED + frozen.width
        self.W1 = rng.normal(0, 1.0 / np.sqrt(d_in), (d_in, HIDDEN2)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN2, dtype=np.float32)
        # output init: full magnitude * L2_OUT_SCALE so the residual can flip
        # argmaxes (selectable). STACK_A used ~1e-3 here and went inert.
        o_scale = L2_OUT_SCALE / np.sqrt(HIDDEN2)
        self.W2 = rng.normal(0, o_scale, (HIDDEN2, vocab_size)).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)

    def _frozen_fwd(self, ctx_rows):
        w = self.f.width
        emb = self.f.E[ctx_rows].reshape(ctx_rows.shape[0], -1)   # (B, CONTEXT*EMBED)
        h1 = np.tanh(emb @ self.f.W1[:, :w] + self.f.b1[:w])      # (B, w)
        logits1 = h1 @ self.f.W2[:w, :] + self.f.b2               # (B, vocab)
        return emb, h1, logits1

    def predict(self, ctx_rows):
        emb, h1, logits1 = self._frozen_fwd(ctx_rows)
        inp = np.concatenate([emb, h1], axis=1)                   # (B, d_in)
        h2 = np.tanh(inp @ self.W1 + self.b1)                     # (B, HIDDEN2)
        logits2 = h2 @ self.W2 + self.b2                          # (B, vocab)
        final = logits2 if NOSKIP else logits1 + logits2          # feature-reuse vs residual
        return np.argmax(final, axis=1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def copy(self):
        g = StackGenome.__new__(StackGenome)
        g.f = self.f                          # share frozen layer-1
        g.W1 = self.W1.copy(); g.b1 = self.b1.copy()
        g.W2 = self.W2.copy(); g.b2 = self.b2.copy()
        return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            step = rng.normal(0, 1, p.shape).astype(np.float32)
            scale = MUT_SCALE * (np.abs(p) + 1e-3)
            p += mask * step * scale


def main():
    rng = np.random.default_rng(SEED)
    print("Loading corpus...")
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences(ids)
    vocab_size = len(vocab)

    frozen, l1_fit = load_frozen_layer1()
    print(f"Frozen layer-1 loaded (plateau fitness {l1_fit}, width {frozen.width}). "
          f"vocab {vocab_size}")

    # sanity: layer-1's own accuracy = the floor the stack starts at
    order0 = rng.permutation(len(X))
    l1_correct, _, _ = evaluate(frozen, X, y, order0)
    print(f"Layer-1 solo fitness this round: {l1_correct}  (<-- the ceiling to break)")

    population = [StackGenome(frozen, vocab_size, rng) for _ in range(POP_SIZE)]

    best_hist = []
    smoothed_best = -1.0
    stall = 0
    for gen in range(1, HARD_CAP + 1):
        order = rng.permutation(len(X))
        results = [evaluate(g, X, y, order) for g in population]
        fits = np.array([r[0] for r in results])
        seens = np.array([r[2] for r in results])

        best_i = int(np.argmax(fits))
        best_fit = int(fits[best_i])
        avg_fit = float(fits.mean())
        print(f"gen {gen:>3} | best {best_fit:>5} | avg {avg_fit:>7.1f} | "
              f"best_seen {int(seens[best_i]):>6} | vs_floor {best_fit - l1_correct:+d}")

        with open(os.path.join(BEST_DIR, "stack_best.pkl"), "wb") as f:
            pickle.dump({"layer2": population[best_i].params(), "gen": gen,
                         "fitness": best_fit, "vocab": vocab}, f)

        if gen % 10 == 0:
            sample_idx = order[:6]
            preds = population[best_i].predict(X[sample_idx])
            fpreds = frozen.predict(X[sample_idx])
            print("  --- stack vs frozen layer-1 (ctx -> stack | L1 | actual) ---")
            for j, si in enumerate(sample_idx):
                ctx = " ".join(vocab[w] for w in X[si])
                mark = "OK " if preds[j] == y[si] else "  x"
                print(f"  {mark} [{ctx}] -> '{vocab[preds[j]]}' | "
                      f"L1 '{vocab[fpreds[j]]}' | actual '{vocab[y[si]]}'")

        best_hist.append(best_fit)
        if len(best_hist) >= PLATEAU_WINDOW:
            recent = float(np.mean(best_hist[-PLATEAU_WINDOW:]))
            if recent > smoothed_best + PLATEAU_EPS:
                smoothed_best = recent; stall = 0
            else:
                stall += 1
            if stall >= PLATEAU_PATIENCE:
                print(f"\nPlateau at gen {gen}: smoothed best ~{smoothed_best:.1f}.")
                break

        new_pop = [population[best_i].copy()]
        while len(new_pop) < POP_SIZE:
            c = rng.integers(0, POP_SIZE, TOURNAMENT_K)
            winner = c[np.argmax(fits[c])]
            child = population[winner].copy()
            child.mutate(rng)
            new_pop.append(child)
        population = new_pop

    print("\nDone. Best stack (layer-2) saved to", os.path.join(BEST_DIR, "stack_best.pkl"))


if __name__ == "__main__":
    main()
