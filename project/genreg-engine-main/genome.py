"""
Piece 1: the Genome.

The irreducible core — what a single individual IS and how it computes. No mutation, no evolution, no
constraints yet; those are later pieces, kept separate so nothing is hidden inside this one.

A genome is a feedforward network:  n_in inputs -> H tanh hidden units -> n_out linear outputs.
Weights are float32. The forward pass is pure and deterministic: same genome + same input -> same output.
"""
import numpy as np


class Genome:
    def __init__(self, n_in, n_out, H, rng):
        """Initialize with small random weights (He/Xavier-style scaling) so activations start in a sane range.
        `rng` is a numpy Generator — all randomness flows through it, so runs are reproducible."""
        self.n_in, self.n_out, self.H = int(n_in), int(n_out), int(H)
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(self.n_in), (self.n_in, self.H)).astype(np.float32)
        self.b1 = np.zeros(self.H, np.float32)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(self.H), (self.H, self.n_out)).astype(np.float32)
        self.b2 = np.zeros(self.n_out, np.float32)

    PARAMS = ("W1", "b1", "W2", "b2")               # the learnable arrays (later pieces iterate over these)

    def forward(self, X):
        """X: (batch, n_in) -> (batch, n_out). tanh hidden, linear output. No gradients are ever computed."""
        X = np.asarray(X, np.float32)
        if X.ndim == 1:
            X = X[None, :]
        h = np.tanh(X @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def n_params(self):
        return sum(getattr(self, p).size for p in self.PARAMS)

    def __repr__(self):
        return f"Genome({self.n_in}->{self.H}->{self.n_out}, {self.n_params()} params)"


# --- self-test: run `python3 genome.py` ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g = Genome(n_in=3, n_out=2, H=8, rng=rng)

    # shape + determinism
    X = rng.normal(size=(5, 3)).astype(np.float32)
    Y1 = g.forward(X)
    Y2 = g.forward(X)
    assert Y1.shape == (5, 2), Y1.shape
    assert np.array_equal(Y1, Y2), "forward pass must be deterministic"

    # 1-D input is accepted and batched
    assert g.forward(X[0]).shape == (1, 2)

    # reproducibility: same seed -> identical weights
    g2 = Genome(3, 2, 8, np.random.default_rng(0))
    assert np.array_equal(g.W1, g2.W1) and np.array_equal(g.W2, g2.W2), "same seed must give same genome"

    print(g)
    print(f"forward {X.shape} -> {Y1.shape}, deterministic OK, reproducible OK")
