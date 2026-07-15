"""CIFAR internal-language v2: the HARD pair — cat (3) vs dog (5). No color/
shape shortcut; tests whether a single end-to-end genome can still evolve an
internal language that separates two visually-entangled animal classes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_internal as ci

if __name__ == "__main__":
    ci.evolve_single(pos=3, neg=5, M=8, pop=64, gens=2500, minibatch=128,
                     n_train=2000, n_val=500, whiten=True, seed=7, log_every=100)
