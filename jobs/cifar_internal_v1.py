"""CIFAR internal-language v1: a single end-to-end genome separates two
categories (automobile vs bird), everything evolved, no gradients."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_internal as ci

if __name__ == "__main__":
    ci.evolve_single(pos=1, neg=2, M=8, pop=64, gens=2500, minibatch=128,
                     n_train=2000, n_val=500, whiten=True, seed=7, log_every=100)
