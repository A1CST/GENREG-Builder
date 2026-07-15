import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import cifar_internal as ci
if __name__ == "__main__":
    ci.evolve_single_masked(pos=3, neg=5, M_max=24, pop=64, gens=2500,
                            minibatch=128, n_train=2000, n_val=500,
                            cost_lambda=0.015, seed=7, log_every=100)
