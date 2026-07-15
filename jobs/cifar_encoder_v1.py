import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import cifar_internal as ci
if __name__ == "__main__":
    ci.evolve_encoder(M=16, d=16, pop=48, gens=2500, n_anchor=1500, V=4,
                      N=64, tau=0.2, seed=7, log_every=100)
