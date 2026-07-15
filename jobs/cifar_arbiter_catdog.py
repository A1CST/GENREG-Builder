import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import cifar_internal as ci
if __name__ == "__main__":
    ci.evolve_arbiter(pos=3, neg=5, M=8, seedA=7, seedB=101, dec_gens=2000,
                      checker_gens=1500, n_train=2000, n_check=2000)
