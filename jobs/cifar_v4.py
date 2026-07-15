"""CIFAR-Pipe first full campaign: evolve the detector bank, then run the v4
battery (evolved environment -> centroid-warm joint head -> pairwise referees
-> margin gate -> one-shot test). GPU-accelerated fitness (evo_gpu)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_pipe as cp

if __name__ == "__main__":
    cp.evolve_detbank(n_bank=96, rounds=48, pop=48, gens=60, sub=2500, seed=7)
    cp.run_battery(version=4, det_gens=0, joint_gens=6000, pair_gens=1500, seed=7)
