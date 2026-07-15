"""Continuation stacking: evolve the v8 pure-stack joint champion onward
(bigger population, fresh 8k generations), re-gate, one test evaluation."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from genreg_train import cifar_pipe as cp
from genreg_train import mnist_pipe as mp

if __name__ == "__main__":
    with open(cp.CACHE, "rb") as f:
        champs = pickle.load(f)
    D = cp.build_features(champs["feat_version"], pca_scale=champs.get("pca_scale", "unit"))
    for r in range(3):
        res = mp.train_joint(champs, gens=8000, pop=240, seed=100 + r, D=D,
                             minibatch=0)                 # fold_stack continues 'joint'
        champs.update(res)
        print(f"continuation {r + 1}: joint_val {res['joint_val_acc']} "
              f"(from {res['joint_base_val_acc']})", flush=True)
    m, vacc = cp.tune_pair_margin(champs, log=lambda *a: None)
    champs["pair_margin"] = m
    te = cp.evaluate(champs, "test", True, m > 0, m)["acc"]
    champs["results"]["full_test"] = te
    with open(cp.CACHE, "wb") as f:
        pickle.dump(champs, f)
    print(f"FINAL after continuations: val {vacc:.4f} (margin {m}) TEST {te:.4f}",
          flush=True)
