"""CIFAR round-2 environment: whitened niched detector bank -> v5 features ->
closed-form ceiling probe (the go/no-go gate before any classifier genome)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from genreg_train import cifar_pipe as cp

if __name__ == "__main__":
    cp.evolve_detbank_niched(n_bank=192, pop=32, gens=40, per_class=400, seed=7)
    D = cp.build_features(5)
    print(f"v5 environment: {D['nf']} dims", flush=True)
    print(f"centroid floor: {cp.centroid_baseline(5):.4f}", flush=True)
    Ftr, ytr = D["Ftr"], D["ytr"]
    n, nf = Ftr.shape
    W = np.zeros((nf, 10), np.float32); b = np.zeros(10, np.float32)
    Y = np.eye(10, dtype=np.float32)[ytr]
    for it in range(400):
        Z = Ftr @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= 0.5 * (Ftr.T @ G + 1e-4 * W); b -= 0.5 * G.sum(0)
    va = float((((D["Fva"] @ W + b).argmax(1)) == D["yva"]).mean())
    te = float((((D["Fte"] @ W + b).argmax(1)) == D["yte"]).mean())
    print(f"CEILING v5: val {va:.4f} test {te:.4f} (v4 was 0.5938/0.5883)", flush=True)
