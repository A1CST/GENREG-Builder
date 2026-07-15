"""CIFAR v6 environment: label-free DIVERSITY bank -> pure evolved features
(no hand statistics) -> regularised ceiling probe (the go/no-go gate).
Comparison target: the label-driven v5 environment's ceiling (~67.0 test)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_pipe as cp

if __name__ == "__main__":
    cp.evolve_detbank_diverse(n_bank=256, max_rounds=64, pop=64, gens=50,
                              probe_n=2000, admit_per_round=8, seed=7)
    D = cp.build_features(6, pca_scale="eig")
    print(f"v6 environment (pure evolved, eig): {D['nf']} dims", flush=True)
    from sklearn.linear_model import LogisticRegression
    for C in (0.001, 0.01):
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(D["Ftr"], D["ytr"])
        print(f"CEILING v6 (C={C}): val {clf.score(D['Fva'], D['yva']):.4f} "
              f"test {clf.score(D['Fte'], D['yte']):.4f} "
              f"(v5 label-driven ceiling was ~0.670)", flush=True)
