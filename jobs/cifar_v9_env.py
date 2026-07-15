"""CIFAR v9: utility-driven encoder (keep a genome only if it increases
fitness), then pure features + ceiling probe."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_pipe as cp

if __name__ == "__main__":
    cp.evolve_detbank_utility(n_bank=512, max_rounds=768, pop=64, gens=30,
                              probe_n=4000, seed=7)
    p = os.path.join(os.path.dirname(cp.DATA_DIR), "feats_v9_eig.npz")
    if os.path.exists(p):
        os.remove(p)
    D = cp.build_features(9, pca_scale="eig")
    print(f"v9 environment (utility encoder): {D['nf']} dims", flush=True)
    from sklearn.linear_model import LogisticRegression
    for C in (0.001, 0.003):
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(D["Ftr"], D["ytr"])
        print(f"CEILING v9 (C={C}): val {clf.score(D['Fva'], D['yva']):.4f} "
              f"test {clf.score(D['Fte'], D['yte']):.4f} "
              f"(occlusion 0.656; label-driven v5 0.670)", flush=True)
