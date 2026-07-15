"""CIFAR v7: layer-2 diversity genomes (512) over the frozen 768-genome v6c
layer-1 bank, then combined L1+L2 pure features and the ceiling probe."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_pipe as cp

if __name__ == "__main__":
    cp.evolve_detbank_l2(n_bank=512, max_rounds=96, pop=48, gens=40,
                         probe_n=1200, admit_per_round=8, seed=7)
    p = os.path.join(os.path.dirname(cp.DATA_DIR), "feats_v7_eig.npz")
    if os.path.exists(p):
        os.remove(p)
    D = cp.build_features(7, pca_scale="eig")
    print(f"v7 environment (L1+L2, pure evolved): {D['nf']} dims", flush=True)
    from sklearn.linear_model import LogisticRegression
    for C in (0.001, 0.003):
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(D["Ftr"], D["ytr"])
        print(f"CEILING v7 (C={C}): val {clf.score(D['Fva'], D['yva']):.4f} "
              f"test {clf.score(D['Fte'], D['yte']):.4f} "
              f"(v6c single-layer 0.589; v5 label-driven 0.670)", flush=True)
