"""CIFAR v10: hybrid encoder — occlusion-contrastive generation x utility
gate (keep only genomes that increase fitness), then features + ceiling."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_pipe as cp

if __name__ == "__main__":
    cp.evolve_detbank_hybrid(n_bank=512, max_rounds=640, pop=64, gens=40,
                             probe_n=2000, util_probe_n=4000, seed=7)
    p = os.path.join(os.path.dirname(cp.DATA_DIR), "feats_v10_eig.npz")
    if os.path.exists(p):
        os.remove(p)
    D = cp.build_features(10, pca_scale="eig")
    print(f"v10 environment (hybrid encoder): {D['nf']} dims", flush=True)
    from sklearn.linear_model import LogisticRegression
    for C in (0.001, 0.003):
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(D["Ftr"], D["ytr"])
        print(f"CEILING v10 (C={C}): val {clf.score(D['Fva'], D['yva']):.4f} "
              f"test {clf.score(D['Fte'], D['yte']):.4f} "
              f"(occlusion-only 0.656; utility-only 0.509; label-driven 0.670)",
              flush=True)
