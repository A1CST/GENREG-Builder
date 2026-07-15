"""CIFAR v6c: label-free diversity bank with the STABILITY information term
(novelty x entropy x jitter-SNR), 768 genomes, then features + ceiling."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_pipe as cp

if __name__ == "__main__":
    cp.evolve_detbank_diverse(n_bank=768, max_rounds=128, pop=64, gens=50,
                              probe_n=2000, admit_per_round=8, dup_cap=0.80,
                              seed=7, out=cp.DETBANK6, stability=True)
    p = os.path.join(os.path.dirname(cp.DATA_DIR), "feats_v6_eig.npz")
    if os.path.exists(p):
        os.remove(p)
    D = cp.build_features(6, pca_scale="eig")
    print(f"v6c environment: {D['nf']} dims", flush=True)
    from sklearn.linear_model import LogisticRegression
    for C in (0.001, 0.003):
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(D["Ftr"], D["ytr"])
        print(f"CEILING v6c (C={C}): val {clf.score(D['Fva'], D['yva']):.4f} "
              f"test {clf.score(D['Fte'], D['yte']):.4f} "
              f"(v6b 0.5838; v5 label-driven 0.670)", flush=True)
