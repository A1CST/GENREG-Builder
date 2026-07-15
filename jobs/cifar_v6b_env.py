"""CIFAR v6b: scale the label-free diversity bank 256 -> 768 (the archive
never rejected a candidate at 256 — behavior space unsaturated), stricter
duplicate cap, then rebuild pure features and re-probe the ceiling."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genreg_train import cifar_pipe as cp

if __name__ == "__main__":
    cp.evolve_detbank_diverse(n_bank=768, max_rounds=128, pop=64, gens=50,
                              probe_n=2000, admit_per_round=8, dup_cap=0.80,
                              seed=7, out=cp.DETBANK6)
    for f in ("feats_v6_eig.npz",):
        p = os.path.join(os.path.dirname(cp.DATA_DIR), f)
        if os.path.exists(p):
            os.remove(p)
    D = cp.build_features(6, pca_scale="eig")
    print(f"v6b environment: {D['nf']} dims", flush=True)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000, C=0.001)
    clf.fit(D["Ftr"], D["ytr"])
    print(f"CEILING v6b: val {clf.score(D['Fva'], D['yva']):.4f} "
          f"test {clf.score(D['Fte'], D['yte']):.4f} "
          f"(v6@256 was 0.5451, v5 label-driven 0.670)", flush=True)
