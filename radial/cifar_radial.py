"""cifar_radial.py — the manufactured-rotation SEED-STACK (built for MNIST in
mnist_radial.py) applied to CIFAR-10, the harder / less-tabulatable task.

The point of moving here: on MNIST the composed genomes earned ZERO residual
(genome_residual −0.0001; force-residual arm = stats-only) because the cross-pose
signal was fully tabulatable — a mean and a variance across poses was the whole
answer. CIFAR perception is NOT tabulatable (that is exactly why the hand-crafted
Coates-Ng patches only reach 0.59 and evolved grammar-v2 climbs to 0.70), so this
is where the across-seed GENOMES should finally earn their keep.

Reuses mnist_radial's machinery wholesale (EnvLite, evolve_roles, the image-pose
seed tensor, cross-seed stats, the composed-across-seed space, the ridge ladder,
the genome ablation, and the force-residual/lean A/B). Only the loader differs:
CIFAR is natively 32x32x3, so no deskew and no grayscale->3 tiling.

Frontier to beat (radial classifier line, gradient-free, test touched once):
best single substrate 0.7074, 7-substrate union 0.7702 (radial_data exports).
"""
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import argparse
import os

import numpy as np

import mnist_radial as mr

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIFAR_NPZ = os.path.join(_HERE, "radial_data", "cifar_full.npz")


def load_cifar(n_train=None, n_test=None, val_frac=0.15, seed=0):
    """(N,32,32,3) [0,1] floats + labels; val is the tail of the shuffled train
    block (rows [n_fit:] gate champions); the 10k test set is untouched."""
    z = np.load(CIFAR_NPZ)
    X = z["Xtr"].astype(np.float32) / 255.0
    y = z["ytr"].astype(np.int64)
    Xt = z["Xte"].astype(np.float32) / 255.0
    yt = z["yte"].astype(np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    if n_train:
        X, y = X[:n_train], y[:n_train]
    if n_test:
        Xt, yt = Xt[:n_test], yt[:n_test]
    n_fit = int(round(len(X) * (1 - val_frac)))
    return {"Xtr": X, "ytr": y, "Xte": Xt, "yte": yt, "n_fit": n_fit}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--max-deg", type=float, default=12.0)
    ap.add_argument("--seed-mode", choices=["image", "feature"], default="image")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--max-roles", type=int, default=256)
    ap.add_argument("--role-rounds", type=int, default=80)
    ap.add_argument("--comp-rounds", type=int, default=80)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--ab", action="store_true",
                    help="force-residual vs lean (do the genomes earn on CIFAR?)")
    args = ap.parse_args()
    n_train, n_test = args.n_train, args.n_test
    if args.smoke:
        n_train = n_train or 4000
        n_test = n_test or 2000
    D = load_cifar(n_train, n_test)
    mr.run(data=D, dataset="cifar", n_seeds=args.seeds, max_deg=args.max_deg,
           seed_mode=args.seed_mode, grid=args.grid, max_roles=args.max_roles,
           role_rounds=args.role_rounds, comp_rounds=args.comp_rounds,
           heavy_union=False, ab=args.ab, smoke=args.smoke)
