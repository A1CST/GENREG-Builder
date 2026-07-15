"""Web backend for the /cifar page — CIFAR-Pipe (staged; trains later).

Mirror of mnist_service over cifar_pipe: lazy-loads the built environment +
any trained genomes (demo/cifar_genomes.pkl), evaluates layer subsets, and
serves 32x32 RGB sample predictions for the image grid. Until a battery has
been run the page shows the environment stats and "no trained genomes".
"""
import os
import pickle
import threading

import numpy as np

from genreg_train import cifar_pipe as cp
from genreg_train import mnist_pipe as mp

CACHE = cp.CACHE


class Service:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False
        self.loading = False
        self.err = None
        self.champs = {}

    def ensure(self):
        with self.lock:
            if self.ready or self.loading:
                return
            self.loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def reload(self):
        with self.lock:
            self.ready = False
            self.loading = False
            self.champs = {}
        self.ensure()

    def _load(self):
        try:
            if os.path.exists(CACHE):
                with open(CACHE, "rb") as f:
                    self.champs = pickle.load(f)
            fv = self.champs.get("feat_version", 2)
            self.D = cp.build_features(fv)
            self.Xte = cp.load_cifar()[4]
            self.centroid = cp.centroid_baseline(fv)
            self.ready = True
        except Exception as exc:                   # pragma: no cover
            import traceback; traceback.print_exc()
            self.err = f"{type(exc).__name__}: {exc}"
        finally:
            self.loading = False

    def status(self):
        s = {"ready": self.ready, "loading": self.loading, "err": self.err,
             "has_genomes": bool(self.champs.get("joint") or self.champs.get("det")),
             "labels": cp.LABELS}
        if self.ready:
            params = sum(_nparams(self.champs.get(k))
                         for k in ("det", "pairs", "mixer", "joint"))
            s.update({
                "nf": int(self.D["nf"]),
                "feat_version": self.champs.get("feat_version", 2),
                "train_n": int(len(self.D["ytr"])), "val_n": int(len(self.D["yva"])),
                "test_n": int(len(self.D["yte"])),
                "n_detectors": len(self.champs.get("det", {})),
                "n_pairs": len(self.champs.get("pairs", {})),
                "has_joint": "joint" in self.champs,
                "params": int(params),
                "centroid_acc": round(self.centroid, 4),
                "results": self.champs.get("results", {}),
                "pair_margin": self.champs.get("pair_margin", 3.0),
            })
        return s

    def evaluate(self, use_mixer=True, use_pairs=True):
        if not self.ready or not (self.champs.get("joint") or self.champs.get("det")):
            return {"err": "no trained genomes — run: python -m genreg_train.cifar_pipe"}
        m = self.champs.get("pair_margin", 3.0)
        r = cp.evaluate(self.champs, "test", use_mixer, use_pairs, m)
        r["centroid_acc"] = round(self.centroid, 4)
        return r

    def sample(self, seed=0, n=48, use_mixer=True, use_pairs=True, only_errors=False):
        if not self.ready or not (self.champs.get("joint") or self.champs.get("det")):
            return {"err": "no trained genomes"}
        m = self.champs.get("pair_margin", 3.0)
        pred, _ = mp.predict(self.champs, self.D["Fte"], use_mixer, use_pairs, m)
        y = self.D["yte"]
        rng = np.random.default_rng(seed)
        pool = np.where(pred != y)[0] if only_errors else np.arange(len(y))
        if len(pool) == 0:
            return {"items": []}
        idx = pool[rng.permutation(len(pool))[:n]]
        items = []
        for i in idx:
            px = (self.Xte[i] * 255).astype(np.uint8).reshape(-1).tolist()  # RGB
            items.append({"px": px, "true": int(y[i]), "pred": int(pred[i]),
                          "ok": bool(pred[i] == y[i])})
        return {"items": items, "n_errors": int((pred != y).sum()),
                "acc": round(float((pred == y).mean()), 4)}


def _nparams(x):
    if isinstance(x, np.ndarray):
        return x.size
    if isinstance(x, (tuple, list)):
        return sum(_nparams(e) for e in x)
    if isinstance(x, dict):
        return sum(_nparams(e) for e in x.values())
    return 0


SERVICE = Service()
