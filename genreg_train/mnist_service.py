"""Web backend for the /mnist page — the MNIST-Pipe specialist pipeline.

Lazy-loads the trained genomes (demo/mnist_genomes.pkl) + the built statistics
layer in a background thread on first request, then evaluates the pipeline with
any subset of the evolved layers enabled and serves sample predictions for the
digit grid. Same shape as wordpipe_service: the page toggles layers, the
service re-runs inference over the frozen champions. Numpy only, no gradients.
"""
import os
import pickle
import threading

import numpy as np

from genreg_train import mnist_pipe as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = mp.CACHE


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
        """Drop state so the next status hit reloads fresh champions (after a
        training run finishes)."""
        with self.lock:
            self.ready = False
            self.loading = False
            self.champs = {}
        self.ensure()

    def _load(self):
        try:
            self.D = mp.build_features()
            self.Xte = mp.load_mnist()[4]
            self.centroid = mp.centroid_baseline()
            if os.path.exists(CACHE):
                with open(CACHE, "rb") as f:
                    self.champs = pickle.load(f)
            self.ready = True
        except Exception as exc:                   # pragma: no cover
            import traceback; traceback.print_exc()
            self.err = f"{type(exc).__name__}: {exc}"
        finally:
            self.loading = False

    @staticmethod
    def _nparams(x):
        if isinstance(x, np.ndarray):
            return x.size
        if isinstance(x, (tuple, list)):
            return sum(Service._nparams(e) for e in x)
        if isinstance(x, dict):
            return sum(Service._nparams(e) for e in x.values())
        return 0

    def status(self):
        s = {"ready": self.ready, "loading": self.loading, "err": self.err,
             "has_genomes": bool(self.champs.get("det"))}
        if self.ready:
            n_det = len(self.champs.get("det", {}))
            n_pairs = len(self.champs.get("pairs", {}))
            params = sum(self._nparams(self.champs.get(k)) for k in
                         ("det", "pairs", "mixer") if k in self.champs)
            s.update({
                "nf": int(self.D["nf"]),
                "train_n": int(len(self.D["ytr"])), "val_n": int(len(self.D["yva"])),
                "test_n": int(len(self.D["yte"])),
                "n_detectors": n_det, "n_pairs": n_pairs,
                "has_mixer": "mixer" in self.champs,
                "params": int(params),
                "centroid_acc": round(self.centroid, 4),
                "results": self.champs.get("results", {}),
                "pair_margin": self.champs.get("pair_margin", 3.0),
            })
        return s

    def evaluate(self, use_mixer=True, use_pairs=True):
        """Test accuracy + confusion for the enabled layer subset."""
        if not self.ready or not self.champs.get("det"):
            return {"err": "no trained genomes"}
        m = self.champs.get("pair_margin", 3.0)
        r = mp.evaluate(self.champs, "test", use_mixer, use_pairs, m)
        r["centroid_acc"] = round(self.centroid, 4)
        return r

    def sample(self, seed=0, n=48, use_mixer=True, use_pairs=True, only_errors=False):
        """A grid of test digits with predictions: [{px (784 ints 0-255), true,
        pred, ok}]. `only_errors` draws from the pipeline's mistakes."""
        if not self.ready or not self.champs.get("det"):
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
            px = (self.Xte[i] * 255).astype(np.uint8).reshape(-1).tolist()
            items.append({"px": px, "true": int(y[i]), "pred": int(pred[i]),
                          "ok": bool(pred[i] == y[i])})
        return {"items": items, "n_errors": int((pred != y).sum()),
                "acc": round(float((pred == y).mean()), 4)}


SERVICE = Service()
