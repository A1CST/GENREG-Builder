"""lm_generate.py — sample text from a trained radial LM stack.

Replays the checkpoint's genome features on the training windows, fits the
closed-form head once (fp64), then generates character-by-character:
window -> one-hot bank -> genome conjunctions -> 27-way logits ->
temperature sample. Frozen per-space standardization stats come from the
training replay, so a single generation row sees exactly the transform the
head was fit on. No gradients anywhere.

    python lm_generate.py            # samples from radial_data/lm_model.json
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401
import json
import os
import time

import numpy as np

import radial_lm
from radial_evo import _tprims
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _san(torch, v):
    return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e6, 1e6)


class RadialLM:
    def __init__(self, model_path=None, data_path=None):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tp = _tprims(torch)
        with open(model_path or os.path.join(_HERE, "radial_data",
                                             "lm_model.json")) as f:
            self.model = json.load(f)
        self.T = int(self.model["frames"])
        radial_lm.T = self.T
        z = np.load(data_path or os.path.join(_HERE, "radial_data",
                                              "lm_ids.npz"))
        assert z["ctx_tr"].shape[1] == self.T, \
            "lm_ids.npz window size does not match the checkpoint"
        self._fit_head(z)

    def _fit_head(self, z):
        torch, dev = self.torch, self.dev
        ytr = z["ytr"]
        Ntr = len(ytr)
        B0 = radial_lm._onehot(torch, dev, z["ctx_tr"])
        cols = [B0[:, j] for j in range(B0.shape[1])]
        self.stats = []                   # per-space (zmu, zsd), frozen
        bank = B0
        for sp in self.model["spaces"]:
            s = torch.stack([_san(torch, rk.feature_vec(torch, self.tp, bank, g))
                             for g in sp], 1)
            zmu, zsd = s.mean(0), s.std(0) + 1e-6
            s = ((s - zmu) / zsd).clamp(-8, 8)
            self.stats.append((zmu, zsd))
            cols.extend(s[:, j] for j in range(s.shape[1]))
            bank = torch.cat([B0, s], 1)
        F = torch.stack(cols, 1)
        Y = -torch.ones((Ntr, radial_lm.N_CLASSES), device=dev)
        Y[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0
        n_fit = int(Ntr * 0.8)
        yv = torch.tensor(ytr[n_fit:], device=dev)

        def _fit(Xf, Yf, lam):
            n, d = Xf.shape
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
            A = torch.hstack([(Xf - mu) / sd,
                              torch.ones(n, 1, device=dev)]).double()
            W = torch.linalg.solve(
                A.T @ A + lam * torch.eye(d + 1, device=dev,
                                          dtype=torch.float64),
                A.T @ Yf.double()).float()
            return mu, sd, W

        best = (3.0, -1.0)
        for lam in (1.0, 3.0, 10.0, 30.0):
            mu, sd, W = _fit(F[:n_fit], Y[:n_fit], lam)
            s = torch.hstack([(F[n_fit:] - mu) / sd,
                              torch.ones(Ntr - n_fit, 1, device=dev)]) @ W
            a = float((s.argmax(1) == yv).float().mean())
            if a > best[1]:
                best = (lam, a)
        self.mu, self.sd, self.W = _fit(F, Y, best[0])
        self.val_acc = best[1]

    def _logits(self, window_ids):
        """window_ids: list of T char indices -> (27,) logits."""
        torch, dev = self.torch, self.dev
        ctx = np.array([window_ids], np.int8)
        B0 = radial_lm._onehot(torch, dev, ctx)
        cols = [B0[:, j] for j in range(B0.shape[1])]
        bank = B0
        for sp, (zmu, zsd) in zip(self.model["spaces"], self.stats):
            s = torch.stack([_san(torch, rk.feature_vec(torch, self.tp, bank, g))
                             for g in sp], 1)
            s = ((s - zmu) / zsd).clamp(-8, 8)
            cols.extend(s[:, j] for j in range(s.shape[1]))
            bank = torch.cat([B0, s], 1)
        F = torch.stack(cols, 1)
        return (torch.hstack([(F - self.mu) / self.sd,
                              torch.ones(1, 1, device=dev)]) @ self.W)[0]

    def generate(self, prompt, n_chars=120, temp=0.8, seed=0):
        rng = np.random.default_rng(seed)
        window = [radial_lm._IDX.get(c, 26) for c in prompt.lower()][-self.T:]
        while len(window) < self.T:
            window.insert(0, 26)          # left-pad with space
        out = []
        for _ in range(n_chars):
            lg = self._logits(window).cpu().numpy().astype(np.float64)
            p = np.exp((lg - lg.max()) / max(temp, 1e-3))
            p /= p.sum()
            nxt = int(rng.choice(radial_lm.N_CLASSES, p=p))
            out.append(radial_lm.CHARS[nxt])
            window = window[1:] + [nxt]
        return prompt + "".join(out)


def main():
    t0 = time.time()
    lm = RadialLM()
    print(f"head refit: val {lm.val_acc:.4f} ({round(time.time()-t0)}s)",
          flush=True)
    prompts = ["the ", "she was ", "in the year ", "a small "]
    samples = []
    for i, pr in enumerate(prompts):
        txt = lm.generate(pr, n_chars=120, temp=0.8, seed=i)
        samples.append({"prompt": pr, "temp": 0.8, "text": txt})
        print(f"  [{pr!r}] -> {txt}", flush=True)
    op = os.path.join(_HERE, "radial_data", "lm_radial.json")
    with open(op) as f:
        out = json.load(f)
    out["samples"] = samples
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[lm-generate] samples appended to {op} "
          f"({round(time.time()-t0)}s)", flush=True)


if __name__ == "__main__":
    main()
