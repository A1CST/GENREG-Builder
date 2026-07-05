"""DiffEvo — denoising diffusion by neuroevolution (a separate program that
shares the Flask server, like tree_service / i2_service).

The bet (user's, restated correctly): a single tiny genome can't evolve into a
whole generative image model, but *one reverse-diffusion step is individually
easy*. So we decompose the hard task into a stack of easy denoisers and evolve
each one as a population that only has to do "an average job" — and we make it
generalize (not memorize one image) by scoring fitness on a fresh **minibatch of
samples** every generation. Stochastic-fitness-over-samples is the averaging that
lets the population "find the signal through the samples," exactly like SGD's
minibatch, with a generation of mutate+select standing in for the gradient step.

Design (pixel space, self-contained numpy — no engine dependency):

  * Data      : small procedural grayscale images (shapes on a background).
  * Forward   : K noise levels, additive Gaussian, std on a schedule.
  * Learner   : a *tiny convolutional per-pixel denoiser*. The genome is a small
                MLP  window(w*w) -> tanh(H) -> 1  applied to every pixel's local
                window. ~90 params. This is the "really easy task": local patch
                -> clean centre pixel. One independent, shared population per
                noise level (each level's denoiser specialises to its own scale).
  * Evolve    : (mu, lambda) ES with Gaussian mutation + elitism. Fitness =
                -L1 (mean absolute pixel error) of the reconstructed minibatch —
                L1 not L2, so a few large-miss pixels can't dominate selection
                and the genome optimises the "average" pixel. Plateau early-stop.
  * Stack     : reverse chain — at level k predict x0, renoise to level k-1,
                repeat down to the cleanest level. That composition is the
                de-noiser; applying it to a noised test image is "predicting an
                image by diffusion."

Streamed over WS /diffuse; training survives page navigation via the JobHub
(same pattern as tree_service.JobHub).
"""

import datetime
import hashlib
import json
import os
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
IMG = 12                 # image side (IMG x IMG grayscale, values in [0,1])
DATASET = 512            # procedural images
SIGMA_LO, SIGMA_HI = 0.06, 0.60   # noise-std schedule endpoints

# guardrails (clamped from the browser config)
LIM = {
    "levels":      (2, 16),
    "pop":         (4, 400),
    "hidden":      (2, 32),
    "window":      (1, 5),      # half-width; actual window = 2*window+1
    "minibatch":   (1, 128),
    "max_gens":    (5, 1000),
    "patience":    (3, 500),
}


def _clamp(v, lo, hi, default):
    try:
        return int(min(hi, max(lo, int(v))))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Procedural dataset — structured images a local prior can actually denoise
# --------------------------------------------------------------------------
def make_dataset(n=DATASET, size=IMG, seed=0):
    """n grayscale images in [0,1]; each is a few filled shapes on a flat bg.

    Deterministic given the seed so runs are reproducible. Structure (edges,
    flat regions) is what gives a local patch denoiser something real to learn.
    """
    rng = np.random.default_rng(seed)
    imgs = np.empty((n, size, size), np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    for i in range(n):
        bg = rng.uniform(0.0, 0.35)
        img = np.full((size, size), bg, np.float32)
        for _ in range(rng.integers(1, 4)):
            val = rng.uniform(0.4, 1.0)
            if rng.random() < 0.5:                       # filled rectangle
                x0, x1 = sorted(rng.integers(0, size, 2))
                y0, y1 = sorted(rng.integers(0, size, 2))
                img[y0:y1 + 1, x0:x1 + 1] = val
            else:                                        # filled disc
                cx, cy = rng.integers(0, size, 2)
                r = rng.integers(1, max(2, size // 2))
                img[(xs - cx) ** 2 + (ys - cy) ** 2 <= r * r] = val
        imgs[i] = np.clip(img, 0.0, 1.0)
    return imgs


def sigma_schedule(levels):
    """Noise stds, level 1 (cleanest) .. level K (noisiest)."""
    return np.linspace(SIGMA_LO, SIGMA_HI, levels).astype(np.float32)


# --------------------------------------------------------------------------
# Patch extraction (vectorised sliding window with edge padding)
# --------------------------------------------------------------------------
def _patches(imgs, half):
    """imgs (B,S,S) -> (B*S*S, (2*half+1)^2) row-major windows, edge-padded."""
    if half == 0:
        return imgs.reshape(imgs.shape[0], -1, 1).reshape(-1, 1)
    padded = np.pad(imgs, ((0, 0), (half, half), (half, half)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, (2 * half + 1, 2 * half + 1),
                                                   axis=(1, 2))
    # win: (B, S, S, w, w) -> (B*S*S, w*w)
    b, s1, s2 = imgs.shape
    return win.reshape(b * s1 * s2, (2 * half + 1) ** 2)


# --------------------------------------------------------------------------
# Population of tiny per-pixel denoisers (one population per noise level)
# --------------------------------------------------------------------------
class DenoiserPop:
    """P genomes, each an MLP  D -> tanh(H) -> 1  applied convolutionally.

    Weights held as stacked arrays so the whole population is evaluated in two
    einsums. Predicts the clean centre pixel from a noisy local window.
    """

    def __init__(self, pop, D, H, seed):
        rng = np.random.default_rng(seed)
        s1 = 1.0 / np.sqrt(D)
        self.W1 = (rng.standard_normal((pop, D, H)) * s1).astype(np.float32)
        self.b1 = np.zeros((pop, H), np.float32)
        self.W2 = (rng.standard_normal((pop, H, 1)) * (1.0 / np.sqrt(H))).astype(np.float32)
        self.b2 = np.zeros((pop, 1), np.float32)
        self.sigma = np.full(pop, 0.08, np.float32)   # self-adaptive step size
        self.pop, self.D, self.H = pop, D, H

    def predict_all(self, patches):
        """(N,D) patches -> (P,N) centre-pixel predictions for every genome."""
        h = np.tanh(np.einsum("nd,pdh->pnh", patches, self.W1) + self.b1[:, None, :])
        out = np.einsum("pnh,pho->pno", h, self.W2)[..., 0] + self.b2
        return out

    def predict_one(self, idx, patches):
        """One genome (N,D) -> (N,) — for the reverse-chain reconstruction."""
        h = np.tanh(patches @ self.W1[idx] + self.b1[idx])
        return (h @ self.W2[idx])[:, 0] + self.b2[idx, 0]

    def evolve_step(self, patches, target, elite_frac, mut, self_adaptive, rng):
        """One ES generation. Returns (best_idx, best_l1, mean_l1). Mutates
        the population in place: keep elites, refill from mutated elite parents.

        Fitness is **L1** (mean absolute pixel error), not L2/MSE: squared error
        lets a handful of large-miss pixels dominate selection, which a tiny
        genome can't chase; L1 rewards getting the bulk of pixels close and is
        the "average job" objective this whole approach is built around."""
        pred = self.predict_all(patches)
        err = np.mean(np.abs(pred - target[None, :]), axis=1)     # (P,) L1
        order = np.argsort(err)
        n_elite = max(1, int(round(self.pop * elite_frac)))
        elite = order[:n_elite]
        best_idx = int(order[0])

        # children: pick a random elite parent, copy, mutate
        n_child = self.pop - n_elite
        parents = elite[rng.integers(0, n_elite, size=n_child)]
        newW1 = self.W1[elite].copy(); newb1 = self.b1[elite].copy()
        newW2 = self.W2[elite].copy(); newb2 = self.b2[elite].copy()
        newSig = self.sigma[elite].copy()

        cW1 = self.W1[parents].copy(); cb1 = self.b1[parents].copy()
        cW2 = self.W2[parents].copy(); cb2 = self.b2[parents].copy()
        cSig = self.sigma[parents].copy()
        if self_adaptive:
            cSig = cSig * np.exp(0.2 * rng.standard_normal(n_child).astype(np.float32))
            cSig = np.clip(cSig, 1e-3, 0.5)
            step = cSig[:, None, None]
            step1 = cSig[:, None]
        else:
            step = np.float32(mut); step1 = np.float32(mut)
        cW1 += (rng.standard_normal(cW1.shape).astype(np.float32) * step)
        cb1 += (rng.standard_normal(cb1.shape).astype(np.float32) * step1)
        cW2 += (rng.standard_normal(cW2.shape).astype(np.float32) * step)
        cb2 += (rng.standard_normal(cb2.shape).astype(np.float32) * step1)

        self.W1 = np.concatenate([newW1, cW1]); self.b1 = np.concatenate([newb1, cb1])
        self.W2 = np.concatenate([newW2, cW2]); self.b2 = np.concatenate([newb2, cb2])
        self.sigma = np.concatenate([newSig, cSig])
        # genome 0 is always the current best (elites came first in `order`)
        return 0, float(err[best_idx]), float(err.mean())

    def champion(self):
        """Snapshot genome 0 (the standing best) as plain arrays."""
        return (self.W1[0].copy(), self.b1[0].copy(),
                self.W2[0].copy(), self.b2[0].copy())


def _apply_champion(champ, img, half):
    """Apply one champion to a single image (S,S) -> its output image."""
    W1, b1, W2, b2 = champ
    p = _patches(img[None], half)
    h = np.tanh(p @ W1 + b1)
    out = (h @ W2)[:, 0] + b2[0]
    return np.clip(out.reshape(img.shape), 0.0, 1.0)


def _apply_champion_batch(champ, imgs, half):
    """Apply one champion to a batch of images (B,S,S) -> (B,S,S). Used to build
    the actual walk-state inputs for unrolled training."""
    W1, b1, W2, b2 = champ
    p = _patches(imgs, half)
    h = np.tanh(p @ W1 + b1)
    out = (h @ W2)[:, 0] + b2[0]
    return np.clip(out.reshape(imgs.shape), 0.0, 1.0)


# --------------------------------------------------------------------------
# Reverse chain (stack the per-level champions into a de-noiser)
# --------------------------------------------------------------------------
def reverse_chain(champs, sigmas, half, x_noisy, start_level, rng, sampler="ddim"):
    """Denoise x_noisy (assumed at `start_level`) down to the cleanest level.

    At level k the level-k champion predicts a clean x0. How we move to level
    k-1 is the whole ballgame — a chain of local averaging denoisers collapses
    to the mean (a grey blob) if you feed it fresh randomness each step:

      * ``single``   — trust the strongest (top-level) denoiser only; one shot.
      * ``ancestral``— renoise x0 with FRESH Gaussian at sigma_{k-1}. Structure-
                       destroying here: each local filter over-smooths the new
                       noise and contracts contrast → grey. (Kept for contrast.)
      * ``ddim``     — deterministic: carry the residual that was ALREADY in x,
                       rescaled to the next level:  x <- x0 + (s_next/s_k)(x-x0).
                       No new randomness; the residual shrinks to 0 so the chain
                       converges onto x0 instead of walking away from it.

    Returns (final, [frame per level]).
    """
    frames = []
    x = x_noisy.copy()
    for k in range(start_level, 0, -1):            # levels are 1-indexed
        sig_k = float(sigmas[k - 1])
        x0 = _apply_champion(champs[k - 1], x, half)
        frames.append(x0.copy())
        if sampler == "single" or k == 1:
            break
        sig_next = float(sigmas[k - 2])
        if sampler == "ancestral":
            x = np.clip(x0 + rng.standard_normal(x0.shape).astype(np.float32) * sig_next,
                        0.0, 1.0)
        else:                                      # ddim (deterministic)
            resid = x - x0
            x = np.clip(x0 + (sig_next / max(sig_k, 1e-6)) * resid, 0.0, 1.0)
    return frames[-1], frames


def reverse_walk(champs, half, x_start, start_level):
    """The diffusion reverse process for 'diffuse'-mode (incremental) champions.

    Each champion_k was trained to take an image at noise level k and produce the
    image at level k-1 (one small step less noisy), NOT to jump to clean. So we
    simply COMPOSE them from the noisy start down to level 0:

        x_{k-1} = champion_k(x_k),   k = start_level .. 1

    No re-noising, no residual algebra — champion_k's output already looks like
    champion_{k-1}'s input, so the walk stays on-distribution and the noise scale
    shrinks every step. Start from x_start (the noised target ≈ noise) and each
    frame is visibly closer to the image. Returns (final, [frame per step]).
    """
    frames = []
    x = x_start.copy()
    for k in range(start_level, 0, -1):
        if champs[k - 1] is None:      # level not trained yet (progressive preview)
            break
        x = _apply_champion(champs[k - 1], x, half)   # level k -> k-1
        frames.append(x.copy())
    return (frames[-1] if frames else x), frames


def _round_img(a):
    return np.round(a.reshape(-1), 3).astype(float).tolist()


# --------------------------------------------------------------------------
# Trainer — evolves each level to plateau, streams events, honours stop()
# --------------------------------------------------------------------------
class DiffuseTrainer:
    def __init__(self, msg, emit):
        self.emit = emit
        self._stop = threading.Event()
        self.levels    = _clamp(msg.get("levels", 8), *LIM["levels"], 8)
        self.pop       = _clamp(msg.get("pop", 48), *LIM["pop"], 48)
        self.hidden    = _clamp(msg.get("hidden", 8), *LIM["hidden"], 8)
        self.half      = _clamp(msg.get("window", 1), *LIM["window"], 1)
        self.minibatch = _clamp(msg.get("minibatch", 16), *LIM["minibatch"], 16)
        self.max_gens  = _clamp(msg.get("max_gens", 120), *LIM["max_gens"], 120)
        self.patience  = _clamp(msg.get("patience", 18), *LIM["patience"], 18)
        try:
            self.elite = float(msg.get("elite_frac", 0.25))
        except (TypeError, ValueError):
            self.elite = 0.25
        self.elite = min(0.9, max(0.05, self.elite))
        try:
            self.mut = float(msg.get("mutation", 0.06))
        except (TypeError, ValueError):
            self.mut = 0.06
        self.self_adaptive = bool(msg.get("self_adaptive", True))
        self.seed = _clamp(msg.get("seed", 1234), 0, 2 ** 31 - 1, 1234)
        # Process mode:
        #  * "diffuse" — each champion learns ONE incremental denoise step
        #    (level k -> k-1); the reverse process starts from noise and walks
        #    to the image, every genome doing an easy job. (the diffusion idea.)
        #  * "denoise" — each champion learns to jump straight to clean (x0) from
        #    its level; best used single-shot with the matched champion.
        self.mode = str(msg.get("mode", "diffuse")).lower()
        if self.mode not in ("diffuse", "denoise"):
            self.mode = "diffuse"
        # Unrolled training (diffuse only): train each champion on the ACTUAL
        # output of the champions above it (the real walk distribution it meets
        # at inference), not on a freshly-noised image. Removes the train/infer
        # mismatch that caused the walk's low-noise tail to drift back up.
        self.unrolled = bool(msg.get("unrolled", True))
        # Sampler only applies to "denoise" mode. Single-shot is optimal there:
        # only the matched step sees the true observation.
        self.sampler = str(msg.get("sampler", "single")).lower()
        if self.sampler not in ("ddim", "ancestral", "single"):
            self.sampler = "single"
        self.raw = {k: v for k, v in msg.items() if k != "op"}
        self._runlog = None         # persisted run handle (runs/diffevo/<id>/)
        self._gidx = 0              # global generation index across all levels

    def stop(self):
        self._stop.set()

    # ---- run persistence (runs/diffevo/<id>/, same layout as runstore) -----
    def _persist_start(self, started):
        """Create runs/diffevo/<id>/ so the run shows on the /runs dashboard.
        Written directly (not via runstore) to keep DiffEvo free of the engine
        checkpoint format; the dashboard reads these files generically."""
        try:
            ts = datetime.datetime.now()
            stamp = ts.strftime("%Y%m%d-%H%M%S")
            h = hashlib.sha1(json.dumps(self.raw, sort_keys=True, default=str).encode()).hexdigest()[:6]
            rid = f"{stamp}-diffevo-{h}"
            d = os.path.join(RUNS_DIR, "diffevo", rid)
            os.makedirs(d, exist_ok=True)
            cfg = dict(self.raw)
            cfg.update(population=self.pop, generations=self.max_gens, device="cpu")
            meta = {"id": rid, "environment": "diffevo",
                    "created": ts.isoformat(timespec="seconds"),
                    "config": cfg, "started": started, "status": "running"}
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            open(os.path.join(d, "history.jsonl"), "w").close()
            self._runlog = {"id": rid, "dir": d}
        except OSError:
            self._runlog = None

    def _persist_gen(self, ev):
        """Append one generation to history.jsonl. `fitness.best/mean` is what
        the dashboard sparkline plots (descending = improving, since it's L1)."""
        if not self._runlog:
            return
        self._gidx += 1
        rec = {"gen": self._gidx, "level": ev.get("level"), "level_gen": ev.get("gen"),
               "fitness": {"best": ev.get("best_l1"), "mean": ev.get("mean_l1")}}
        try:
            with open(os.path.join(self._runlog["dir"], "history.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _persist_done(self, done):
        if not self._runlog:
            return
        d = self._runlog["dir"]
        imp = done.get("test_improvement")
        summary = {"id": self._runlog["id"], "environment": "diffevo",
                   "status": done.get("reason", "finished"),
                   "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                   "gen": self._gidx,
                   "best": {"score": imp, "final_l1": done.get("test_out_l1"),
                            "noisy_l1": done.get("test_in_l1")},
                   "l1_by_level": done.get("l1_by_level"),
                   "test_improvement": imp, "sampler": self.sampler,
                   "checkpoint": None}
        try:
            with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            cfg = None
            with open(os.path.join(d, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["status"] = summary["status"]
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except (OSError, ValueError):
            pass

    def run(self):
        try:
            self._run()
        except Exception as exc:                       # pragma: no cover
            self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    def _run(self):
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        D = (2 * self.half + 1) ** 2
        sigmas = sigma_schedule(self.levels)
        data = make_dataset(seed=self.seed)
        n_train = int(len(data) * 0.85)
        train, test = data[:n_train], data[n_train:]

        started = {"levels": self.levels,
                   "sigmas": [round(float(s), 3) for s in sigmas],
                   "pop": self.pop, "hidden": self.hidden, "window": 2 * self.half + 1,
                   "mode": self.mode, "sampler": self.sampler,
                   "unrolled": self.unrolled and self.mode == "diffuse",
                   "img": IMG, "minibatch": self.minibatch, "n_train": n_train,
                   "n_test": len(test), "params_per_genome": D * self.hidden + self.hidden
                   + self.hidden + 1}
        self._persist_start(started)
        self.emit({"type": "started", **started})

        champs = [None] * self.levels
        l1_by_level = [None] * self.levels

        # A fixed test image for the live preview — pick one with MEDIAN structure
        # (std), so the filmstrip reflects typical behavior rather than a near-flat
        # outlier (a flat image can't be denoised below its noise floor and any
        # texture only adds error). The held-out AVERAGE is reported separately.
        stds = test.reshape(len(test), -1).std(axis=1)
        preview_img = test[int(np.argsort(stds)[len(stds) // 2])]

        # Training order: diffuse trains noisiest-first (K..1) so each champion's
        # upstream neighbours already exist for unrolled inputs; denoise levels are
        # independent so order is irrelevant.
        order = (range(self.levels, 0, -1) if self.mode == "diffuse"
                 else range(1, self.levels + 1))

        for lvl in order:
            if self._stop.is_set():
                break
            sigma = float(sigmas[lvl - 1])
            self.emit({"type": "level_start", "level": lvl, "sigma": round(sigma, 3),
                       "of": self.levels})
            pop = DenoiserPop(self.pop, D, self.hidden, self.seed + lvl)

            best_hist = []
            best_l1 = float("inf")
            reason = "max_gens"
            gen = 0
            for gen in range(1, self.max_gens + 1):
                if self._stop.is_set():
                    reason = "stopped"
                    break
                idx = rng.integers(0, len(train), size=self.minibatch)
                clean = train[idx]
                z = rng.standard_normal(clean.shape).astype(np.float32)
                if self.mode == "diffuse":
                    # incremental target: same noise realization z, one level less
                    # noisy (level k -> k-1; level 0 = clean).
                    s_prev = float(sigmas[lvl - 2]) if lvl > 1 else 0.0
                    target = (clean + s_prev * z).reshape(-1)
                    if self.unrolled:
                        # real walk input: noised start run through champions K..lvl+1
                        x_in = self._walk_state(champs, sigmas, clean, z, lvl)
                    else:
                        x_in = clean + sigma * z       # freshly-noised (independent)
                else:                                  # denoise: jump to clean x0
                    target = clean.reshape(-1)
                    x_in = clean + sigma * z
                patches = _patches(x_in, self.half)
                _, bl1, ml1 = pop.evolve_step(
                    patches, target, self.elite, self.mut, self.self_adaptive, rng)
                if bl1 < best_l1:
                    best_l1 = bl1
                best_hist.append(bl1)
                gen_ev = {"type": "gen", "level": lvl, "gen": gen,
                          "generations": self.max_gens,
                          "best_l1": round(bl1, 6), "mean_l1": round(ml1, 6),
                          "best_ever": round(best_l1, 6)}
                self._persist_gen(gen_ev)
                self.emit(gen_ev)
                # plateau: no meaningful best improvement over `patience` gens
                if len(best_hist) > self.patience:
                    window = best_hist[-self.patience:]
                    if (min(best_hist[:-self.patience]) - min(window)) < 1e-5:
                        reason = "plateau"
                        break

            champs[lvl - 1] = pop.champion()
            l1_by_level[lvl - 1] = round(best_l1, 6)
            self.emit({"type": "level_done", "level": lvl, "gen": gen,
                       "best_l1": round(best_l1, 6), "reason": reason,
                       "sigma": round(sigma, 3)})

            # live preview: diffuse walks from the top (deepening as lower levels
            # train); denoise single-shots the level just trained.
            top = self.levels if self.mode == "diffuse" else lvl
            self._preview(champs, sigmas, preview_img, top, rng)

        # final evaluation over the held-out test set (full process from the top)
        summary = self._evaluate(champs, sigmas, test, rng) if not self._stop.is_set() else {}
        done = {"type": "done",
                "reason": "stopped" if self._stop.is_set() else "finished",
                "l1_by_level": l1_by_level, "seconds": round(time.time() - t0, 1),
                **summary}
        self._persist_done(done)
        self.emit(done)

    def _walk_state(self, champs, sigmas, clean, z, level):
        """Unrolled-training input for the level-`level` champion: the noised start
        (level K) run through the already-trained champions K..level+1 — i.e. the
        ACTUAL image the reverse walk hands this champion at inference. clean and z
        are (B,S,S). Returns (B,S,S)."""
        x = np.clip(clean + float(sigmas[self.levels - 1]) * z, 0.0, 1.0)
        for j in range(self.levels, level, -1):        # champions K..(level+1)
            cj = champs[j - 1]
            if cj is not None:
                x = _apply_champion_batch(cj, x, self.half)
        return x

    def _reverse(self, champs, sigmas, noisy, top_level, rng):
        """Run the reverse process for the active mode: an incremental walk
        (diffuse) or the x0 chain/single-shot (denoise). Returns (final, frames)."""
        if self.mode == "diffuse":
            return reverse_walk(champs, self.half, noisy, top_level)
        return reverse_chain(champs, sigmas, self.half, noisy, top_level, rng, self.sampler)

    def _preview(self, champs, sigmas, img, top_level, rng):
        """Reconstruct the fixed test image via the process built so far and emit
        frames. In diffuse mode this is a real reverse diffusion: start from the
        noised image and each frame walks visibly closer to the target."""
        sigma_top = float(sigmas[top_level - 1])
        noisy = np.clip(img + rng.standard_normal(img.shape).astype(np.float32) * sigma_top,
                        0.0, 1.0)
        final, frames = self._reverse(champs, sigmas, noisy, top_level, rng)
        in_l1 = float(np.mean(np.abs(noisy - img)))
        out_l1 = float(np.mean(np.abs(final - img)))
        chain_l1 = [round(float(np.mean(np.abs(f - img))), 4) for f in frames]
        self.emit({"type": "sample", "top_level": top_level,
                   "mode": self.mode, "sampler": self.sampler,
                   "clean": _round_img(img), "noisy": _round_img(noisy),
                   "final": _round_img(final),
                   "chain": [_round_img(f) for f in frames], "chain_l1": chain_l1,
                   "in_l1": round(in_l1, 5), "out_l1": round(out_l1, 5)})

    def _evaluate(self, champs, sigmas, test, rng):
        """Average reconstruction improvement (L1 to target) across the whole
        held-out test set, starting from the noisiest level."""
        if any(c is None for c in champs):
            return {}
        top = self.levels
        sigma_top = float(sigmas[top - 1])
        in_l1s, out_l1s = [], []
        for img in test:
            noisy = np.clip(img + rng.standard_normal(img.shape).astype(np.float32) * sigma_top,
                            0.0, 1.0)
            final, _ = self._reverse(champs, sigmas, noisy, top, rng)
            in_l1s.append(float(np.mean(np.abs(noisy - img))))
            out_l1s.append(float(np.mean(np.abs(final - img))))
        in_m, out_m = float(np.mean(in_l1s)), float(np.mean(out_l1s))
        return {"test_in_l1": round(in_m, 5), "test_out_l1": round(out_m, 5),
                "test_improvement": round((in_m - out_m) / in_m, 4) if in_m > 0 else 0.0}


# --------------------------------------------------------------------------
# Job hub — training survives WebSocket disconnects (same as tree_service)
# --------------------------------------------------------------------------
def _notify_board(program, ev):
    """End-of-run alarm: post terminal job events to the shared Agent panel.
    Best-effort — the board must never be able to break training."""
    try:
        import agent_board
        agent_board.post_run_event(program, ev)
    except Exception:
        pass


class JobHub:
    """Generic single-job hub (also reused by animation_evo with its own
    program name); `program` labels Agent-panel notices and the job thread."""

    JOURNAL_CAP = 40000

    def __init__(self, program="diffevo"):
        self.program = program
        self._lock = threading.Lock()
        self._trainer = None
        self._thread = None
        self._journal = []
        self._cur_gens = []
        self._subs = []

    def running(self):
        with self._lock:
            th = self._thread
        return bool(th is not None and th.is_alive())

    def subscribe(self, fn):
        with self._lock:
            self._subs.append(fn)

    def unsubscribe(self, fn):
        with self._lock:
            if fn in self._subs:
                self._subs.remove(fn)

    def snapshot(self):
        with self._lock:
            return list(self._journal) + list(self._cur_gens)

    def _emit(self, ev):
        with self._lock:
            t = ev.get("type")
            if t == "gen":
                self._cur_gens.append(ev)
            elif t == "level_start":
                self._cur_gens = []
                self._journal.append(ev)
            else:
                self._journal.append(ev)
            if len(self._journal) > self.JOURNAL_CAP:
                self._journal = self._journal[-self.JOURNAL_CAP:]
            subs = list(self._subs)
        if ev.get("type") in ("done", "error"):
            _notify_board(self.program, ev)   # Agent-panel alarm on job end/crash
        for fn in subs:
            try:
                fn(ev)
            except Exception:
                self.unsubscribe(fn)

    def start(self, msg, cls):
        with self._lock:
            old_tr, old_th = self._trainer, self._thread
        if old_tr is not None:
            old_tr.stop()
        if old_th is not None:
            old_th.join(timeout=5.0)
        trainer = cls(msg, self._emit)
        th = threading.Thread(target=trainer.run, name=f"{self.program}-job", daemon=True)
        with self._lock:
            self._trainer, self._thread = trainer, th
            self._journal, self._cur_gens = [], []
        th.start()

    def stop(self):
        with self._lock:
            tr = self._trainer
        if tr is not None:
            tr.stop()


HUB = JobHub()
