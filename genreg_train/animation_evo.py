"""Animation Evo — a VERY simple evolutionary shape classifier.

The task: the animation dataset (animation_data) is 10 clips of a single
white shape moving over black. Each frame shows one of the shape classes
(circle / square / diamond / ring / triangle) at a different position.
A genome must output the CORRECT SHAPE for every presented frame — i.e.
recognise the shape *despite the movement*.

The model (per genome, fixed tiny architecture):

    frame 64*64=4096  ->  encoder tanh(ENC=12)  ->  tanh(HID=24)  ->  logits(n_shapes)

Presentation: ONE clip per generation. The genome is shown a single
clip's 24 frames in temporal order (the shape moving along its path from
start to finish); it never sees more than one clip at a time. Clip order
is shuffled and reshuffled after each epoch (all 10 clips shown once).
Fitness for that generation is accuracy on just those 24 frames — a
stochastic (minibatch-of-one-clip) objective, the same idea as the
DiffEvo fresh-minibatch fitness. The genome is stateless, so the frames'
temporal order doesn't change the accuracy value, but they are presented
in sequence as specified.

Evolution is deliberately minimal — **mutation only, no crossover**:
elitism keeps the top fraction unchanged; every child is a copy of ONE
elite parent plus fixed-step Gaussian noise on all weights. Training
metrics are the noisy per-clip fitness plus a rolling epoch average; the
true cross-clip accuracy (and confusion / per-clip / encoder-PCA report)
is measured only at the END, clip-by-clip — never as a 240-frame batch.

Streamed over WS /animevo via the shared JobHub (diffuse_service.JobHub);
also runs headless:

    python -m genreg_train.animation_evo          # train + verify + PNG report
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import threading
import time

import numpy as np

from . import animation_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")

ENC = 12                 # encoder dims (user-fixed for now)
HID = 24                 # hidden units (user-fixed for now)

LIM = {
    "pop":      (4, 400),
    "enc":      (2, 64),
    "hidden":   (2, 128),
    "max_gens": (5, 5000),
    "patience": (5, 2000),
    "seed":     (0, 2 ** 31 - 1),
}


def _clamp(v, lo, hi, default):
    try:
        return int(min(hi, max(lo, int(v))))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Dataset: all animation frames, labelled by the shape they contain
# --------------------------------------------------------------------------
def make_dataset():
    """Returns (X, y, shape_names, clip_names, clip_of_frame).

    X: (N, 4096) float32 flattened frames; y: (N,) int shape-class labels.
    N = 10 clips * 24 frames = 240. The same shape appears at many positions
    (its motion path), which is exactly the invariance the classifier must
    evolve."""
    clips = animation_data.generate_all()
    shape_of_clip = {name: shape.__name__ for name, _, shape in animation_data.ANIMATIONS}
    shape_names = sorted(set(shape_of_clip.values()))
    xs, ys, clip_names, clip_of_frame = [], [], [], []
    for ci, (name, frames) in enumerate(clips.items()):
        label = shape_names.index(shape_of_clip[name])
        clip_names.append(name)
        for f in frames:
            xs.append(f.reshape(-1))
            ys.append(label)
            clip_of_frame.append(ci)
    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return X, y, shape_names, clip_names, np.array(clip_of_frame)


# --------------------------------------------------------------------------
# Population — stacked weights, mutation-only ES with elitism
# --------------------------------------------------------------------------
class ClassifierPop:
    """P genomes of  IN -> tanh(ENC) -> tanh(HID) -> logits(C)."""

    def __init__(self, pop, n_in, enc, hid, n_out, seed):
        rng = np.random.default_rng(seed)
        def init(a, b):
            return (rng.standard_normal((pop, a, b)) / np.sqrt(a)).astype(np.float32)
        self.W1 = init(n_in, enc); self.b1 = np.zeros((pop, enc), np.float32)
        self.W2 = init(enc, hid);  self.b2 = np.zeros((pop, hid), np.float32)
        self.W3 = init(hid, n_out); self.b3 = np.zeros((pop, n_out), np.float32)
        self.pop = pop
        self.n_params = n_in * enc + enc + enc * hid + hid + hid * n_out + n_out

    def forward(self, X):
        """(N,IN) -> logits (P,N,C). Encoder activations are recomputed on
        demand for the champion only (see champion_encode)."""
        h1 = np.tanh(np.einsum("ni,pie->pne", X, self.W1) + self.b1[:, None, :])
        h2 = np.tanh(np.einsum("pne,peh->pnh", h1, self.W2) + self.b2[:, None, :])
        return np.einsum("pnh,phc->pnc", h2, self.W3) + self.b3[:, None, :]

    def fitness(self, X, y):
        """Accuracy per genome (P,) on the frames given — the fraction whose
        shape the genome names correctly. `X` is ONE clip's 24 frames per
        generation, never the whole set."""
        pred = self.forward(X).argmax(axis=2)
        return (pred == y[None, :]).mean(axis=1).astype(np.float64)

    def evolve_step(self, X, y, elite_frac, mut, rng):
        """One generation on ONE presented clip: rank by accuracy on those 24
        frames, keep elites, refill with mutated copies of single elite parents.
        MUTATION ONLY — children never mix weights from two parents; fixed
        Gaussian step `mut` on every weight. Fitness is stochastic across
        generations because a different clip is presented each time."""
        acc = self.fitness(X, y)
        order = np.argsort(-acc)                  # accuracy on this clip, desc
        n_elite = max(1, int(round(self.pop * elite_frac)))
        elite = order[:n_elite]

        n_child = self.pop - n_elite
        parents = elite[rng.integers(0, n_elite, size=n_child)]
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            w = getattr(self, name)
            child = w[parents] + rng.standard_normal(
                w[parents].shape).astype(np.float32) * np.float32(mut)
            setattr(self, name, np.concatenate([w[elite], child]))
        # genome 0 is now best-on-this-clip
        b = int(order[0])
        return float(acc[b]), float(acc.mean())

    def champion_predict(self, X, idx=0):
        """Genome `idx` -> (N,) predicted classes for the frames given."""
        h1 = np.tanh(X @ self.W1[idx] + self.b1[idx])
        h2 = np.tanh(h1 @ self.W2[idx] + self.b2[idx])
        return (h2 @ self.W3[idx] + self.b3[idx]).argmax(axis=1)

    def champion_encode(self, X, idx=0):
        """Genome `idx`'s ENC-dim encoder activations (N, ENC)."""
        return np.tanh(X @ self.W1[idx] + self.b1[idx])


# --------------------------------------------------------------------------
# Trainer — streams events through the JobHub (same shape as DiffuseTrainer)
# --------------------------------------------------------------------------
class AnimEvoTrainer:
    def __init__(self, msg, emit):
        self.emit = emit
        self._stop = threading.Event()
        self.pop      = _clamp(msg.get("pop", 48), *LIM["pop"], 48)
        self.enc      = _clamp(msg.get("enc", ENC), *LIM["enc"], ENC)
        self.hidden   = _clamp(msg.get("hidden", HID), *LIM["hidden"], HID)
        self.max_gens = _clamp(msg.get("max_gens", 500), *LIM["max_gens"], 500)
        self.patience = _clamp(msg.get("patience", 100), *LIM["patience"], 100)
        self.seed     = _clamp(msg.get("seed", 1234), *LIM["seed"], 1234)
        try:
            self.elite = min(0.9, max(0.05, float(msg.get("elite_frac", 0.25))))
        except (TypeError, ValueError):
            self.elite = 0.25
        try:
            self.mut = min(1.0, max(1e-4, float(msg.get("mutation", 0.05))))
        except (TypeError, ValueError):
            self.mut = 0.05
        self.raw = {k: v for k, v in msg.items() if k != "op"}
        self._runlog = None

    def stop(self):
        self._stop.set()

    # ---- run persistence (runs/animevo/<id>/) ------------------------------
    def _persist_start(self, started):
        try:
            ts = datetime.datetime.now()
            h = hashlib.sha1(json.dumps(self.raw, sort_keys=True,
                                        default=str).encode()).hexdigest()[:6]
            rid = f"{ts.strftime('%Y%m%d-%H%M%S')}-animevo-{h}"
            d = os.path.join(RUNS_DIR, "animevo", rid)
            os.makedirs(d, exist_ok=True)
            cfg = dict(self.raw)
            cfg.update(population=self.pop, generations=self.max_gens, device="cpu")
            meta = {"id": rid, "environment": "animevo",
                    "created": ts.isoformat(timespec="seconds"),
                    "config": cfg, "started": started, "status": "running"}
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            open(os.path.join(d, "history.jsonl"), "w").close()
            self._runlog = {"id": rid, "dir": d}
        except OSError:
            self._runlog = None

    def _persist_gen(self, ev):
        if not self._runlog:
            return
        rec = {"gen": ev["gen"],
               "fitness": {"best": ev["roll_acc"], "mean": ev["pop_mean"]}}
        try:
            with open(os.path.join(self._runlog["dir"], "history.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _persist_done(self, done):
        if not self._runlog:
            return
        summary = {"id": self._runlog["id"], "environment": "animevo",
                   "status": done.get("reason", "finished"),
                   "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                   "gen": done.get("gen"),
                   "best": {"score": done.get("full_acc")},
                   "full_acc": done.get("full_acc"),
                   "best_roll": done.get("best_roll"), "checkpoint": None}
        try:
            d = self._runlog["dir"]
            with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
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
        X, y, shapes, clips, clip_of = make_dataset()
        n_clips = len(clips)
        # frames of each clip, IN TEMPORAL ORDER (make_dataset appends frame
        # 0..23 per clip), so X[masks[ci]] is that clip's shape moving along its
        # path from start to finish — never the whole 240-frame set at once.
        masks = [np.where(clip_of == ci)[0] for ci in range(n_clips)]
        pop = ClassifierPop(self.pop, X.shape[1], self.enc, self.hidden,
                            len(shapes), self.seed)

        started = {"pop": self.pop, "enc": self.enc, "hidden": self.hidden,
                   "shapes": shapes, "clips": clips,
                   "labels": y.tolist(), "frames": int(X.shape[0]),
                   "frames_per_clip": animation_data.FRAMES,
                   "size": animation_data.SIZE,
                   "elite_frac": self.elite, "mutation": self.mut,
                   "max_gens": self.max_gens, "n_clips": n_clips,
                   "regime": "one clip per generation; clip order shuffled each "
                             "epoch, 24 frames in temporal order",
                   "params_per_genome": pop.n_params}
        self._persist_start(started)
        self.emit({"type": "started", **started})

        # shuffled clip presentation order, reshuffled after each epoch (all
        # n_clips clips shown once). ONE clip per generation.
        seq = []
        def next_clip():
            if not seq:
                seq.extend(int(i) for i in rng.permutation(n_clips))
            return seq.pop()

        roll = []                      # champion's clip-accuracy, last epoch
        best_roll, best_gen, reason, gen = 0.0, 0, "max_gens", 0
        for gen in range(1, self.max_gens + 1):
            if self._stop.is_set():
                reason = "stopped"
                break
            ci = next_clip()
            m = masks[ci]
            # present ONLY this clip's 24 frames (in order) to the population
            bacc, macc = pop.evolve_step(X[m], y[m], self.elite, self.mut, rng)
            roll.append(bacc)
            if len(roll) > n_clips:
                roll.pop(0)
            racc = sum(roll) / len(roll)          # smoothed across the epoch
            if racc > best_roll + 1e-9:
                best_roll, best_gen = racc, gen
            ev = {"type": "gen", "gen": gen, "generations": self.max_gens,
                  "clip": ci, "clip_name": clips[ci],
                  "clip_acc": round(bacc, 4), "pop_mean": round(macc, 4),
                  "roll_acc": round(racc, 4), "best_roll": round(best_roll, 4),
                  "pred": pop.champion_predict(X[m]).tolist()}
            self._persist_gen(ev)
            self.emit(ev)
            if gen - best_gen >= self.patience:
                reason = "plateau"
                break

        done = {"type": "done", "reason": reason, "gen": gen,
                "best_roll": round(best_roll, 4),
                "seconds": round(time.time() - t0, 1),
                **self._final_report(pop, X, y, shapes, clips, masks)}
        self._persist_done(done)
        self.emit(done)

    def _final_report(self, pop, X, y, shapes, clips, masks):
        """Final champion diagnostics, computed ONE CLIP AT A TIME (ten separate
        24-frame passes — never a 240-wide batch). Picks the genome with the
        best accuracy summed across all clips, then builds its confusion matrix,
        per-clip accuracy, and a 2D PCA of the ENC-dim encoder space."""
        C = len(shapes)
        n_clips = len(clips)
        # per-genome accuracy accumulated clip-by-clip
        correct = np.zeros(pop.pop, dtype=np.float64)
        total = 0
        for m in masks:
            pr = pop.forward(X[m]).argmax(axis=2)          # (P, 24)
            correct += (pr == y[m][None, :]).sum(axis=1)
            total += len(m)
        full = correct / total
        champ = int(full.argmax())

        # champion predictions + encoder codes, gathered clip-by-clip
        pred = np.empty(len(y), dtype=int)
        z = np.empty((len(y), self.enc), dtype=np.float32)
        for m in masks:
            pred[m] = pop.champion_predict(X[m], champ)
            z[m] = pop.champion_encode(X[m], champ)

        conf = np.zeros((C, C), dtype=int)
        for t, p in zip(y, pred):
            conf[t, p] += 1
        per_clip = [round(float((pred[masks[ci]] == y[masks[ci]]).mean()), 4)
                    for ci in range(n_clips)]
        zc = z - z.mean(axis=0)
        _, _, vt = np.linalg.svd(zc, full_matrices=False)
        p2 = zc @ vt[:2].T
        span = np.abs(p2).max() or 1.0
        return {"full_acc": round(float(full[champ]), 4), "champ_idx": champ,
                "confusion": conf.tolist(), "per_clip_acc": per_clip,
                "pred": pred.tolist(),
                "encoder_2d": np.round(p2 / span, 3).tolist()}


# --------------------------------------------------------------------------
# Job hub — shared implementation, own program label
# --------------------------------------------------------------------------
from .diffuse_service import JobHub  # noqa: E402  (reused, parameterised)

HUB = JobHub("animevo")


# --------------------------------------------------------------------------
# Headless verification: train, print, and render a PNG report (PIL only)
# --------------------------------------------------------------------------
def _report_png(events, done, started, path):
    """Fitness curve + confusion matrix + encoder PCA, no matplotlib."""
    from PIL import Image, ImageDraw

    W, H = 1240, 360
    img = Image.new("RGB", (W, H), (10, 12, 16))
    dr = ImageDraw.Draw(img)

    # fitness curve (left 2/3)
    gens = [e for e in events if e["type"] == "gen"]
    px, py, pw, ph = 40, 30, 560, 270
    dr.rectangle([px, py, px + pw, py + ph], outline=(60, 70, 90))
    for frac in (0.25, 0.5, 0.75, 1.0):
        yy = py + ph - int(ph * frac)
        dr.line([px, yy, px + pw, yy], fill=(28, 34, 44))
        dr.text((6, yy - 6), f"{frac:.2f}", fill=(120, 130, 150))
    n = max(2, len(gens))
    def xy(i, v):
        return px + int(pw * i / (n - 1)), py + ph - int(ph * v)
    # faint = per-clip fitness (noisy, one clip/gen); bold = rolling epoch avg
    for key, col in (("clip_acc", (70, 84, 104)), ("roll_acc", (120, 200, 255))):
        pts = [xy(i, g[key]) for i, g in enumerate(gens)]
        if len(pts) > 1:
            dr.line(pts, fill=col, width=2)
    dr.text((px, 8), f"clip fitness (faint) + rolling epoch avg (bold) vs gen  —  "
                     f"final champion {done['full_acc']:.3f} over all clips, "
                     f"{done['reason']} @ gen {done['gen']}", fill=(200, 210, 225))

    # confusion matrix (top right)
    shapes = started["shapes"]
    C = len(shapes)
    conf = np.array(done["confusion"], dtype=float)
    cell = 24
    ox, oy = 700, 60
    m = conf.max() or 1.0
    for i in range(C):
        for j in range(C):
            v = conf[i, j] / m
            col = (int(30 + 90 * v), int(40 + 180 * v), int(60 + 160 * v))
            dr.rectangle([ox + j * cell, oy + i * cell,
                          ox + (j + 1) * cell - 2, oy + (i + 1) * cell - 2], fill=col)
            if conf[i, j]:
                dr.text((ox + j * cell + 4, oy + i * cell + 6),
                        str(int(conf[i, j])), fill=(230, 235, 245))
        dr.text((ox - 52, oy + i * cell + 6), shapes[i][:7], fill=(160, 170, 190))
        dr.text((ox + i * cell + 2, oy + C * cell + 4), shapes[i][:3],
                fill=(160, 170, 190))
    dr.text((ox - 52, oy - 26), "confusion (true row / pred col)", fill=(200, 210, 225))

    # encoder PCA scatter (far right)
    sx, sy, ss = 1000, 115, 90
    dr.text((sx - 20, sy - 24), "encoder PCA (colour = shape)", fill=(200, 210, 225))
    pal = [(120, 200, 255), (255, 170, 90), (140, 235, 140), (240, 120, 200),
           (250, 240, 120), (180, 154, 255), (90, 224, 210), (255, 140, 140),
           (200, 180, 120), (154, 180, 200)]
    labels = started["labels"]
    for (x2, y2), lab in zip(done["encoder_2d"], labels):
        cx = sx + int(ss + x2 * ss)
        cy = sy + int(45 + y2 * 45)
        dr.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=pal[lab % len(pal)])
    img.save(path)


def main():
    events = []
    def emit(ev):
        events.append(ev)
        if ev["type"] == "gen" and ev["gen"] % 20 == 0:
            print(f"  gen {ev['gen']:4d}  clip {ev['clip_name']:9s} "
                  f"clip_acc {ev['clip_acc']:.3f}  roll {ev['roll_acc']:.3f}  "
                  f"pop_mean {ev['pop_mean']:.3f}")

    tr = AnimEvoTrainer({}, emit)
    print(f"animevo: pop={tr.pop} enc={tr.enc} hid={tr.hidden} "
          f"mut={tr.mut} elite={tr.elite} max_gens={tr.max_gens} "
          f"(mutation only; ONE clip/gen, shuffled order, frames in sequence)")
    tr.run()
    started = next(e for e in events if e["type"] == "started")
    done = next(e for e in events if e["type"] == "done")
    print(f"done: {done['reason']} at gen {done['gen']}, "
          f"final champion accuracy over all clips {done['full_acc']:.4f} "
          f"in {done['seconds']}s")
    print("per-clip accuracy:",
          {c: a for c, a in zip(started["clips"], done["per_clip_acc"])})
    out = os.path.join(ROOT, "animations", "evo_report.png")
    _report_png(events, done, started, out)
    print(f"report image -> {out}")


if __name__ == "__main__":
    main()
