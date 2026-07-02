"""SPATIAL MEMORY, reachable version (graded survival).

The index-query task hit a reachability wall: spreading items pays nothing unless navigation
ALSO works, so there is no gradient. Fix (no answer wired in, just a graded world): the organism
must STORE a sequence of cues and REPRODUCE it. Now storing one more item correctly = one more
point, so partial spatial use is rewarded and the search can climb. Read starts at the tape origin
(rewind) -- the organism must still decide where to write and how to scan.

Still strictly feedforward, no internal state. Contrast W=1 (one cell, ceiling ~1 item) vs W>1
(room to lay a sequence out); freeze-head ablation to prove space is load-bearing.
"""
import numpy as np

V, NCUE = 4, 3


def IN(W):
    return V + (V + 1) + W + 1          # input, content(+empty), head onehot, mode


def new_pop(N, rng, W):
    n = IN(W)
    return dict(Wo=rng.normal(0, 0.5, (N, V, n)), Wv=rng.normal(0, 0.5, (N, V, n)),
                wg=rng.normal(0, 0.5, (N, n)), Wm=rng.normal(0, 0.5, (N, 3, n)), W=W)


def perceive(x, content, head, mode, W):
    p = np.zeros(IN(W)); o = 0
    p[o + x] = 1; o += V
    p[o + content] = 1; o += V + 1
    p[o + head] = 1; o += W
    p[o] = mode
    return p


def run(g, j, cues, freeze=False, trace=False):
    W = g["W"]; tape = [V] * W; head = 0
    for c in cues:                                       # WRITE: decide what + where
        p = perceive(c, tape[head], head, 0, W)
        if (g["wg"][j] @ p) > 0:
            tape[head] = int(np.argmax(g["Wv"][j] @ p))
        if not freeze:
            head = int(np.clip(head + (int(np.argmax(g["Wm"][j] @ p)) - 1), 0, W - 1))
    head = 0; outs = []                                  # READ: rewind, scan, reproduce
    for _ in range(NCUE):
        p = perceive(V, tape[head], head, 1, W)
        outs.append(int(np.argmax(g["Wo"][j] @ p)))
        if not freeze:
            head = int(np.clip(head + (int(np.argmax(g["Wm"][j] @ p)) - 1), 0, W - 1))
    return (outs, tape) if trace else outs


def score(g, j, rng, freeze=False):
    cues = [int(rng.integers(V)) for _ in range(NCUE)]
    outs = run(g, j, cues, freeze)
    return sum(int(outs[i] == cues[i]) for i in range(NCUE))


def play(g, rng, trials=12, freeze=False):
    N = len(g["Wo"])
    return np.array([sum(score(g, j, rng, freeze) for _ in range(trials)) for j in range(N)], float)


def recall(g, rng, n=500, freeze=False):
    N = len(g["Wo"])
    return np.mean([score(g, int(rng.integers(N)), rng, freeze) for _ in range(n)]) / NCUE


def evolve(gens=450, N=44, W=4, mut=0.22, cull=0.25, seed=1):
    rng = np.random.default_rng(seed); g = new_pop(N, rng, W)
    for t in range(gens):
        en = play(g, rng)
        Kc = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for key in ("Wo", "Wv", "wg", "Wm"):
                m = rng.random(g[key][pa].shape) < 0.5
                g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"Store a {NCUE}-cue sequence and reproduce it. per-position chance {1/V:.2f}.", flush=True)
    print("One cell holds one item; reproducing the whole sequence needs SPACE.", flush=True)
    print("=" * 70, flush=True)
    print("T1 — fraction of the sequence reproduced, vs tape width W", flush=True)
    for W in [1, 3, 5]:
        rc = [recall(evolve(gens=450, W=W, seed=s), rng) for s in range(2)]
        print(f"  W={W}: reproduced = {np.mean(rc):.2f}", flush=True)
    print("=" * 70, flush=True)
    print("T2 — layout invented (trace W=5)", flush=True)
    g = evolve(gens=550, W=5, seed=1)
    en = play(g, np.random.default_rng(5), trials=40); j = int(np.argmax(en))
    for _ in range(5):
        cues = [int(rng.integers(V)) for _ in range(NCUE)]
        outs, tape = run(g, j, cues, trace=True)
        nz = sum(1 for c in tape if c != V)
        print(f"    cues={cues} -> {outs} {'OK' if outs==cues else '~'}  tape={tape} ({nz} cells used)", flush=True)
    print("=" * 70, flush=True)
    print("T3 — ablation: freeze head (deny space)", flush=True)
    print(f"  movable={recall(g, rng):.2f}   frozen={recall(g, rng, freeze=True):.2f}", flush=True)
