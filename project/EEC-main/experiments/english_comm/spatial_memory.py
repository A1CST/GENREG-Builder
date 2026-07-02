"""FROM ONE MARK TO ORGANIZING SPACE.

memory_invent.py: a feedforward organism invented external memory using ONE persistent world
slot. Next jump (decide not just WHETHER to write but WHERE): give it a TAPE of many cells and a
movable head. Still strictly feedforward -- no internal state, no update rule. To survive it must
store SEVERAL cues at once, which one cell cannot hold, so it has to:
   - decide where to write (lay items out without colliding)
   - navigate back to the right place to read

World state (persistent): the tape contents AND the head position. Organism perceives (current
input, content under the head, head position, mode, the query index) and acts (response, write?,
what, MOVE in {-1,0,+1}). Nothing prescribes a layout.

Tests:
   - tape width W = 1 (one cell, can't lay out) vs W > 1 (room to organize): recall of a queried item
   - distinct cells used during writing (did it spread items across space?)
   - ablation: freeze the head at 0 (deny it space) -> multi-item recall collapses to the 1-slot ceiling
"""
import numpy as np

V, NCUE = 4, 3                          # symbol alphabet, number of cues to store


def IN(W):
    return V + (V + 1) + W + 1 + NCUE   # input, content(+empty), head onehot, mode, query onehot


def new_pop(N, rng, W):
    n = IN(W)
    return dict(Wo=rng.normal(0, 0.5, (N, V, n)),        # response
                Wv=rng.normal(0, 0.5, (N, V, n)),        # value to write
                wg=rng.normal(0, 0.5, (N, n)),           # write gate
                Wm=rng.normal(0, 0.5, (N, 3, n)), W=W)   # move {-1,0,+1}


def perceive(x, content, head, mode, query, W):
    p = np.zeros(IN(W)); o = 0
    p[o + x] = 1; o += V                                 # current input (x==V means none)
    p[o + content] = 1; o += V + 1                       # content under head (==V means empty)
    p[o + head] = 1; o += W                              # head position
    p[o] = mode; o += 1                                  # 0=write phase, 1=read phase
    p[o + query] = 1                                     # which index is being asked (read phase)
    return p


def run(g, j, cues, q, freeze=False, trace=False):
    W = g["W"]; tape = [V] * W; head = 0
    for c in cues:                                       # WRITE phase
        p = perceive(c, tape[head], head, 0, 0, W)
        if (g["wg"][j] @ p) > 0:
            tape[head] = int(np.argmax(g["Wv"][j] @ p))
        if not freeze:
            head = int(np.clip(head + (int(np.argmax(g["Wm"][j] @ p)) - 1), 0, W - 1))
    out = 0
    for _ in range(W + 1):                               # READ phase: navigate then answer
        p = perceive(V, tape[head], head, 1, q, W)
        out = int(np.argmax(g["Wo"][j] @ p))
        if not freeze:
            head = int(np.clip(head + (int(np.argmax(g["Wm"][j] @ p)) - 1), 0, W - 1))
    return (out, tape) if trace else out


def trial(g, j, rng, freeze=False):
    cues = [int(rng.integers(V)) for _ in range(NCUE)]
    q = int(rng.integers(NCUE))
    return int(run(g, j, cues, q, freeze) == cues[q])


def play(g, rng, trials=18, freeze=False):
    N = len(g["Wo"])
    return np.array([sum(trial(g, j, rng, freeze) for _ in range(trials)) for j in range(N)], float)


def recall(g, rng, n=600, freeze=False):
    N = len(g["Wo"])
    return np.mean([trial(g, int(rng.integers(N)), rng, freeze) for _ in range(n)])


def evolve(gens=600, N=48, W=4, mut=0.22, cull=0.25, seed=1):
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
    print(f"Feedforward organism, tape of W cells + movable head. Store {NCUE} cues, recall a queried one.")
    print(f"chance = {1/V:.2f}.  One cell can hold one item; {NCUE} items need SPACE.")
    print("=" * 72)
    print("T1 — does spatial organization pay? recall vs tape width W", flush=True)
    for W in [1, 4]:
        rc = [recall(evolve(gens=400, W=W, seed=s), rng) for s in range(2)]
        print(f"  W={W}: recall = {np.mean(rc):.2f}", flush=True)
    print("=" * 72, flush=True)
    print("T2 — what layout did it invent? (trace an evolved W=4 organism)", flush=True)
    g = evolve(gens=500, W=4, seed=1)
    en = play(g, np.random.default_rng(5), trials=40); j = int(np.argmax(en))
    for _ in range(5):
        cues = [int(rng.integers(V)) for _ in range(NCUE)]; q = int(rng.integers(NCUE))
        out, tape = run(g, j, cues, q, trace=True)
        print(f"    cues={cues} ask#{q}(={cues[q]}) -> {out} {'OK' if out==cues[q] else 'x'}   tape={tape}")
    print("  (distinct non-empty cells => it spread items across space, not one slot)")
    print("=" * 72)
    print("T3 — ablation: freeze the head at cell 0 (deny it space)")
    print(f"  recall with movable head = {recall(g, rng):.2f}   "
          f"head frozen = {recall(g, rng, freeze=True):.2f}  (collapse => it used the SPACE)")
