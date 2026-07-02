"""A RICHER WORLD, fitness = food only. No shaped reward, no designed gradient.

An organism lives on a line of W cells. A few food patches sit at RANDOM-PER-LIFE locations,
deplete when eaten, and regrow IN PLACE. Vision is LOCAL (it senses only the cell it is on), but
it knows its own position and the world PERSISTS -- it can leave a mark on a cell that stays.

Fitness is exactly how much food it eats over its life -- the world's own consequence, nothing
graded toward a target. An organism that remembers where food is (by marking patches and walking
back) eats more and is selected; one that cannot only stumbles on food. Strictly feedforward + one
hidden layer; gradient-free evolution. Memory must be the organism's own invention, paid for in food.

Honest controls (different WORLD PHYSICS, not different rewards):
  - PERSISTENT food (regrows in place -> remembering a patch pays) vs RELOCATING food (a depleted
    patch reappears elsewhere -> past locations are worthless, memory cannot help)
  - marks enabled vs disabled (does it use the world as an external store?)
"""
import numpy as np

W, F, T = 9, 2, 48                       # cells, food patches, life length
M, H = 3, 24                             # mark symbols, hidden units
CAP, BITE, REGEN = 1.0, 0.34, 0.06
IN = 3 + (M + 1) + W                     # food-here bucket, mark-here(+empty), position


def new_pop(N, rng):
    return dict(W1=rng.normal(0, 0.4, (N, H, IN)), b1=rng.normal(0, 0.1, (N, H)),
                wg=rng.normal(0, 0.4, (N, H)), Wv=rng.normal(0, 0.4, (N, M, H)),
                Wm=rng.normal(0, 0.4, (N, 3, H)))


def perceive(amt, mark, pos):
    p = np.zeros(IN)
    p[0 if amt < 0.08 else (1 if amt < 0.5 else 2)] = 1
    p[3 + mark] = 1
    p[3 + (M + 1) + pos] = 1
    return p


def life(g, j, rng, relocate=False, no_marks=False, trace=False):
    sites = list(rng.permutation(W)[:F]); amt = {c: CAP for c in sites}
    marks = [M] * W; pos = int(rng.integers(W)); eaten = 0.0; visits = []
    for _ in range(T):
        a = amt.get(pos, 0.0)
        h = np.tanh(g["W1"][j] @ perceive(a, marks[pos], pos) + g["b1"][j])
        if a > 0.05:                                 # eat what's here (this is the only payoff)
            b = min(BITE, a); eaten += b; amt[pos] = a - b
            if relocate and amt[pos] < 0.05:         # relocating world: depleted patch jumps elsewhere
                amt.pop(pos); free = [c for c in range(W) if c not in amt]
                amt[int(rng.choice(free))] = CAP
        if not no_marks and (g["wg"][j] @ h) > 0:
            marks[pos] = int(np.argmax(g["Wv"][j] @ h))
        if trace: visits.append((pos, round(a, 2), marks[pos]))
        pos = int(np.clip(pos + (int(np.argmax(g["Wm"][j] @ h)) - 1), 0, W - 1))
        for c in list(amt):                          # food regrows IN PLACE (persistent world)
            amt[c] = min(CAP, amt[c] + REGEN)
    return (eaten, visits) if trace else eaten


def fitness(g, rng, relocate=False, no_marks=False, lives=6):
    N = len(g["W1"])
    return np.array([sum(life(g, j, rng, relocate, no_marks) for _ in range(lives)) for j in range(N)], float)


def evaluate(g, rng, n=200, relocate=False, no_marks=False):
    N = len(g["W1"])
    return float(np.mean([life(g, int(rng.integers(N)), rng, relocate, no_marks) for _ in range(n)]))


def evolve(gens=500, N=44, mut=0.2, cull=0.25, seed=1, relocate=False, no_marks=False):
    rng = np.random.default_rng(seed); g = new_pop(N, rng)
    for t in range(gens):
        en = fitness(g, rng, relocate, no_marks)
        Kc = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for key in ("W1", "b1", "wg", "Wv", "Wm"):
                m = rng.random(g[key][pa].shape) < 0.5
                g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"Foraging life: {W} cells, {F} patches at random-per-life spots, local vision. Fitness = food eaten.", flush=True)
    print(f"a purely random walker eats ~ {0.0:.1f} baseline below; more = exploiting structure", flush=True)
    print("=" * 70, flush=True)
    print("T1 — does memory pay? PERSISTENT (regrows in place) vs RELOCATING (memory useless) world", flush=True)
    print("     and with vs without the ability to MARK the world", flush=True)
    rows = [("PERSISTENT + marks", False, False), ("PERSISTENT + no marks", False, True),
            ("RELOCATING + marks", True, False)]
    res = {}
    for tag, rel, nm in rows:
        f = np.mean([evaluate(evolve(gens=500, seed=s, relocate=rel, no_marks=nm), rng, relocate=rel, no_marks=nm)
                     for s in range(3)])
        res[tag] = f
        print(f"  {tag:24}: food eaten per life = {f:.2f}", flush=True)
    print("=" * 70, flush=True)
    print("T2 — behaviour trace (does it mark patches and walk back to them?)", flush=True)
    g = evolve(gens=650, seed=1)
    en = fitness(g, rng, lives=20); j = int(np.argmax(en))
    eaten, visits = life(g, j, np.random.default_rng(3), trace=True)
    seen = " ".join(f"{c}{'*' if mk!=M else ''}{'+' if a>0.1 else ''}" for c, a, mk in visits[:30])
    print(f"    path (cell, *=marked, +=had food): {seen}", flush=True)
    print(f"    eaten this life = {eaten:.2f}", flush=True)
    print(f"  ablation: marks on={evaluate(g, rng):.2f}  marks disabled={evaluate(g, rng, no_marks=True):.2f}", flush=True)
