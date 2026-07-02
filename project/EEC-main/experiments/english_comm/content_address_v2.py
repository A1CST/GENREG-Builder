"""CONTENT-ADDRESSED memory, take 2: a richer BRAIN (one nonlinear hidden layer).

v1 failed at an EXPRESSIVITY wall: a single linear layer over separate one-hots cannot compute
'move toward the cell indexed by the key' (a join of key x head-position). That is not wiring in
the answer -- it is a missing capacity. So we give the organism a hidden layer (tanh), a body
part, and let it DISCOVER content-addressing. We do not give it a lookup/attention mechanism.

Also: values are DISTINCT per trial, so 'guess one value' scores exactly 1/K (no degenerate trap).
Random arrival order forces content- over position-addressing. Graded over all keys. Ablation:
freeze the head.
"""
import numpy as np

K, V, H = 2, 20, 28   # large value space -> guessing is worthless -> gradient toward real addressing                       # pairs/keys, value alphabet, hidden units
W = K


def IN():
    return K + V + K + W + (V + 1) + 1


def new_pop(N, rng):
    n = IN()
    return dict(W1=rng.normal(0, 0.4, (N, H, n)), b1=rng.normal(0, 0.1, (N, H)),
                Wo=rng.normal(0, 0.4, (N, V, H)), Wv=rng.normal(0, 0.4, (N, V, H)),
                wg=rng.normal(0, 0.4, (N, H)), Wm=rng.normal(0, 0.4, (N, 3, H)))


def perceive(key, val, qkey, head, content, mode):
    p = np.zeros(IN()); o = 0
    if key >= 0: p[o + key] = 1
    o += K
    if val >= 0: p[o + val] = 1
    o += V
    if qkey >= 0: p[o + qkey] = 1
    o += K
    p[o + head] = 1; o += W
    p[o + content] = 1; o += V + 1
    p[o] = mode
    return p


def act(g, j, p):
    h = np.tanh(g["W1"][j] @ p + g["b1"][j])             # the join lives here
    return h


def run(g, j, keys, vals, queries, freeze=False, trace=False):
    tape = [V] * W; head = 0
    for k, v in zip(keys, vals):
        for _ in range(W):
            h = act(g, j, perceive(k, v, -1, head, tape[head], 0))
            if (g["wg"][j] @ h) > 0:
                tape[head] = int(np.argmax(g["Wv"][j] @ h))
            if not freeze:
                head = int(np.clip(head + (int(np.argmax(g["Wm"][j] @ h)) - 1), 0, W - 1))
    outs = []
    for q in queries:
        out = 0
        for _ in range(W + 1):
            h = act(g, j, perceive(-1, -1, q, head, tape[head], 1))
            out = int(np.argmax(g["Wo"][j] @ h))
            if not freeze:
                head = int(np.clip(head + (int(np.argmax(g["Wm"][j] @ h)) - 1), 0, W - 1))
        outs.append(out)
    return (outs, tape) if trace else outs


def trial(g, j, rng, freeze=False):
    vals = list(rng.permutation(V)[:K])                  # DISTINCT values, one per key
    order = list(rng.permutation(K))                     # RANDOM arrival
    outs = run(g, j, order, [vals[k] for k in order], list(range(K)), freeze)
    return sum(int(outs[k] == vals[k]) for k in range(K))


def play(g, rng, trials=12):
    N = len(g["W1"])
    return np.array([sum(trial(g, j, rng) for _ in range(trials)) for j in range(N)], float)


def score(g, rng, n=400, freeze=False):
    N = len(g["W1"])
    return np.mean([trial(g, int(rng.integers(N)), rng, freeze) for _ in range(n)]) / K


def evolve(gens=600, N=44, mut=0.2, cull=0.25, seed=1):
    rng = np.random.default_rng(seed); g = new_pop(N, rng)
    for t in range(gens):
        en = play(g, rng)
        Kc = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for key in ("W1", "b1", "Wo", "Wv", "wg", "Wm"):
                m = rng.random(g[key][pa].shape) < 0.5
                g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"Content-addressed retrieval with a hidden layer (H={H}). {K} pairs, distinct values.", flush=True)
    print(f"random arrival; guess-one-value floor = 1/K = {1/K:.2f}; perfect = 1.0", flush=True)
    print("=" * 70, flush=True)
    sc = [score(evolve(gens=600, seed=s), rng) for s in range(3)]
    print(f"  retrieval = {np.mean(sc):.2f}  (per seed: {[round(x,2) for x in sc]})", flush=True)
    print("=" * 70, flush=True)
    print("trace (did it invent key->location storage?):", flush=True)
    g = evolve(gens=700, seed=1)
    en = play(g, np.random.default_rng(5), trials=40); j = int(np.argmax(en))
    for _ in range(6):
        vals = list(rng.permutation(V)[:K]); order = list(rng.permutation(K))
        outs, tape = run(g, j, order, [vals[k] for k in order], list(range(K)), trace=True)
        ok = sum(int(outs[k] == vals[k]) for k in range(K))
        print(f"    val(by key)={vals} arrival={order} -> recalled={outs} ({ok}/{K})  tape={tape}", flush=True)
    print(f"  ablation: movable={score(g, rng):.2f}  frozen={score(g, rng, freeze=True):.2f}", flush=True)
