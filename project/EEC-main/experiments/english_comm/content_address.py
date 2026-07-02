"""CONTENT-ADDRESSED memory: find by WHAT, not where.

Position-addressing (J) retrieves the i-th thing. Real memory retrieves the thing associated
with a KEY. To force content- over position-addressing, the (key,value) pairs are presented in
RANDOM order -- so arrival position carries no information; only the key does. A feedforward
organism with a persistent tape must therefore invent a scheme that puts each value at a location
DETERMINED BY ITS KEY (its own hash), and recompute that location to read.

No lookup hardware is given: the organism only perceives (current key/value or query, head
position, cell content) and acts (write?, what, move +/-1). The key->location policy is its own.

Graded: store K pairs, then query ALL K keys; credit = correct retrievals (partial progress pays).
Control: RANDOM arrival (content-addressing required) vs FIXED arrival (position would suffice).
Ablation: freeze the head (deny it the ability to go to the key's place).
"""
import numpy as np

K, V = 3, 4                              # number of pairs/keys, value alphabet
W = K                                    # tape cells


def IN():
    return K + V + K + W + (V + 1) + 1   # key, value, query-key, head, content(+empty), mode


def new_pop(N, rng):
    n = IN()
    return dict(Wo=rng.normal(0, 0.5, (N, V, n)), Wv=rng.normal(0, 0.5, (N, V, n)),
                wg=rng.normal(0, 0.5, (N, n)), Wm=rng.normal(0, 0.5, (N, 3, n)))


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


def run(g, j, keys, vals, queries, freeze=False, trace=False):
    tape = [V] * W; head = 0
    for k, v in zip(keys, vals):                     # WRITE phase (keys/vals already in arrival order)
        for _ in range(W):                           # steps to navigate to a key-determined cell, then write
            p = perceive(k, v, -1, head, tape[head], 0)
            if (g["wg"][j] @ p) > 0:
                tape[head] = int(np.argmax(g["Wv"][j] @ p))
            if not freeze:
                head = int(np.clip(head + (int(np.argmax(g["Wm"][j] @ p)) - 1), 0, W - 1))
    outs = []
    for q in queries:                                # READ phase: query by KEY, navigate, answer
        out = 0
        for _ in range(W + 1):
            p = perceive(-1, -1, q, head, tape[head], 1)
            out = int(np.argmax(g["Wo"][j] @ p))
            if not freeze:
                head = int(np.clip(head + (int(np.argmax(g["Wm"][j] @ p)) - 1), 0, W - 1))
        outs.append(out)
    return (outs, tape) if trace else outs


def trial(g, j, rng, fixed=False, freeze=False):
    vals = [int(rng.integers(V)) for _ in range(K)]      # value for key 0..K-1
    order = list(range(K))
    if not fixed:
        rng.shuffle(order)                               # RANDOM arrival -> position is useless
    keys = order; kv = [vals[k] for k in order]
    queries = list(range(K)); rng.shuffle(queries)
    outs = run(g, j, keys, kv, queries, freeze)
    return sum(int(outs[i] == vals[queries[i]]) for i in range(K))


def play(g, rng, trials=12, fixed=False):
    N = len(g["Wo"])
    return np.array([sum(trial(g, j, rng, fixed) for _ in range(trials)) for j in range(N)], float)


def score(g, rng, n=400, fixed=False, freeze=False):
    N = len(g["Wo"])
    return np.mean([trial(g, int(rng.integers(N)), rng, fixed, freeze) for _ in range(n)]) / K


def evolve(gens=500, N=44, mut=0.22, cull=0.25, seed=1, fixed=False):
    rng = np.random.default_rng(seed); g = new_pop(N, rng)
    for t in range(gens):
        en = play(g, rng, fixed=fixed)
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
    print(f"Content-addressed retrieval: store {K} (key,value) pairs, recall each by KEY.", flush=True)
    print(f"per-query chance = {1/V:.2f}. Pairs arrive in RANDOM order, so position cannot be the address.", flush=True)
    print("=" * 70, flush=True)
    print("T1 — RANDOM arrival (content-addressing required) vs FIXED arrival (position suffices)", flush=True)
    for fixed in [True, False]:
        sc = [score(evolve(gens=500, seed=s, fixed=fixed), rng, fixed=fixed) for s in range(3)]
        tag = "FIXED arrival" if fixed else "RANDOM arrival (content-addr)"
        print(f"  {tag:30}: retrieval = {np.mean(sc):.2f}", flush=True)
    print("=" * 70, flush=True)
    print("T2 — what scheme did it invent? (trace, random arrival)", flush=True)
    g = evolve(gens=600, seed=1, fixed=False)
    en = play(g, np.random.default_rng(5), trials=40); j = int(np.argmax(en))
    for _ in range(5):
        vals = [int(rng.integers(V)) for _ in range(K)]; order = list(range(K)); rng.shuffle(order)
        outs, tape = run(g, j, order, [vals[k] for k in order], list(range(K)), trace=True)
        ok = sum(int(outs[q] == vals[q]) for q in range(K))
        print(f"    vals(by key)={vals} arrival={order} -> recalled={outs} ({ok}/{K})  tape={tape}", flush=True)
    print("=" * 70, flush=True)
    print("T3 — ablation: freeze head (deny going to the key's place)", flush=True)
    print(f"  movable={score(g, rng):.2f}   frozen={score(g, rng, freeze=True):.2f}", flush=True)
