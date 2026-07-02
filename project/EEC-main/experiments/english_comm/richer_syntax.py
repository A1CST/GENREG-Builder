"""RICHER WORLD -> richer syntax. No structure wired in.

The relational world gave word order because roles needed marking. Now entities are
COMPOSITE: each is (attribute, type), and several things share a type. So survival needs
not just who-did-what-to-whom but WHICH one -- "big wolf chases small deer" vs "small wolf
chases big deer" are the same symbols with different meaning. Disambiguating requires the
listener to BIND each attribute to the right noun. A flat bag cannot; only grouped / ordered
structure can. If constituent structure (phrase grouping) emerges, it is emergent richer syntax.

Event = ( (attr_a, type_a), action, (attr_t, type_t) ).
Channel = L sequential symbol slots. Survival graded by how many of the 5 components arrive.
Everything (per-slot speak maps, the 5 role-readers) is free and random.
"""
import numpy as np

AT, TY, ACT, L = 3, 4, 3, 6          # attributes, types, actions, sequence slots
V = AT + TY + ACT                    # symbols (meanings NOT pre-assigned)
D = AT + TY + ACT + AT + TY          # event encoding


def ents():
    return [(a, t) for a in range(AT) for t in range(TY)]


def all_events():
    ev = []
    for a in ents():
        for act in range(ACT):
            for b in ents():
                if a != b:
                    ev.append((a[0], a[1], act, b[0], b[1]))
    return ev


def ev_vec(aa, ta, act, at, tt):
    v = np.zeros(D); o = 0
    v[o + aa] = 1; o += AT
    v[o + ta] = 1; o += TY
    v[o + act] = 1; o += ACT
    v[o + at] = 1; o += AT
    v[o + tt] = 1
    return v


ALL = all_events()
EVV = {e: ev_vec(*e) for e in ALL}


def new_pop(N, rng, bag=False):
    rd = V if bag else L * V
    return dict(spk=rng.normal(0, 0.4, (N, L, V, D)),
                aa=rng.normal(0, 0.4, (N, AT, rd)), ta=rng.normal(0, 0.4, (N, TY, rd)),
                act=rng.normal(0, 0.4, (N, ACT, rd)),
                at=rng.normal(0, 0.4, (N, AT, rd)), tt=rng.normal(0, 0.4, (N, TY, rd)), bag=bag)


def emit(g, i, e):
    v = EVV[e]
    return [int(np.argmax(g["spk"][i, l] @ v)) for l in range(L)]


def feats(msg, scramble_rng=None, bag=False):
    if scramble_rng is not None:
        msg = list(msg); scramble_rng.shuffle(msg)
    if bag:
        x = np.zeros(V)
        for s in msg: x[s] += 1
        return x
    x = np.zeros(L * V)
    for l, s in enumerate(msg):
        x[l * V + s] = 1
    return x


def decode(g, j, msg, scramble_rng=None):
    x = feats(msg, scramble_rng, g.get("bag", False))
    return (int(np.argmax(g["aa"][j] @ x)), int(np.argmax(g["ta"][j] @ x)), int(np.argmax(g["act"][j] @ x)),
            int(np.argmax(g["at"][j] @ x)), int(np.argmax(g["tt"][j] @ x)))


def play(g, rng, events, rounds=6):
    N = len(g["spk"]); en = np.zeros(N)
    for _ in range(rounds):
        perm = rng.permutation(N)
        for k in range(0, N - 1, 2):
            i, j = perm[k], perm[k + 1]
            for sp, ls in ((i, j), (j, i)):
                e = events[rng.integers(len(events))]
                d = decode(g, ls, emit(g, sp, e))
                # credit PER ENTITY: an entity counts only if attribute AND type are both right
                # (and bound to the correct role) -- so wrong binding earns nothing.
                agent_ok = (d[0] == e[0] and d[1] == e[1])
                act_ok = (d[2] == e[2])
                target_ok = (d[3] == e[3] and d[4] == e[4])
                c = agent_ok + act_ok + target_ok
                en[sp] += c; en[ls] += c
    return en


def metrics(g, events, rng, n=900, scramble=False):
    N = len(g["spk"]); comp = np.zeros(5); full = 0; bind = bind_n = 0
    sr = np.random.default_rng(0) if scramble else None
    for _ in range(n):
        i, j = rng.integers(N), rng.integers(N); e = events[rng.integers(len(events))]
        d = decode(g, j, emit(g, i, e), sr)
        for r in range(5): comp[r] += (d[r] == e[r])
        full += (d == e)
        # binding: when the two attributes differ, are BOTH bound to the right noun?
        if e[0] != e[3]:
            bind_n += 1; bind += (d[0] == e[0] and d[3] == e[3])
    return comp / n, full / n, bind / max(bind_n, 1)


def evolve(gens=800, N=48, mut=0.22, cull=0.25, seed=1, repro="sexual", bag=False, train=None):
    rng = np.random.default_rng(seed); g = new_pop(N, rng, bag=bag)
    ev = train if train is not None else ALL
    for t in range(gens):
        en = play(g, rng, ev)
        Kc = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            if repro == "clone":
                p = int(top[rng.integers(len(top))])
                for key in ("spk", "aa", "ta", "act", "at", "tt"):
                    g[key][w] = g[key][p] + rng.normal(0, mut, g[key][p].shape)
            else:
                pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
                for key in ("spk", "aa", "ta", "act", "at", "tt"):
                    m = rng.random(g[key][pa].shape) < 0.5
                    g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"Richer world: {AT} attrs x {TY} types = {AT*TY} entities, {ACT} actions, {len(ALL)} events")
    print(f"  channel {L} slots, {V} symbols; 5 components per event; chance full ~ {1/len(ALL):.4f}")
    print("=" * 74)
    print("T1 — does constituent structure (attribute binding) emerge?")
    for sd in range(2):
        g = evolve(gens=800, seed=sd)
        comp, full, bind = metrics(g, ALL, rng)
        _, fs, _ = metrics(g, ALL, rng, scramble=True)
        print(f"  seed{sd}: components [aAttr={comp[0]:.2f} aType={comp[1]:.2f} act={comp[2]:.2f} "
              f"tAttr={comp[3]:.2f} tType={comp[4]:.2f}]")
        print(f"          full={full:.2f}  BINDING={bind:.2f}  scrambled_full={fs:.2f}")
    print("=" * 74)
    print("T2 — does the WORLD require structure? sequential vs order-free BAG")
    for bag in [False, True]:
        fl, bd = [], []
        for sd in range(3):
            g = evolve(gens=800, seed=sd, bag=bag)
            _, full, bind = metrics(g, ALL, rng)
            fl.append(full); bd.append(bind)
        tag = "BAG (no order)" if bag else "SEQUENTIAL"
        print(f"  {tag:16}: full={np.mean(fl):.2f}  binding={np.mean(bd):.2f}")
    print("=" * 74)
    print("T3 — sample (entity = attr+type; can it bind correctly?):")
    g = evolve(gens=800, seed=0)
    AN = ["big", "small", "red"]; TN = ["wolf", "deer", "bird", "fish"]; VB = ["chase", "eat", "see"]
    for _ in range(7):
        e = ALL[rng.integers(len(ALL))]; d = decode(g, 1, emit(g, 0, e))
        say = f"{AN[e[0]]} {TN[e[1]]} {VB[e[2]]} {AN[e[3]]} {TN[e[4]]}"
        got = f"{AN[d[0]]} {TN[d[1]]} {VB[d[2]]} {AN[d[3]]} {TN[d[4]]}"
        print(f'    "{say:34}" -> heard "{got:34}"  {"OK" if d==e else "x"}')
