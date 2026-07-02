"""DOES SYNTAX EMERGE WHEN THE WORLD HAS RELATIONS?

No grammar is wired in. The world contains EVENTS = (agent, action, target), where agent
and target are drawn from the SAME pool of entities. So "wolf chases deer" and "deer chases
wolf" are the same symbols in different roles with opposite meaning. Survival depends on the
listener recovering WHO DID WHAT TO WHOM.

The channel is sequential: the speaker emits a sequence of L symbols (one per position) and
the listener reads the sequence. The per-position speak maps and the listener's role-readers
are ALL free and randomly initialised -- nothing says "position 0 = agent". If a consistent
positional convention (word order) emerges, it is emergent syntax, driven only by the world's
relational ambiguity.

Controls/tests:
  - swap test: can the protocol distinguish "A acts B" from "B acts A"? (needs role marking)
  - scramble test: shuffle message positions before decoding -> if accuracy drops, ORDER is
    load-bearing (the protocol is genuinely using word order, not a bag of words)
  - order-free ceiling: a listener given only the BAG of symbols (positions destroyed) cannot
    exceed chance on the agent/target distinction -- proves the world REQUIRES order
  - held-out events: generalisation to relations never trained on (compositionality)
"""
import numpy as np

E, A, L = 5, 3, 3            # entities, actions, sequence length (slots)
V = E + A                    # symbols available (meanings NOT pre-assigned to symbols)
D = 2 * E + A                # event encoding dim: [agent onehot | action onehot | target onehot]


def all_events():
    return [(ag, act, tg) for ag in range(E) for act in range(A) for tg in range(E) if ag != tg]


def ev_vec(ag, act, tg):
    v = np.zeros(D); v[ag] = 1; v[E + act] = 1; v[E + A + tg] = 1
    return v


EVV = {e: ev_vec(*e) for e in all_events()}


def new_pop(N, rng, bag=False):
    rd = V if bag else L * V                                # bag listener: positions destroyed
    g = dict(spk=rng.normal(0, 0.4, (N, L, V, D)),
             ag=rng.normal(0, 0.4, (N, E, rd)),
             act=rng.normal(0, 0.4, (N, A, rd)),
             tg=rng.normal(0, 0.4, (N, E, rd)), bag=bag)
    return g


def emit(g, i, e):
    v = EVV[e]
    return [int(np.argmax(g["spk"][i, l] @ v)) for l in range(L)]


def seq_onehot(msg, scramble_rng=None, bag=False):
    if scramble_rng is not None:
        msg = list(msg); scramble_rng.shuffle(msg)
    if bag:                                                # set of symbols, no position info
        x = np.zeros(V)
        for s in msg: x[s] += 1
        return x
    x = np.zeros(L * V)
    for l, s in enumerate(msg):
        x[l * V + s] = 1
    return x


def decode(g, j, msg, scramble_rng=None):
    x = seq_onehot(msg, scramble_rng, g.get("bag", False))
    return (int(np.argmax(g["ag"][j] @ x)), int(np.argmax(g["act"][j] @ x)), int(np.argmax(g["tg"][j] @ x)))


def play(g, rng, events, rounds=6):
    """energy is GRADED by how many roles get through (conveying more of the event = surviving
    more). This makes the search reachable; the agent/target distinction -- the part that needs
    ORDER -- is the last and hardest thing to earn."""
    N = len(g["spk"]); en = np.zeros(N)
    for _ in range(rounds):
        perm = rng.permutation(N)
        for k in range(0, N - 1, 2):
            i, j = perm[k], perm[k + 1]
            for sp, ls in ((i, j), (j, i)):
                ag, act, tg = events[rng.integers(len(events))]
                dag, dact, dtg = decode(g, ls, emit(g, sp, (ag, act, tg)))
                c = (dag == ag) + (dact == act) + (dtg == tg)
                en[sp] += c; en[ls] += c
    return en


def per_role(g, events, rng, n=800):
    N = len(g["spk"]); a = ac = t = 0
    for _ in range(n):
        i, j = rng.integers(N), rng.integers(N); ag, act, tg = events[rng.integers(len(events))]
        dag, dact, dtg = decode(g, j, emit(g, i, (ag, act, tg)))
        a += (dag == ag); ac += (dact == act); t += (dtg == tg)
    return a / n, ac / n, t / n


def role_acc(g, events, rng, scramble=False, n=600):
    N = len(g["spk"]); ok = 0
    sr = np.random.default_rng(0) if scramble else None
    for _ in range(n):
        i, j = rng.integers(N), rng.integers(N)
        e = events[rng.integers(len(events))]
        if decode(g, j, emit(g, i, e), sr) == e:
            ok += 1
    return ok / n


def swap_acc(g, events, rng, n=600):
    """fraction of times the protocol distinguishes (a,act,b) from (b,act,a) correctly."""
    N = len(g["spk"]); ok = 0; tot = 0
    for _ in range(n):
        a, b = rng.integers(E), rng.integers(E)
        if a == b: continue
        act = rng.integers(A); i, j = rng.integers(N), rng.integers(N)
        e1 = (a, act, b)
        if e1 not in EVV: continue
        d1 = decode(g, j, emit(g, i, e1))
        ok += (d1 == e1); tot += 1
    return ok / max(tot, 1)


def evolve(gens=500, N=48, mut=0.22, cull=0.25, seed=1, repro="sexual", train=None, bag=False):
    rng = np.random.default_rng(seed); g = new_pop(N, rng, bag=bag)
    ev = train if train is not None else all_events()
    for t in range(gens):
        en = play(g, rng, ev)
        Kc = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            if repro == "clone":
                p = int(top[rng.integers(len(top))])
                for key in ("spk", "ag", "act", "tg"):
                    g[key][w] = g[key][p] + rng.normal(0, mut, g[key][p].shape)
            else:
                pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
                for key in ("spk", "ag", "act", "tg"):
                    m = rng.random(g[key][pa].shape) < 0.5
                    g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    allev = all_events()
    print(f"Relational world: {E} entities, {A} actions, {len(allev)} events; channel = {L} symbols, {V} vocab")
    print(f"chance role accuracy ~ {1.0/ (E*A*(E-1)):.3f};  chance swap ~ {1.0/(E*A*(E-1)):.3f}")
    print("=" * 72)
    print("T1 — train on ALL events; does order-based role marking emerge?")
    for repro in ["sexual"]:
        for sd in range(2):
            g = evolve(gens=700, seed=sd, repro=repro)
            pa, pac, pt = per_role(g, allev, rng)
            ra = role_acc(g, allev, rng); sw = swap_acc(g, allev, rng)
            scr = role_acc(g, allev, rng, scramble=True)
            print(f"  {repro} seed{sd}: per-role agent={pa:.2f} action={pac:.2f} target={pt:.2f} | "
                  f"full={ra:.2f} swap={sw:.2f} scrambled={scr:.2f}")
    print("=" * 72)
    print("T2 — HELD-OUT events (generalisation = compositional syntax)")
    rng2 = np.random.default_rng(7); ev2 = allev[:]; rng2.shuffle(ev2)
    test = ev2[:12]; train = ev2[12:]
    g = evolve(gens=600, seed=1, repro="sexual", train=train)
    print(f"  train role_acc={role_acc(g, train, rng):.2f}   HELD-OUT role_acc={role_acc(g, test, rng):.2f}")
    print(f"  held-out scrambled={role_acc(g, test, rng, scramble=True):.2f}  (low => uses order)")
    print("=" * 72)
    print("T3 — does the WORLD require order? sequential channel vs order-free BAG")
    print("  (bag listener gets the SET of symbols; it cannot tell 'A acts B' from 'B acts A')")
    for bag in [False, True]:
        ra, sw = [], []
        for sd in range(3):
            g = evolve(gens=600, seed=sd, repro="sexual", bag=bag)
            ra.append(role_acc(g, allev, rng)); sw.append(swap_acc(g, allev, rng))
        tag = "BAG (no order)" if bag else "SEQUENTIAL"
        print(f"  {tag:16}: full role_acc={np.mean(ra):.2f}   swap_acc={np.mean(sw):.2f}")
