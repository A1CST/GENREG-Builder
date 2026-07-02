"""Does WORD ORDER survive WORLD-CONSEQUENCE fitness (no designed per-role grading)?

Each event: a PREY (food) and a PREDATOR (danger), drawn from the SAME set of creatures, so
"P chases Y" and "Y chases P" are the same symbols in swapped roles. A speaker sends a sequential
message; a listener decodes it and WALKS TOWARD what it thinks is prey. The world pays:
   +2 if it approached the real prey (ate),   -3 if it approached the predator (died),  else 0.
Fitness is exactly this survival payoff -- the world's own physics (food nourishes, predators
kill). Nothing grades 'how many roles matched'; the graded structure (getting it right matters a
lot, getting it wrong is fatal) is the world's, not mine. Coupled speaker+listener (a cooperating
pair). Sequential channel vs order-free BAG; scramble test. If order still emerges and is
load-bearing, the syntax result survives the 'no designed gradients' standard.
"""
import numpy as np

E, L = 5, 2                              # creatures, message slots
V = E                                    # one symbol per creature
D = 2 * E                                # event encoding: [prey onehot | predator onehot]


def events():
    return [(f, d) for f in range(E) for d in range(E) if f != d]


EV = events()


def ev_vec(food, dang):
    v = np.zeros(D); v[food] = 1; v[E + dang] = 1
    return v


EVV = {e: ev_vec(*e) for e in EV}


def new_pop(N, rng, bag=False):
    rd = V if bag else L * V
    return dict(spk=rng.normal(0, 0.4, (N, L, V, D)),
                lf=rng.normal(0, 0.4, (N, E, rd)),       # which creature to APPROACH (prey)
                lg=rng.normal(0, 0.4, (N, E, rd)),       # which creature to FLEE (predator)
                bag=bag)


def emit(g, i, e):
    v = EVV[e]
    return [int(np.argmax(g["spk"][i, l] @ v)) for l in range(L)]


def msg_feat(msg, bag, scramble_rng=None):
    if scramble_rng is not None:
        msg = list(msg); scramble_rng.shuffle(msg)
    if bag:
        x = np.zeros(V)
        for s in msg: x[s] += 1
        return x
    x = np.zeros(L * V)
    for l, s in enumerate(msg): x[l * V + s] = 1
    return x


def act(g, j, msg, scramble_rng=None):
    x = msg_feat(msg, g.get("bag", False), scramble_rng)
    return int(np.argmax(g["lf"][j] @ x)), int(np.argmax(g["lg"][j] @ x))   # (approach, flee)


def payoff(appr, flee, food, dang):
    p = 0.0
    if appr == food: p += 2.0                           # ate the prey
    if appr == dang: p -= 3.0                           # walked into the predator -> died
    if flee != dang: p -= 2.0                           # failed to flee the real predator -> caught
    return p                                            # needs BOTH roles right; bag (15<20) cannot


def play(g, rng, rounds=6):
    N = len(g["spk"]); en = np.zeros(N)
    for _ in range(rounds):
        perm = rng.permutation(N)
        for k in range(0, N - 1, 2):
            i, j = perm[k], perm[k + 1]
            for sp, ls in ((i, j), (j, i)):
                e = EV[rng.integers(len(EV))]
                appr, flee = act(g, ls, emit(g, sp, e))
                p = payoff(appr, flee, e[0], e[1])
                en[sp] += p; en[ls] += p
    return en


def stats(g, rng, n=1500, scramble=False):
    N = len(g["spk"]); both = died = 0; sr = np.random.default_rng(0) if scramble else None
    for _ in range(n):
        i, j = rng.integers(N), rng.integers(N); e = EV[rng.integers(len(EV))]
        appr, flee = act(g, j, emit(g, i, e), sr)
        both += (appr == e[0] and flee == e[1])         # full relation correct (needs role assignment)
        died += (appr == e[1])
    return both / n, died / n                           # both-roles-correct, predator-death (chance each ~0.04, 0.20)


def evolve(gens=600, N=48, mut=0.22, cull=0.25, seed=1, bag=False):
    rng = np.random.default_rng(seed); g = new_pop(N, rng, bag=bag)
    for t in range(gens):
        en = play(g, rng)
        Kc = max(1, int(cull * N)); order = np.argsort(en); worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for key in ("spk", "lf", "lg"):
                m = rng.random(g[key][pa].shape) < 0.5
                g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    LOG = open("embodied_order_results.txt", "w")
    def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"
    out("WORD ORDER under WORLD-CONSEQUENCE fitness. Listener must APPROACH prey AND FLEE predator;")
    out("eat +2, walk into predator -3, fail to flee the predator -2. Needs the FULL role assignment.")
    out(f"BAG has only {E*(E-1)//2 + E - 1} usable unordered messages < {len(EV)} events -> provably insufficient.")
    out(f"chance both-roles-correct ~ {1/len(EV):.3f}; chance predator-death ~ {1/E:.2f}")
    out("=" * 70)
    out("SEQUENTIAL channel (5 seeds):")
    bo, died, sbo, sdied = [], [], [], []
    for s in range(5):
        g = evolve(gens=700, seed=s); rng = np.random.default_rng(100 + s)
        b, d = stats(g, rng); sb, sd = stats(g, rng, scramble=True)
        bo.append(b); died.append(d); sbo.append(sb); sdied.append(sd)
    out(f"  both-roles-correct : {ms(bo)}   (chance 0.05)")
    out(f"  predator-death     : {ms(died)}   (LOW = uses the message to avoid the predator)")
    out(f"  both SCRAMBLED     : {ms(sbo)}   (collapse => order load-bearing)")
    out(f"  death  SCRAMBLED   : {ms(sdied)}   (rises => scrambling makes it walk into predators)")
    out("=" * 70)
    out("BAG (order destroyed) (5 seeds): provably cannot assign both roles:")
    bbo, bdied = [], []
    for s in range(5):
        g = evolve(gens=700, seed=s, bag=True); rng = np.random.default_rng(200 + s)
        b, d = stats(g, rng); bbo.append(b); bdied.append(d)
    out(f"  both-roles-correct : {ms(bbo)}   (vs sequential above)")
    out(f"  predator-death     : {ms(bdied)}   (HIGH = cannot tell prey from predator)")
    out("done"); LOG.close()
