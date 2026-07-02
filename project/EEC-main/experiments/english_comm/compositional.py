"""COMPOSITIONAL English: two-word messages [attribute, object].
Tests whether agents use words COMPOSITIONALLY -- i.e. whether, having learned each
attribute word and each object word from a SUBSET of combinations, they can handle
HELD-OUT combinations zero-shot. Compositional generalisation is the hallmark of grammar.
"""
import numpy as np

ATTR = ["big", "small", "red", "cold"]
OBJ = ["food", "water", "home", "friend", "rock"]
DIST = ["z3", "qk", "vx", "wb"]
WORDS = ATTR + OBJ + DIST
A, O, V = len(ATTR), len(OBJ), len(WORDS)
A0, O0 = 0, A                                   # english word index: attr a -> a ; obj o -> A+o


def new_pop(N, rng):
    return dict(sa=rng.normal(0, 0.3, (N, A, V)), so=rng.normal(0, 0.3, (N, O, V)),
                la=rng.normal(0, 0.3, (N, V, A)), lo=rng.normal(0, 0.3, (N, V, O)))


def _prior(shape, idx_map, anchor):
    pr = np.zeros(shape)
    if anchor > 0:
        for k, w in idx_map:
            pr[k, w] = anchor
    return pr


def eff(g, anchor):
    sa = g["sa"] + _prior((A, V), [(a, A0 + a) for a in range(A)], anchor)
    so = g["so"] + _prior((O, V), [(o, O0 + o) for o in range(O)], anchor)
    la = g["la"] + _prior((V, A), [(A0 + a, a) for a in range(A)], anchor)
    lo = g["lo"] + _prior((V, O), [(O0 + o, o) for o in range(O)], anchor)
    return sa, so, la, lo


def play(g, anchor, rng, combos, rounds=8):
    N = len(g["sa"]); en = np.zeros(N)
    sa, so, la, lo = eff(g, anchor)
    succ = tot = 0
    for _ in range(rounds):
        perm = rng.permutation(N)
        for k in range(0, N - 1, 2):
            i, j = perm[k], perm[k + 1]
            a, o = combos[rng.integers(len(combos))]
            w1 = int(np.argmax(sa[i, a])); w2 = int(np.argmax(so[i, o]))   # speak 2 words
            ah = int(np.argmax(la[j, w1])); oh = int(np.argmax(lo[j, w2])) # listen 2 words
            ok = (ah == a and oh == o)
            if ok: en[i] += 1; en[j] += 1
            succ += ok; tot += 1
    return en, succ / max(tot, 1)


def accuracy(g, anchor, combos):
    sa, so, la, lo = eff(g, anchor)
    emit_a = sa.argmax(2); emit_o = so.argmax(2)          # (N,A),(N,O)
    dec_a = la.argmax(2); dec_o = lo.argmax(2)            # (N,V),(N,V)
    N = len(g["sa"]); ok = 0; n = 0
    for i in range(N):
        for (a, o) in combos:
            w1 = emit_a[i, a]; w2 = emit_o[i, o]
            if dec_a[i, w1] == a and dec_o[i, w2] == o: ok += 1
            n += 1
    return ok / n


def evolve(anchor=2.0, repro="sexual", gens=300, N=40, mut=0.25, cull=0.25, seed=1, holdout=6):
    rng = np.random.default_rng(seed)
    g = new_pop(N, rng)
    allc = [(a, o) for a in range(A) for o in range(O)]
    rng.shuffle(allc)
    test = allc[:holdout]; train = allc[holdout:]          # never trained on `test`
    for t in range(gens):
        en, _ = play(g, anchor, rng, train)
        K = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:K]; top = order[N - max(2, N // 3):]
        for w in worst:
            if repro == "clone":
                p = int(top[rng.integers(len(top))])
                for key in ("sa", "so", "la", "lo"):
                    g[key][w] = g[key][p] + rng.normal(0, mut, g[key][p].shape)
            else:
                pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
                for key in ("sa", "so", "la", "lo"):
                    m = rng.random(g[key][pa].shape) < 0.5
                    g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g, train, test


def sample(g, anchor, combos, rng, n=6):
    sa, so, la, lo = eff(g, anchor); i, j = 0, 1; out = []
    for _ in range(n):
        a, o = combos[rng.integers(len(combos))]
        w1 = int(np.argmax(sa[i, a])); w2 = int(np.argmax(so[i, o]))
        ah = int(np.argmax(la[j, w1])); oh = int(np.argmax(lo[j, w2]))
        out.append(f'    "{ATTR[a]:5} {OBJ[o]:6}" -> says "{WORDS[w1]} {WORDS[w2]}" '
                   f'-> heard "{ATTR[ah]:5} {OBJ[oh]:6}"  {"OK" if (ah==a and oh==o) else "x"}')
    return "\n".join(out)


if __name__ == "__main__":
    print("COMPOSITIONAL English — train on a subset of (attr,obj) combos, test on held-out")
    print(f"  {A}x{O}=20 combos; 6 held out (never trained). Generalising to them = compositional.")
    print(f"  {'condition':<26}{'train acc':>10}{'HELD-OUT acc':>14}")
    for anchor in [0.0, 2.0]:
        for repro in ["clone", "sexual"]:
            tr_a, te_a = [], []
            for sd in range(4):
                g, train, test = evolve(anchor=anchor, repro=repro, gens=300, seed=sd)
                tr_a.append(accuracy(g, anchor, train)); te_a.append(accuracy(g, anchor, test))
            tag = f"anchor={anchor:.0f} {repro}"
            print(f"  {tag:<26}{np.mean(tr_a):>10.2f}{np.mean(te_a):>14.2f}")
    print()
    g, train, test = evolve(anchor=2.0, repro="sexual", gens=300, seed=1)
    print("  zero-shot on HELD-OUT combinations (anchor=2, sexual) — never seen in training:")
    print(sample(g, 2.0, test, np.random.default_rng(4)))
