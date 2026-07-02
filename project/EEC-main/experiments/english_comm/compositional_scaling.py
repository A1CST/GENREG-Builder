"""COMPOSITIONAL SCALING (#5) -- the compact-AND-decomposable representation: vocabulary for free.

Scaling synthesis: a flat lexicon needs O(M^2) to learn M meanings and can't generalize. The only rep
that is both compact and decomposable is COMPOSITION -- meanings built from a small shared set of parts.
Here meanings are COMPOSED: a meaning is (a, b) with a,b in [0,V), so M = V^2 meanings. A compositional
organism learns PER-SLOT machinery (two V x V associations, shared across all combinations); a flat
organism treats each (a,b) as an atomic symbol (M x M lookup, every combination independent).

The decisive scaling test = ZERO-SHOT: train on a SUBSET of combinations, measure comprehension on
HELD-OUT combinations never seen in training. Compositional should generalize (each slot value was
seen in *some* combo, so a new pairing is understood); flat cannot (an unseen meaning index is
untrained). And compositional params scale as O(V) per slot, not O(V^2) -- vocabulary for free.
Per-slot energy (conveying one attribute right partially helps); usage measured, never rewarded.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P_NATIVE = 0.7
N = int(os.environ.get("CO_N", "48")); BUDGET = int(os.environ.get("CO_BUDGET", "1500"))
SEEDS = int(os.environ.get("CO_SEEDS", "3"))
VS = [int(x) for x in os.environ.get("CO_VS", "4,6,8,10").split(",")]
TRAIN_FRAC = 0.7

LOG = open(os.path.join(HERE, "compositional_scaling_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def split(V, seed=0):
    rng = np.random.default_rng(1234 + seed)
    combos = np.array([(a, b) for a in range(V) for b in range(V)])
    perm = rng.permutation(len(combos)); ntr = int(TRAIN_FRAC * len(combos))
    return combos[perm[:ntr]], combos[perm[ntr:]]                 # seen, held-out


def breed(P, en, rng, mut=0.22):
    Ng = len(en); order = np.argsort(en); worst = order[:int(0.25 * Ng)]; top = order[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        msk = rng.random(P[pa].shape) < 0.5
        P[w] = np.where(msk, P[pa], P[pb]) + rng.normal(0, mut * 0.6, P[pa].shape)


# ---- compositional organism: genome A (N,2,V,V); per-slot speak=row argmax, hear=col argmax ----
def comp_run(V, seed):
    seen, held = split(V, seed); rng = np.random.default_rng(seed); B = N * 16
    A = rng.normal(0, 0.3, (N, 2, V, V))
    for t in range(BUDGET):
        e0, e1 = A[:, 0].argmax(2), A[:, 1].argmax(2); h0, h1 = A[:, 0].argmax(1), A[:, 1].argmax(1)
        en = np.zeros(N); idx = rng.integers(len(seen), size=B); a = seen[idx, 0]; b = seen[idx, 1]
        sp = rng.integers(N, size=B); ls = rng.integers(N, size=B)
        nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5
        s0, s1 = e0[sp, a], e1[sp, b]
        rr = ~nat
        ok0 = (h0[ls, s0] == a) & rr; ok1 = (h1[ls, s1] == b) & rr
        np.add.at(en, sp[ok0], 1.0); np.add.at(en, ls[ok0], 1.0)
        np.add.at(en, sp[ok1], 1.0); np.add.at(en, ls[ok1], 1.0)
        nsp = nat & half                                          # native speaks identity -> resident hears
        n0 = (h0[ls, a] == a) & nsp; n1 = (h1[ls, b] == b) & nsp
        np.add.at(en, ls[n0], 1.0); np.add.at(en, ls[n1], 1.0)
        nrs = nat & ~half                                         # resident speaks -> native understands iff identity
        r0 = (e0[sp, a] == a) & nrs; r1 = (e1[sp, b] == b) & nrs
        np.add.at(en, sp[r0], 1.0); np.add.at(en, sp[r1], 1.0)
        breed(A, en, rng)
    return eval_comp(A, seen, held)


def eval_comp(A, seen, held, T=6000):
    N = len(A); e0, e1 = A[:, 0].argmax(2), A[:, 1].argmax(2); h0, h1 = A[:, 0].argmax(1), A[:, 1].argmax(1)
    def acc(combos):
        idx = np.random.default_rng(7).integers(len(combos), size=T); a = combos[idx, 0]; b = combos[idx, 1]
        sp = np.random.default_rng(8).integers(N, size=T); ls = np.random.default_rng(9).integers(N, size=T)
        a2 = h0[ls, e0[sp, a]]; b2 = h1[ls, e1[sp, b]]
        return float(np.mean((a2 == a) & (b2 == b)))             # FULL-meaning comprehension
    return acc(seen), acc(held)


# ---- flat organism: meaning is atomic index in [0,M); genome A (N,M,M) lookup ----
def flat_run(V, seed):
    seen, held = split(V, seed); M = V * V; rng = np.random.default_rng(seed); B = N * 16
    smu = seen[:, 0] * V + seen[:, 1]; hmu = held[:, 0] * V + held[:, 1]
    A = rng.normal(0, 0.3, (N, M, M))
    for t in range(BUDGET):
        emit = A.argmax(2); hear = A.argmax(1); en = np.zeros(N)
        mu = smu[rng.integers(len(smu), size=B)]; sp = rng.integers(N, size=B); ls = rng.integers(N, size=B)
        nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5
        rr = ~nat; s = emit[sp, mu]; ok = (hear[ls, s] == mu) & rr
        np.add.at(en, sp[ok], 1.0); np.add.at(en, ls[ok], 1.0)
        ok2 = (hear[ls, mu] == mu) & nat & half; np.add.at(en, ls[ok2], 1.0)
        ok3 = (emit[sp, mu] == mu) & nat & ~half; np.add.at(en, sp[ok3], 1.0)
        breed(A, en, rng)
    emit = A.argmax(2); hear = A.argmax(1)
    def acc(mus):
        idx = np.random.default_rng(7).integers(len(mus), size=T_); mu = mus[idx]
        sp = np.random.default_rng(8).integers(N, size=T_); ls = np.random.default_rng(9).integers(N, size=T_)
        return float(np.mean(hear[ls, emit[sp, mu]] == mu))
    T_ = 6000
    return acc(smu), acc(hmu)


if __name__ == "__main__":
    out(f"COMPOSITIONAL SCALING (#5): zero-shot to HELD-OUT combinations. pop {N}, budget {BUDGET}, "
        f"{SEEDS} seeds, train {int(TRAIN_FRAC*100)}% of V^2 combos")
    out(f"{'V':>4} {'M=V^2':>6} {'chance':>7} | {'COMPOSITIONAL seen/HELDOUT':>30} | {'FLAT seen/HELDOUT':>26}")
    out("=" * 86)
    res = {}
    for V in VS:
        cs = [comp_run(V, s) for s in range(SEEDS)]; fs = [flat_run(V, s) for s in range(SEEDS)]
        cse, che = np.mean([c[0] for c in cs]), np.mean([c[1] for c in cs])
        fse, fhe = np.mean([f[0] for f in fs]), np.mean([f[1] for f in fs])
        res[V] = (che, fhe)
        out(f"{V:>4} {V*V:>6} {1/(V*V):>7.3f} | {cse:>13.3f} /{che:>13.3f} | {fse:>12.3f} /{fhe:>11.3f}")
    out("=" * 86)
    out("READING: COMPOSITIONAL held-out >> chance and >> FLAT held-out => zero-shot generalisation:")
    out("the compositional rep understands meanings it never saw -> vocabulary scales for free.")
    for V in VS:
        out(f"  V={V}: compositional held-out {res[V][0]:.3f}  vs  flat held-out {res[V][1]:.3f}  "
            f"(chance {1/(V*V):.3f})")
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = [V * V for V in VS]
    plt.figure(figsize=(8, 5))
    plt.plot(M, [res[V][0] for V in VS], "o-", color="#1b5e9e", lw=2.5, label="compositional (held-out)")
    plt.plot(M, [res[V][1] for V in VS], "s--", color="#c44", lw=2.5, label="flat (held-out)")
    plt.plot(M, [1 / m for m in M], ":", color="#999", label="chance")
    plt.xlabel("meaning-space size M = V²"); plt.ylabel("held-out (zero-shot) comprehension")
    plt.title("Composition = vocabulary for free (zero-shot to unseen meanings)", weight="bold")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    p = os.path.join(HERE, "compositional_scaling.png"); plt.savefig(p, dpi=120); print("saved", p)
