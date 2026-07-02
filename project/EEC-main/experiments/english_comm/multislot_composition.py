"""MULTISLOT COMPOSITION -- push vocab AND fluency up with what we know.

Composition turns ONE huge-vocab problem into S independent small ones. A meaning is an S-tuple
(c1..cS), ci in [0,V), so effective vocabulary M = V^S. The organism learns per-slot machinery
(S associations of V x V, shared across all the exponentially-many meanings); params are O(S*V^2),
independent of M. Per-slot reward (conveying one attribute right partially helps). The world samples
random tuples; comprehension is measured on FRESH random tuples (at large M almost all unseen ->
zero-shot generalisation over the whole exponential space).

Reports per-slot accuracy and FULL-meaning comprehension (all S slots right) as M = V^S explodes.
full = perslot^S, so the lever for fluent huge-vocab is driving per-slot accuracy high.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P_NATIVE = 0.7
N = int(os.environ.get("MS_N", "48")); BUDGET = int(os.environ.get("MS_BUDGET", "1500"))
SEEDS = int(os.environ.get("MS_SEEDS", "3")); V = int(os.environ.get("MS_V", "8"))
FULL_BONUS = float(os.environ.get("MS_FULLBONUS", "0"))          # world-consequence: partner acts only on FULL meaning
SS = [int(x) for x in os.environ.get("MS_SS", "1,2,3,4,5,6").split(",")]

LOG = open(os.path.join(HERE, "multislot_composition_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def breed(P, en, rng, mut=0.22):
    Ng = len(en); order = np.argsort(en); worst = order[:int(0.25 * Ng)]; top = order[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        msk = rng.random(P[pa].shape) < 0.5
        P[w] = np.where(msk, P[pa], P[pb]) + rng.normal(0, mut * 0.6, P[pa].shape)


def run(S, seed):
    rng = np.random.default_rng(seed); B = N * 16
    A = rng.normal(0, 0.3, (N, S, V, V))                          # per-slot V x V association
    for t in range(BUDGET):
        emit = A.argmax(3); hear = A.argmax(2)                    # emit[n,s,c]=signal, hear[n,s,sig]=value
        en = np.zeros(N); c = rng.integers(V, size=(B, S))        # random S-tuple meanings
        sp = rng.integers(N, size=B); ls = rng.integers(N, size=B)
        nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5
        rr = ~nat; allc = np.ones(B, bool)
        for s in range(S):
            cs = c[:, s]; sig = emit[sp, s, cs]
            cor = hear[ls, s, sig] == cs; allc &= cor
            ok = cor & rr
            np.add.at(en, sp[ok], 1.0); np.add.at(en, ls[ok], 1.0)
            n0 = (hear[ls, s, cs] == cs) & nat & half; np.add.at(en, ls[n0], 1.0)   # native speaks identity
            n1 = (emit[sp, s, cs] == cs) & nat & ~half; np.add.at(en, sp[n1], 1.0)  # speak to native
        if FULL_BONUS:                                            # partner acts correctly only on FULL meaning
            fb = allc & rr; np.add.at(en, sp[fb], FULL_BONUS); np.add.at(en, ls[fb], FULL_BONUS)
        breed(A, en, rng)
    return evaluate(A, S)


def evaluate(A, S, T=6000):
    emit = A.argmax(3); hear = A.argmax(2); Ng = len(A)
    c = np.random.default_rng(7).integers(V, size=(T, S))
    sp = np.random.default_rng(8).integers(Ng, size=T); ls = np.random.default_rng(9).integers(Ng, size=T)
    slot_ok = np.ones(T, bool); per = []
    for s in range(S):
        cs = c[:, s]; dec = hear[ls, s, emit[sp, s, cs]]
        m = (dec == cs); per.append(m.mean()); slot_ok &= m
    return float(np.mean(per)), float(slot_ok.mean())            # per-slot acc, full-meaning comp


if __name__ == "__main__":
    out(f"MULTISLOT COMPOSITION: V={V} values/slot, slots S -> vocab M=V^S. pop {N}, budget {BUDGET}, "
        f"{SEEDS} seeds. params O(S*V^2), independent of M.")
    out(f"{'S':>3} {'M=V^S':>10} {'chance':>9} | {'per-slot acc':>14} | {'FULL-meaning comp':>18}")
    out("=" * 70)
    res = {}
    for S in SS:
        r = [run(S, s) for s in range(SEEDS)]
        ps, fm = [x[0] for x in r], [x[1] for x in r]
        M = V ** S; res[S] = (np.mean(ps), np.mean(fm), M)
        out(f"{S:>3} {M:>10} {1/M:>9.2e} | {ms(ps):>14} | {ms(fm):>18}")
    out("=" * 70)
    out("READING: per-slot acc ~flat across S (each slot is the same small problem) => vocab M=V^S")
    out("explodes for free; FULL-meaning = perslot^S, so driving per-slot accuracy up is the fluency lever.")
    for S in SS:
        out(f"  S={S}: vocab {res[S][2]:>8}  per-slot {res[S][0]:.3f}  full-meaning {res[S][1]:.3f}")
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ms = [res[S][2] for S in SS]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ms, [res[S][0] for S in SS], "o-", color="#2a8", lw=2.5, label="per-slot accuracy")
    ax.plot(Ms, [res[S][1] for S in SS], "s-", color="#1b5e9e", lw=2.5, label="full-meaning comprehension")
    ax.set_xscale("log"); ax.set_xlabel("effective vocabulary M = V^S (log)"); ax.set_ylabel("accuracy")
    ax.set_title("Composition: vocabulary explodes, per-slot fluency holds", weight="bold")
    ax.legend(); ax.grid(alpha=.3); ax.set_ylim(0, 1)
    plt.tight_layout(); p = os.path.join(HERE, "multislot_composition.png"); plt.savefig(p, dpi=120); print("saved", p)
