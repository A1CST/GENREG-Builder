"""COMPUTE SCALING -- is the fluency sag at large vocab COMPUTE-limited or a WALL?

The lookup (most-decomposable) rep scales best but still sags with K at fixed budget. The decisive
question for fluent English at large vocab: does more COMPUTE recover it? Track uniform two-way
comprehension over generations for several K. If each K climbs to high fluency given enough gens,
fluency-at-scale is compute-limited (it scales, costs more compute) and we can read the law:
gens-to-target vs K. If it plateaus below target, it is a wall.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P_NATIVE, U = 0.7, 0.25
N = int(os.environ.get("CS_N", "48")); BUDGET = int(os.environ.get("CS_BUDGET", "4500"))
SEEDS = int(os.environ.get("CS_SEEDS", "2")); EVERY = 300; TARGET = 0.70
KS = [int(x) for x in os.environ.get("CS_KS", "32,64,128").split(",")]

LOG = open(os.path.join(HERE, "compute_scaling_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def wcfg(K):
    freq = 1.0 / (np.arange(1, K + 1) ** 1.1); freq /= freq.sum()
    st = np.clip((1.0 / freq) / (1.0 / freq).mean(), 0.2, 12.0); st = st / st.mean()
    return freq, st


def play(A, rng, freq, st, K, B):
    Ng = len(A); emit = A.argmax(2); hear = A.argmax(1); en = np.zeros(Ng)
    p = (1 - U) * freq + U * (st / st.sum())
    m = rng.choice(K, size=B, p=p); sp = rng.integers(Ng, size=B); ls = rng.integers(Ng, size=B)
    nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5
    rr = ~nat; s = emit[sp, m]; ok = (hear[ls, s] == m) & rr
    np.add.at(en, sp[ok], st[m[ok]]); np.add.at(en, ls[ok], st[m[ok]])
    ok2 = (hear[ls, m] == m) & nat & half; np.add.at(en, ls[ok2], st[m[ok2]])
    ok3 = (emit[sp, m] == m) & nat & ~half; np.add.at(en, sp[ok3], st[m[ok3]])
    return en


def breed(A, en, rng, mut=0.22):
    Ng = len(en); order = np.argsort(en); worst = order[:int(0.25 * Ng)]; top = order[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        msk = rng.random(A[pa].shape) < 0.5
        A[w] = np.where(msk, A[pa], A[pb]) + rng.normal(0, mut * 0.6, A[pa].shape)


def uniform_comp(A, rng, K, T=8000):
    emit = A.argmax(2); hear = A.argmax(1); Ng = len(A)
    m = rng.integers(K, size=T); sp = rng.integers(Ng, size=T); ls = rng.integers(Ng, size=T)
    return float(np.mean(hear[ls, emit[sp, m]] == m))


def run(K, seed):
    freq, st = wcfg(K); rng = np.random.default_rng(seed); B = N * 12
    A = rng.normal(0, 0.3, (N, K, K)); traj = []
    for t in range(1, BUDGET + 1):
        breed(A, play(A, rng, freq, st, K, B), rng)
        if t % EVERY == 0:
            traj.append((t, uniform_comp(A, np.random.default_rng(9000 + seed), K)))
    return traj


if __name__ == "__main__":
    out(f"COMPUTE SCALING (lookup): uniform comprehension vs generations, pop {N}, {SEEDS} seeds, "
        f"budget {BUDGET}. target={TARGET}")
    checkpts = list(range(EVERY, BUDGET + 1, EVERY))
    out("gen:      " + "  ".join(f"{c:>5}" for c in checkpts))
    g2t = {}; means = {}
    for K in KS:
        trajs = [run(K, s) for s in range(SEEDS)]
        mean = np.mean([[p[1] for p in tr] for tr in trajs], axis=0); means[K] = mean
        out(f"K={K:>3}:    " + "  ".join(f"{v:5.3f}" for v in mean))
        hit = next((checkpts[i] for i, v in enumerate(mean) if v >= TARGET), None)
        g2t[K] = hit
    out("=" * 70)
    out(f"generations to reach {TARGET} uniform comprehension:")
    for K in KS:
        out(f"  K={K:>3}: {g2t[K] if g2t[K] else '>'+str(BUDGET)} gens"
            + (f"  ({g2t[K]/K:.1f} gens/word)" if g2t[K] else "  (target not reached)"))
    out("READING: all K reach target given enough gens => compute-limited (scales). gens-to-target vs K")
    out("         growing ~linearly => benign scaling law; super-linear/plateau => harder wall.")
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    for K in KS:
        plt.plot(checkpts, means[K], "o-", lw=2, label=f"K={K}")
    plt.axhline(TARGET, color="#999", ls="--", label=f"target {TARGET}")
    plt.xlabel("generations (compute)"); plt.ylabel("uniform comprehension")
    plt.title("Compute-to-fluency vs vocabulary size (lookup)", weight="bold"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); p = os.path.join(HERE, "compute_scaling.png"); plt.savefig(p, dpi=120); print("saved", p)
