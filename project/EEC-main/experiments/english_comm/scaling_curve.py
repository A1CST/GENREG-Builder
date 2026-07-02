"""SCALING CURVE -- convert the scaling MECHANISM into a scaling CURVE.

Claim under test: at a FIXED compute budget, the frequent-first CURRICULUM extends the reachable
vocabulary size -- cold-start collapses as the lexicon grows, the curriculum collapses later, so the
advantage WIDENS with scale (the reachability cliff from curriculum.py, now swept up the K axis).

Organism = the shared lexicon (the listener-fix winner): one association A (K x K), speak = row argmax,
hear = column argmax; natives anchor A toward identity (English). No embeddings needed (the shared
organism doesn't use them), so K scales freely. Landscape = criticality stakes + urgency u=0.25.
The interaction loop is VECTORISED so large K is feasible. Metric = two-way comprehension.

cold        : all K words active from gen 0, BUDGET gens.
freq_first  : grow the active vocab most-frequent-first over STAGES, carry the population, same BUDGET.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P_NATIVE, U = 0.7, 0.25
BUDGET, STAGES, N, INTER = 1800, 6, 60, None   # INTER set per-run = N*12
SEEDS = 3
KS = [24, 48, 96, 160, 240]

LOG = open(os.path.join(HERE, "scaling_curve_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def world(K):
    freq = 1.0 / (np.arange(1, K + 1) ** 1.1); freq /= freq.sum()
    st = np.clip((1.0 / freq) / (1.0 / freq).mean(), 0.2, 12.0); st = st / st.mean()
    rank = np.argsort(-freq)
    return freq, st, rank


def play(A, rng, active, freq, st, B):
    """Vectorised generation: B random interactions; energy accrues to genomes; no python pair loop."""
    Ng = len(A); emit = A.argmax(2); hear = A.argmax(1); en = np.zeros(Ng)
    pa = (1 - U) * freq[active] + U * (st[active] / st[active].sum()); pa = pa / pa.sum()
    m = rng.choice(active, size=B, p=pa)
    sp = rng.integers(Ng, size=B); ls = rng.integers(Ng, size=B)
    nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5
    rr = ~nat                                                   # resident<->resident
    s = emit[sp, m]; ok = (hear[ls, s] == m) & rr
    np.add.at(en, sp[ok], st[m[ok]]); np.add.at(en, ls[ok], st[m[ok]])
    nsp = nat & half                                           # native speaks English -> resident hears
    ok2 = (hear[ls, m] == m) & nsp; np.add.at(en, ls[ok2], st[m[ok2]])
    nrs = nat & ~half                                          # resident speaks -> native understands iff English
    ok3 = (emit[sp, m] == m) & nrs; np.add.at(en, sp[ok3], st[m[ok3]])
    return en


def breed(A, en, rng, mut=0.22):
    Ng = len(en); order = np.argsort(en); worst = order[:int(0.25 * Ng)]; top = order[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        msk = rng.random(A[pa].shape) < 0.5
        A[w] = np.where(msk, A[pa], A[pb]) + rng.normal(0, mut * 0.6, A[pa].shape)


def comprehension(A, rng, freq, st, K, T=8000):
    """Returns (freq-weighted, UNIFORM-over-words). Uniform is the right lens for vocab scaling:
    it tests the rare tail equally, where a scaling cliff would actually show."""
    emit = A.argmax(2); hear = A.argmax(1); Ng = len(A)
    p = 0.75 * freq + 0.25 * st / st.sum()
    sp = rng.integers(Ng, size=T); ls = rng.integers(Ng, size=T)
    mf = rng.choice(K, size=T, p=p); fw = float(np.mean(hear[ls, emit[sp, mf]] == mf))
    mu = rng.integers(K, size=T); uni = float(np.mean(hear[ls, emit[sp, mu]] == mu))
    return fw, uni


def run(cond, K, seed):
    freq, st, rank = world(K); rng = np.random.default_rng(seed); B = N * 12
    A = rng.normal(0, 0.3, (N, K, K))
    if cond == "cold":
        for t in range(BUDGET): breed(A, play(A, rng, np.arange(K), freq, st, B), rng)
    else:
        per = BUDGET // STAGES
        for k in range(1, STAGES + 1):
            active = rank[:max(2, round(k / STAGES * K))]
            for t in range(per): breed(A, play(A, rng, active, freq, st, B), rng)
    return comprehension(A, np.random.default_rng(9000 + seed), freq, st, K)   # (freq-weighted, uniform)


if __name__ == "__main__":
    out(f"SCALING CURVE: shared lexicon, fixed budget {BUDGET} gens, pop {N}, {SEEDS} seeds. "
        f"cold vs frequent-first curriculum ({STAGES} stages)")
    out("UNIFORM = every word tested equally (the right lens for vocab scaling; reveals tail collapse)")
    out(f"{'K':>6} | {'chance':>6} | {'cold freq/UNIF':>20} | {'curric freq/UNIF':>20} | {'UNIF adv':>9}")
    out("=" * 80)
    res = {}
    for K in KS:
        cold = [run("cold", K, s) for s in range(SEEDS)]
        grow = [run("freq_first", K, s) for s in range(SEEDS)]
        cf, cu = np.mean([c[0] for c in cold]), np.mean([c[1] for c in cold])
        gf, gu = np.mean([g[0] for g in grow]), np.mean([g[1] for g in grow])
        res[K] = dict(cf=cf, cu=cu, gf=gf, gu=gu)
        out(f"{K:>6} | {1/K:>6.3f} | {cf:>9.3f}/{cu:<9.3f} | {gf:>9.3f}/{gu:<9.3f} | {gu-cu:>+9.3f}")
    out("=" * 80)
    out("READING: if UNIFORM cold collapses toward chance as K grows while curriculum UNIFORM holds,")
    out("the curriculum protects the rare tail -> the scaling advantage is real on the right metric.")
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(KS, [res[K]["cu"] for K in KS], "s--", color="#c44", lw=2.5, label="cold (uniform)")
    ax1.plot(KS, [res[K]["gu"] for K in KS], "o-", color="#1b5e9e", lw=2.5, label="curriculum (uniform)")
    ax1.plot(KS, [res[K]["cf"] for K in KS], "s:", color="#e8a", lw=1.5, alpha=.7, label="cold (freq-wtd)")
    ax1.plot(KS, [1 / K for K in KS], ":", color="#999", label="chance")
    ax1.set_xlabel("vocabulary size K"); ax1.set_ylabel("two-way comprehension")
    ax1.set_title("Scaling on the UNIFORM metric (whole lexicon)", weight="bold"); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(KS, [res[K]["gu"] - res[K]["cu"] for K in KS], "^-", color="#2a8", lw=2.5, label="uniform")
    ax2.plot(KS, [res[K]["gf"] - res[K]["cf"] for K in KS], "^:", color="#999", lw=1.5, label="freq-wtd")
    ax2.axhline(0, color="#999", ls=":"); ax2.set_xlabel("vocabulary size K"); ax2.legend()
    ax2.set_ylabel("curriculum advantage (grow − cold)")
    ax2.set_title("Does the advantage WIDEN with scale?", weight="bold"); ax2.grid(alpha=.3)
    plt.tight_layout(); p = os.path.join(HERE, "scaling_curve.png"); plt.savefig(p, dpi=120); print("saved", p)
