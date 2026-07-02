"""SCALING DIAGNOSTIC -- is fluency-at-scale a LOOKUP-TABLE wall, fixable by STRUCTURE?

The scaling_curve showed comprehension sags as vocab K grows. Hypothesis: the shared lexicon is a
K x K LOOKUP TABLE -- quadratic parameters, every word learned independently, collisions grow with K.
Real language scales because words share STRUCTURE in a fixed-dimensional space. So compare two shared
lexicons (both: speak & hear tied; natives anchor to identity-English):

  lookup     : genome A (K x K). emit=row argmax, hear=col argmax. Params grow as K^2.
  structured : genome W (D x D) acting in the grounded embedding space E (K x D). One map for both
               speak and hear: map[m] = argmax_a (E[a] . W . E[m]). Params = D^2, CONSTANT in K.
               W -> identity reproduces English (E E^T diagonal). Words share structure via E.

If `structured` holds uniform comprehension ~flat as K grows while `lookup` sags, the scaling fix for
fluent English at large vocab is a STRUCTURED grounded representation, not a bigger table.
Uniform comprehension (every word tested equally) is the metric.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(HERE, "grounding_large.npz"), allow_pickle=True)
EMB_ALL = G["emb"].astype(np.float64); VOCAB_ALL = list(G["vocab"]); DMAX = EMB_ALL.shape[1]
P_NATIVE, U = 0.7, 0.25
N = int(os.environ.get("SD_N", "44")); BUDGET = int(os.environ.get("SD_BUDGET", "1200"))
SEEDS = int(os.environ.get("SD_SEEDS", "2")); D = int(os.environ.get("SD_D", "64"))
KS = [int(x) for x in os.environ.get("SD_KS", "24,48,96,160").split(",")]

LOG = open(os.path.join(HERE, "scaling_diagnostic_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def wcfg(K):
    freq = 1.0 / (np.arange(1, K + 1) ** 1.1); freq /= freq.sum()
    st = np.clip((1.0 / freq) / (1.0 / freq).mean(), 0.2, 12.0); st = st / st.mean()
    E = EMB_ALL[:K, :D].copy(); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return freq, st, E


def play(emit, hear, rng, freq, st, K, B):
    Ng = len(emit); en = np.zeros(Ng)
    p = (1 - U) * freq + U * (st / st.sum())
    m = rng.choice(K, size=B, p=p); sp = rng.integers(Ng, size=B); ls = rng.integers(Ng, size=B)
    nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5
    rr = ~nat; s = emit[sp, m]; ok = (hear[ls, s] == m) & rr
    np.add.at(en, sp[ok], st[m[ok]]); np.add.at(en, ls[ok], st[m[ok]])
    ok2 = (hear[ls, m] == m) & nat & half; np.add.at(en, ls[ok2], st[m[ok2]])
    ok3 = (emit[sp, m] == m) & nat & ~half; np.add.at(en, sp[ok3], st[m[ok3]])
    return en


def breed(P, en, rng, mut=0.22):
    Ng = len(en); order = np.argsort(en); worst = order[:int(0.25 * Ng)]; top = order[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        msk = rng.random(P[pa].shape) < 0.5
        P[w] = np.where(msk, P[pa], P[pb]) + rng.normal(0, mut * 0.6, P[pa].shape)


def emit_hear(arch, P, E):
    if arch == "lookup":                                        # A (K x K): decomposable, K^2 params
        return P.argmax(2), P.argmax(1)                          # speak=row, hear=col
    if arch == "structured":                                    # W (D x D): shared map, entangled
        S = np.einsum('kd,nde,me->nkm', E, P, E, optimize=True)  # (N,K,K): E W E^T
        mp = S.argmax(1); return mp, mp
    # codes C (K x D): per-word grounded code -> decomposable AND linear-in-K
    Sw = np.einsum('wd,nmd->nwm', E, P, optimize=True)           # (N,K,K): E . C[m] -> emit=argmax word w
    emit = Sw.argmax(1)
    Sm = np.einsum('nmd,sd->nms', P, E, optimize=True)           # (N,K,K): C[m] . E[s] -> hear=argmax meaning m
    hear = Sm.argmax(1)
    return emit, hear


def run(arch, K, seed):
    freq, st, E = wcfg(K); rng = np.random.default_rng(seed); B = N * 12
    shape = {"lookup": (N, K, K), "structured": (N, D, D), "codes": (N, K, D)}[arch]
    P = rng.normal(0, 0.3, shape)
    for t in range(BUDGET):
        emit, hear = emit_hear(arch, P, E)
        breed(P, play(emit, hear, rng, freq, st, K, B), rng)
    emit, hear = emit_hear(arch, P, E)
    rng2 = np.random.default_rng(9000 + seed); T = 8000
    m = rng2.integers(K, size=T); sp = rng2.integers(N, size=T); ls = rng2.integers(N, size=T)
    return float(np.mean(hear[ls, emit[sp, m]] == m))            # UNIFORM comprehension


if __name__ == "__main__":
    out(f"SCALING DIAGNOSTIC: lookup (K^2, decomposable) vs structured (D^2, entangled) vs "
        f"codes (K x D={D}, decomposable+linear). pop {N}, budget {BUDGET}, {SEEDS} seeds. UNIFORM comp.")
    out(f"{'K':>6} | {'chance':>7} | {'lookup':>15} | {'structured':>15} | {'codes (KxD)':>15} | {'winner':>10}")
    out("=" * 86)
    ARCH = ["lookup", "structured", "codes"]; res = {}
    for K in KS:
        r = {a: [run(a, K, s) for s in range(SEEDS)] for a in ARCH}
        res[K] = {a: np.mean(r[a]) for a in ARCH}
        win = max(ARCH, key=lambda a: res[K][a])
        out(f"{K:>6} | {1/K:>7.3f} | {ms(r['lookup']):>15} | {ms(r['structured']):>15} | "
            f"{ms(r['codes']):>15} | {win:>10}")
    out("=" * 86)
    out("READING: which architecture holds uniform comprehension best as K grows = the scaling rep.")
    for a in ARCH:
        out(f"  {a:>11} K{KS[0]}->K{KS[-1]}: {res[KS[0]][a]:.3f} -> {res[KS[-1]][a]:.3f}")
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5)); col = {"lookup": "#c44", "structured": "#999", "codes": "#1b5e9e"}
    for a in ARCH:
        plt.plot(KS, [res[K][a] for K in KS], "o-", color=col[a], lw=2.5, label=a)
    plt.plot(KS, [1 / K for K in KS], ":", color="#bbb", label="chance")
    plt.xlabel("vocabulary size K"); plt.ylabel("uniform two-way comprehension")
    plt.title("Scaling representation: decomposable vs entangled vs grounded codes", weight="bold")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    p = os.path.join(HERE, "scaling_diagnostic.png"); plt.savefig(p, dpi=120); print("saved", p)
