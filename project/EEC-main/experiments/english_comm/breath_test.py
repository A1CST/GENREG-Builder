"""BREATH TEST -- does a finite consumable resource ('breath') make calibrated utterance length
emerge, dodging the search-wall that a learned STOP hits?

Sequential comm: a meaning has C active slots (C ~ 1..Smax, varies per round); the speaker emits one
word per position t (a value via a shared per-position association A), the listener decodes it. To
convey the meaning the utterance must reach all active slots. Terminators (same substrate otherwise):
  breath : utterance length L = the genome's evolvable BREATH budget (physics; no stop decision).
  stop   : the genome evolves a STOP rule (logistic on [is-current-slot-active, bias]); L = first stop.
           Unbounded otherwise (rambling past Smax wastes energy = death pressure).
  designed: L = C exactly (ORACLE upper bound; the hand-designed length we are NOT allowed to use).

Fitness = slots correctly conveyed (+full-meaning bonus) - energy per word. We measure: full-meaning
comprehension, mean utterance length vs C (does length CALIBRATE?), energy efficiency, evolved breath
depth. CONFIRM breath: length tracks C and breath >> stop (stop fails to calibrate). FALSIFY: stop
calibrates as well as breath (then a learned stop suffices, breath not needed in this regime).
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SMAX, V = 6, 6
P_NATIVE, FULL_BONUS, EWORD = 0.7, 2.0, 0.15
N = int(os.environ.get("BR_N", "64")); BUDGET = int(os.environ.get("BR_BUDGET", "2500"))
SEEDS = int(os.environ.get("BR_SEEDS", "4"))

LOG = open(os.path.join(HERE, "breath_test_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def breed(g, en, rng, keys, mut=0.22):
    Ng = len(en); o = np.argsort(en); worst = o[:int(0.25 * Ng)]; top = o[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        for k in keys:
            m = rng.random(g[k][pa].shape) < 0.5
            g[k][w] = np.where(m, g[k][pa], g[k][pb]) + rng.normal(0, mut * 0.6, g[k][pa].shape)


def lengths(cond, g, sp, C, B):
    """utterance length L per interaction for the chosen terminator."""
    if cond == "designed":
        return C.copy()
    if cond == "breath":
        return np.clip(np.round(g["breath"][sp]).astype(int), 1, SMAX)
    # stop: scan positions; stop when sigmoid(w0*active_t + w1) > 0.5. active_t = (t < C).
    w = g["stop"]; L = np.full(B, SMAX, int); stopped = np.zeros(B, bool)
    for t in range(SMAX):
        act = (t < C).astype(float)
        z = w[sp, 0] * act + w[sp, 1]
        fire = (z > 0) & (~stopped) & (t >= 1)        # can stop after >=1 word
        L[fire] = t; stopped |= fire
    return L


def play(cond, g, rng, train=True):
    A = g["A"]; Ng = len(A); emit = A.argmax(3); hear = A.argmax(2)
    B = N * 16
    C = rng.integers(1, SMAX + 1, size=B)              # meaning complexity
    vals = rng.integers(V, size=(B, SMAX))             # value per slot
    sp = rng.integers(Ng, size=B); ls = rng.integers(Ng, size=B)
    nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5
    L = lengths(cond, g, sp, C, B)
    en = np.zeros(Ng); conv = np.zeros(B, int)
    for t in range(SMAX):
        spoke = (t < L) & (t < C)                      # position t emitted AND it's an active slot
        if not spoke.any(): continue
        idx = np.where(spoke)[0]; vt = vals[idx, t]
        cor = hear[ls[idx], t, emit[sp[idx], t, vt]] == vt
        conv[idx] += cor
        if train:
            ok = cor & (~nat[idx]); np.add.at(en, sp[idx][ok], 1.0); np.add.at(en, ls[idx][ok], 1.0)
            n0 = (hear[ls[idx], t, vt] == vt) & nat[idx] & half[idx]; np.add.at(en, ls[idx][n0], 1.0)
            n1 = (emit[sp[idx], t, vt] == vt) & nat[idx] & ~half[idx]; np.add.at(en, sp[idx][n1], 1.0)
    full = conv == C                                   # whole meaning conveyed
    if train:
        fb = full & (~nat); np.add.at(en, sp[fb], FULL_BONUS); np.add.at(en, ls[fb], FULL_BONUS)
        np.add.at(en, sp, -EWORD * L)                  # energy: every word costs breath/metabolism
        return en
    return float(full.mean()), float(L.mean()), float(np.corrcoef(C, L)[0, 1]), float(L.mean() / C.mean())


def run(cond, seed):
    rng = np.random.default_rng(seed)
    g = {"A": rng.normal(0, 0.3, (N, SMAX, V, V)),
         "breath": rng.uniform(1, SMAX, N), "stop": rng.normal(0, 0.5, (N, 2))}
    keys = ["A"] + ({"breath": ["breath"], "stop": ["stop"], "designed": []}[cond])
    for t in range(BUDGET):
        breed(g, play(cond, g, rng), rng, keys)
    evals = [play(cond, g, np.random.default_rng(9000 + seed + 17 * k), train=False) for k in range(3)]
    return np.mean(evals, axis=0)


if __name__ == "__main__":
    out(f"BREATH TEST: emergent utterance length. Smax={SMAX}, V={V}, pop {N}, budget {BUDGET}, "
        f"{SEEDS} seeds. meaning complexity C~1..{SMAX}; energy/word={EWORD}.")
    out(f"{'terminator':>10} | {'full-meaning':>13} | {'mean length':>12} | {'corr(C,len)':>12} | {'len/C ratio':>11}")
    out("=" * 72)
    res = {}
    for cond in ["designed", "breath", "stop"]:
        r = np.array([run(cond, s) for s in range(SEEDS)])
        res[cond] = r.mean(0)
        out(f"{cond:>10} | {ms(r[:,0]):>13} | {ms(r[:,1]):>12} | {ms(r[:,2]):>12} | {ms(r[:,3]):>11}")
    out("=" * 72)
    out("READING: corr(C,len)>0 => length CALIBRATES to meaning complexity. len/C~1 => efficient (no waste).")
    out("  designed = oracle upper bound (length=C). breath = physics. stop = learned conditional halt.")
    out(f"  CONFIRM breath: breath calibrates (corr>0, len/C~1) & full-meaning ~ designed, stop lags (search-wall).")
    out(f"  FALSIFY: stop calibrates as well as breath => a learned stop suffices, breath not needed here.")
    out("done"); LOG.close()
