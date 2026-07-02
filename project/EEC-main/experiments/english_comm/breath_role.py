"""BREATH ROLE -- is breath the length CALIBRATOR, or does a learned STOP own that because the speaker
always knows its own content?

The speaker always knows the meaning it wants to convey, so "have I said it all yet?" is ALWAYS
perceivable to it. That predicts a learned stop can calibrate length in ANY regime, and breath (a fixed
budget) never can. We test contiguous AND scattered active slots (scattered = the meaning's parts are
spread out, so calibrated length = reach the last active slot). If stop calibrates in both and breath in
neither, breath is NOT the length mechanism -- its real job is elsewhere (chunking; the urgency coupling).
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SMAX, V = 8, 6; EWORD = 0.08
N = int(os.environ.get("BR_N", "64")); BUDGET = int(os.environ.get("BR_BUDGET", "2500")); SEEDS = 4
FULL_BONUS = 2.0
LOG = open(os.path.join(HERE, "breath_role_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def make_meaning(rng, B, regime):
    C = rng.integers(1, SMAX + 1, size=B); active = np.zeros((B, SMAX), bool)
    for i in range(B):
        if regime == "contiguous":
            active[i, :C[i]] = True
        else:
            active[i, rng.choice(SMAX, C[i], replace=False)] = True
    return C, active


def breed(g, en, rng, keys, mut=0.22):
    Ng = len(en); o = np.argsort(en); worst = o[:int(0.25 * Ng)]; top = o[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        for k in keys:
            m = rng.random(g[k][pa].shape) < 0.5
            g[k][w] = np.where(m, g[k][pa], g[k][pb]) + rng.normal(0, mut * 0.6, g[k][pa].shape)


def length(cond, g, sp, active, B):
    if cond == "breath":
        return np.clip(np.round(g["breath"][sp]).astype(int), 1, SMAX)
    # stop: at each position, feature = 'any active slot at position >= t' (speaker knows its meaning)
    w = g["stop"]; L = np.full(B, SMAX, int); stopped = np.zeros(B, bool)
    future_active = np.zeros((B, SMAX), bool)
    for t in range(SMAX - 1, -1, -1):
        future_active[:, t] = active[:, t] | (future_active[:, t + 1] if t + 1 < SMAX else False)
    for t in range(SMAX):
        z = w[sp, 0] * future_active[:, t].astype(float) + w[sp, 1]
        fire = (z < 0) & (~stopped) & (t >= 1); L[fire] = t; stopped |= fire   # stop when no active ahead
    return L


def play(cond, regime, g, rng, train=True):
    A = g["A"]; Ng = len(A); emit = A.argmax(3); hear = A.argmax(2); B = N * 16
    C, active = make_meaning(rng, B, regime); vals = rng.integers(V, size=(B, SMAX))
    sp = rng.integers(Ng, size=B); ls = rng.integers(Ng, size=B); L = length(cond, g, sp, active, B)
    conv = np.zeros(B, int)
    for t in range(SMAX):
        spoke = (t < L) & active[:, t]
        if not spoke.any(): continue
        idx = np.where(spoke)[0]; vt = vals[idx, t]
        conv[idx] += hear[ls[idx], t, emit[sp[idx], t, vt]] == vt
    full = conv == C
    if not train:
        # calibration target = index of last active slot + 1 (span the meaning needs)
        span = np.array([(np.where(active[i])[0].max() + 1) if active[i].any() else 1 for i in range(B)])
        return float(full.mean()), float(L.mean()), float(np.corrcoef(span, L)[0, 1]), float((L - span).mean())
    base = conv + FULL_BONUS * full
    en = np.zeros(Ng); np.add.at(en, sp, base - EWORD * L); np.add.at(en, ls, base)
    return en


def run(cond, regime, seed):
    rng = np.random.default_rng(seed)
    g = {"A": rng.normal(0, 0.3, (N, SMAX, V, V)), "breath": rng.uniform(1, SMAX, N), "stop": rng.normal(0, 0.5, (N, 2))}
    keys = ["A"] + ({"breath": ["breath"], "stop": ["stop"]}[cond])
    for t in range(BUDGET):
        breed(g, play(cond, regime, g, rng), rng, keys)
    ev = [play(cond, regime, g, np.random.default_rng(9000 + seed + 11 * k), train=False) for k in range(3)]
    return np.mean(ev, axis=0)


if __name__ == "__main__":
    out(f"BREATH ROLE: does STOP calibrate length (speaker knows its content) where BREATH cannot? "
        f"Smax={SMAX}. pop {N}, budget {BUDGET}, {SEEDS} seeds.")
    out(f"{'regime':>11} {'terminator':>10} | {'full-meaning':>13} | {'mean len':>9} | {'corr(span,len)':>15} | {'len-span':>9}")
    out("=" * 80)
    for regime in ["contiguous", "scattered"]:
        for cond in ["stop", "breath"]:
            r = np.array([run(cond, regime, s) for s in range(SEEDS)])
            out(f"{regime:>11} {cond:>10} | {ms(r[:,0]):>13} | {ms(r[:,1]):>9} | {ms(r[:,2]):>15} | {ms(r[:,3]):>9}")
    out("=" * 80)
    out("READING: stop corr(span,len)~1 in BOTH regimes => the learned stop calibrates length because the")
    out("speaker knows its own content. breath corr~0 => breath is NOT the calibrator (its role is chunking).")
    out("done"); LOG.close()
