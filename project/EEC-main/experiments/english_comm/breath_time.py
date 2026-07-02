"""BREATH x TIME -- breath weighted by production SPEED, coupled to the time constraint.

Breath is a finite reservoir B (a lungful). A word's breath cost = its production SPEED sigma (a fast
'fat' word burns more air); so words-per-breath = floor(B/sigma). Speed also sets time: utterance time
= L/sigma (fast = less time). So ONE evolvable knob sigma trades breath against time:
  fast (high sigma): little time, but few words per breath -> breath truncates -> SHORT messages.
  slow (low sigma):  many words per breath, but lots of time -> penalised by time pressure.

Substrate: sequential comm, meaning has C active slots (C~1..Smax), shared per-position association A,
a learned STOP (which we showed calibrates length). Utterance length L = min(stop, breath_cap=B/sigma).
Coupled speaker+listener survival (no native anchor; the code co-evolves). Fitness = meaning conveyed,
DISCOUNTED by time 1/(1+time/tau), minus energy per word. sigma, stop, A all evolve.

PREDICTION (the coupling): as time pressure rises (tau down), evolved sigma RISES, breath_cap FALLS,
breath becomes the binding limit (truncation up), and conveyable complexity / message length FALL.
Urgency -> fast speech -> breath-limited -> short messages, emergent from the two laws. FALSIFY: sigma
ignores time pressure, or breath never binds (then the breath x time coupling does nothing).
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SMAX, V, B_LUNG = 6, 6, 6.0
EWORD = 0.05; SIG_LO, SIG_HI = 0.5, 3.0
N = int(os.environ.get("BT_N", "64")); BUDGET = int(os.environ.get("BT_BUDGET", "2500"))
SEEDS = int(os.environ.get("BT_SEEDS", "4")); FULL_BONUS = 2.0
TAUS = [float(x) for x in os.environ.get("BT_TAUS", "1.5,6,24").split(",")]   # small tau = strong time pressure

LOG = open(os.path.join(HERE, "breath_time_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.2f}+/-{a.std():.2f}"


def breed(g, en, rng, mut=0.22):
    Ng = len(en); o = np.argsort(en); worst = o[:int(0.25 * Ng)]; top = o[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        for k in ("A", "sigma", "stop"):
            m = rng.random(g[k][pa].shape) < 0.5
            g[k][w] = np.where(m, g[k][pa], g[k][pb]) + rng.normal(0, mut * 0.6, g[k][pa].shape)
    g["sigma"][:] = np.clip(g["sigma"], SIG_LO, SIG_HI)


def stop_len(g, sp, C, B):
    w = g["stop"]; L = np.full(B, SMAX, int); stopped = np.zeros(B, bool)
    for t in range(SMAX):
        z = w[sp, 0] * (t < C).astype(float) + w[sp, 1]
        fire = (z > 0) & (~stopped) & (t >= 1); L[fire] = t; stopped |= fire
    return L


def play(g, rng, tau, train=True):
    A = g["A"]; Ng = len(A); emit = A.argmax(3); hear = A.argmax(2); B = N * 16
    C = rng.integers(1, SMAX + 1, size=B); vals = rng.integers(V, size=(B, SMAX))
    sp = rng.integers(Ng, size=B); ls = rng.integers(Ng, size=B)
    sig = g["sigma"][sp]; cap = np.clip(np.floor(B_LUNG / sig).astype(int), 1, SMAX)
    L = np.minimum(stop_len(g, sp, C, B), cap)
    trunc = (cap < stop_len(g, sp, C, B)) & (cap < C)      # breath cut the message before the meaning ended
    conv = np.zeros(B, int)
    for t in range(SMAX):
        spoke = (t < L) & (t < C)
        if not spoke.any(): continue
        idx = np.where(spoke)[0]; vt = vals[idx, t]
        conv[idx] += hear[ls[idx], t, emit[sp[idx], t, vt]] == vt
    full = conv == C; time = L / sig
    if not train:
        return float(full.mean()), float(L.mean()), float(g["sigma"].mean()), float(cap.mean()), float(trunc.mean())
    base = conv + FULL_BONUS * full
    sp_r = base / (1.0 + time / tau) - EWORD * L           # speaker bears time + energy
    en = np.zeros(Ng); np.add.at(en, sp, sp_r); np.add.at(en, ls, base)
    return en


def run(tau, seed):
    rng = np.random.default_rng(seed)
    g = {"A": rng.normal(0, 0.3, (N, SMAX, V, V)), "sigma": rng.uniform(SIG_LO, SIG_HI, N),
         "stop": rng.normal(0, 0.5, (N, 2))}
    for t in range(BUDGET):
        breed(g, play(g, rng, tau), rng)
    ev = [play(g, np.random.default_rng(9000 + seed + 13 * k), tau, train=False) for k in range(3)]
    return np.mean(ev, axis=0)


if __name__ == "__main__":
    out(f"BREATH x TIME: lung B={B_LUNG}, sigma in [{SIG_LO},{SIG_HI}], Smax={SMAX}. pop {N}, budget {BUDGET}, "
        f"{SEEDS} seeds. small tau = STRONG time pressure.")
    out(f"{'tau':>6} {'pressure':>9} | {'full-meaning':>13} | {'mean len':>9} | {'evolved speed sigma':>20} | "
        f"{'breath cap':>11} | {'breath-truncated':>16}")
    out("=" * 100)
    for tau in TAUS:
        r = np.array([run(tau, s) for s in range(SEEDS)])
        lab = "STRONG" if tau <= 2 else ("medium" if tau <= 10 else "weak")
        out(f"{tau:>6} {lab:>9} | {ms(r[:,0]):>13} | {ms(r[:,1]):>9} | {ms(r[:,2]):>20} | {ms(r[:,3]):>11} | {ms(r[:,4]):>16}")
    out("=" * 100)
    out("READING (CONFIRM): as time pressure rises (tau down) -> sigma UP, breath cap DOWN,")
    out("breath-truncation UP, messages shorter. Urgency -> fast speech -> breath-limited -> short. ")
    out("FALSIFY: sigma flat across tau, or breath never truncates (coupling does nothing).")
    out("done"); LOG.close()
