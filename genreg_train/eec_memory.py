"""EEC memory world — grow recurrent memory the RIGHT way: not by training a cell
on log-prob, but by imposing an environment where a memoryless organism cannot
survive, and letting memory EMERGE. Mirrors project/EEC-main/engine/mind.py.

Paradigm (project/EEC-main/docs/PARADIGM.md): do NOT grade by accuracy/loss/vs-
baseline. The organism eats by anticipating its world (prediction = metabolism);
a miss burns energy; energy at 0 = death. Selection is pure SURVIVAL (lifespan).
Memory size M costs rent every step, so memory is vestigial UNLESS the world
makes it pay (P1: reachability — the world must pay).

The world here is our induced-class stream. The memory-forcing law is OCCLUSION:
with prob rho the input is blacked out (zeroed), so to keep anticipating through
the blackout the organism MUST carry the recent stream in its recurrent state.
A feedforward (W_rec≈0) organism goes blind every blackout, misses, and starves.

We read the STATE, not the token output: evolved M, recurrent-weight norm, and
the decisive load-bearing test — ablate W_rec and watch survival collapse (only
if memory was doing real work). P1 test: rho=0 (control) vs rho>0.
"""
import numpy as np

from genreg_train.wordpipe import induce_word_classes

EMBED = 16
MAX_M = 24
MIN_M = 2
MUT_RATE = 0.08
MUT_SCALE = 0.5


class Mind:
    """Recurrent organism with evolvable memory size M (mirrors mind.py)."""

    def __init__(self, V, rng):
        self.V = V
        self.E = rng.normal(0, 0.1, (V, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1 / np.sqrt(EMBED), (EMBED, MAX_M)).astype(np.float32)
        self.W_rec = rng.normal(0, 1 / np.sqrt(MAX_M), (MAX_M, MAX_M)).astype(np.float32)
        self.b = np.zeros(MAX_M, np.float32)
        self.W_out = rng.normal(0, 1 / np.sqrt(MAX_M), (MAX_M, V)).astype(np.float32)
        self.b_out = np.zeros(V, np.float32)
        self.M = int(rng.integers(MIN_M, MAX_M + 1))

    def run_states(self, seg, mask, decay=1.0, kill_rec=False):
        """Recurrence over the class segment. mask[t]=True -> input blacked out
        (occlusion). kill_rec -> ablate the recurrent connection. Returns S (T,M)."""
        M = self.M
        drive = self.E[seg] @ self.W_in[:, :M]        # (T,M) input drive
        drive[mask] = 0.0                             # OCCLUSION: sensory blackout
        Wrec = np.zeros((M, M), np.float32) if kill_rec else self.W_rec[:M, :M]
        b = self.b[:M]
        T = len(seg)
        S = np.empty((T, M), np.float32)
        s = np.zeros(M, np.float32)
        for t in range(T):
            s = np.tanh(drive[t] + (decay * s) @ Wrec + b)
            S[t] = s
        return S

    def params(self):
        return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out]

    def copy(self):
        g = Mind.__new__(Mind)
        g.V = self.V
        g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out = [p.copy() for p in self.params()]
        g.M = self.M
        return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            p += mask * rng.normal(0, 1, p.shape).astype(np.float32) * (MUT_SCALE * (np.abs(p) + 1e-3))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE * self.M)), MIN_M, MAX_M))


def live(mind, seg, mask, start_energy, rent, decay=1.0, kill_rec=False):
    """One life in the stream. prediction=metabolism: rent every step + 1 per
    MISS; death when energy exhausted. Returns lifespan (steps survived)."""
    S = mind.run_states(seg, mask, decay, kill_rec)
    logits = S[:-1] @ mind.W_out[:mind.M, :] + mind.b_out   # predict next class
    preds = logits.argmax(1)
    hits = preds == seg[1:]
    cost = rent * mind.M + (~hits)                          # energy burned per step
    cum = np.cumsum(cost)
    if cum[-1] < start_energy:
        return len(hits)
    return int(np.searchsorted(cum, start_energy)) + 1


def reproduce(pop, lives, rng, elite_frac=0.2, cull_frac=0.2):
    """Steady-state: carry top (1-cull), refill bottom with mutated elite clones."""
    order = np.argsort(lives)[::-1]
    n = len(pop)
    n_elite = max(1, round(elite_frac * n))
    carried = [pop[i] for i in order[:n - round(cull_frac * n)]]
    top = order[:n_elite]
    new = list(carried)
    while len(new) < n:
        child = pop[int(top[rng.integers(0, len(top))])].copy()
        child.mutate(rng)
        new.append(child)
    return new


def rec_norm(mind):
    """Recurrent-weight magnitude on the active memory (proxy for memory reliance)."""
    return float(np.linalg.norm(mind.W_rec[:mind.M, :mind.M]))


def measure_internals(mind, stream, V, decay, rng):
    """The EEC memory signals (mirrors run_matrix.py) — read the STATE:
      * gain    = spectral radius of W_rec (active maintenance; entropy drives it)
      * horizon = how many steps a one-token perturbation persists in the state
                  (memory span). Both measured WITHOUT occlusion, on the champion."""
    M = mind.M
    Wr = mind.W_rec[:M, :M]
    gain = float(np.max(np.abs(np.linalg.eigvals(Wr)))) if M else 0.0
    start = int(rng.integers(0, len(stream) - 1100))
    seg = stream[start:start + 1000]
    nomask = np.zeros(1000, bool)
    S = mind.run_states(seg, nomask, decay)
    hors = []
    for t0 in rng.integers(50, 900, 16):
        s2 = seg.copy(); s2[t0] = int((s2[t0] + 3) % V)
        S2 = mind.run_states(s2, nomask, decay)
        div = np.linalg.norm(S - S2, axis=1)
        peak = div[t0:t0 + 3].max()
        if peak < 1e-6:
            continue
        after = div[t0:]; below = np.where(after < 0.1 * peak)[0]
        hors.append(int(below[0]) if len(below) else len(after))
    horizon = float(np.mean(hors)) if hors else 0.0
    return {"M": M, "gain": round(gain, 3), "eff_gain": round(decay * gain, 3),
            "horizon": round(horizon, 2)}


# --------------------------------------------------------------------------
# Worlds. class_stream = our induced classes ("text" — thin, memory rarely pays).
# longrange_stream = the doc's positive control: a periodic pattern where the
# next symbol is set by your PHASE in the cycle, so ONLY memory (phase tracking)
# predicts — memory provably pays. Same as EEC's long-range world.
# --------------------------------------------------------------------------
def class_stream(n_classes=32):
    _, cids, nc, _ = induce_word_classes(n_classes)
    return np.asarray(cids, np.int64), nc


def longrange_stream(period=16, alphabet=6, noise=0.05, length=60000, seed=1):
    rng = np.random.default_rng(seed)
    pattern = rng.integers(0, alphabet, size=period)
    reps = length // period + 1
    stream = np.tile(pattern, reps)[:length].copy()
    flip = rng.random(length) < noise
    stream[flip] = rng.integers(0, alphabet, size=int(flip.sum()))
    return stream.astype(np.int64), alphabet


def markov_stream(order=2, alphabet=6, noise=0.03, length=60000, seed=1):
    """A world where the next symbol is a DETERMINISTIC function of the last
    `order` symbols. The current symbol alone is AMBIGUOUS about the next, so
    the organism MUST remember `order-1` symbols to predict — memory pays on
    EVERY step (not just during occlusion), and partial memory pays partially
    (reachable by mutation). This is the honest 'memory-required' world."""
    rng = np.random.default_rng(seed)
    table = rng.integers(0, alphabet, size=(alphabet,) * order)   # the rule
    s = list(rng.integers(0, alphabet, order))
    for _ in range(length - order):
        s.append(int(table[tuple(s[-order:])]))
    stream = np.array(s, np.int64)
    flip = rng.random(len(stream)) < noise
    stream[flip] = rng.integers(0, alphabet, size=int(flip.sum()))
    return stream, alphabet


def _occ_mask(n, rho, burst, rng):
    """Occlusion mask. burst=1 -> per-step Bernoulli; burst>1 -> contiguous
    blackout RUNS of ~burst steps (forces a memory HORIZON, not just 1-step)."""
    if burst <= 1:
        return rng.random(n) < rho
    mask = np.zeros(n, bool)
    t = 0
    while t < n:
        if rng.random() < rho:
            L = int(rng.integers(1, burst * 2))
            mask[t:t + L] = True; t += L
        else:
            t += 1
    return mask


def evolve_world(stream, V, rho=0.3, burst=1, gens=200, pop=150, seg=1000,
                 start_energy=120, rent=0.01, decay=1.0, tag="", seed=0, log=print):
    """Evolve under occlusion. Selection = LIFESPAN (never accuracy). Returns the
    champion + emergent-state trajectory (M, rec_norm)."""
    rng = np.random.default_rng(seed)
    population = [Mind(V, rng) for _ in range(pop)]
    hist, champ = [], None
    for gen in range(1, gens + 1):
        start = int(rng.integers(0, len(stream) - seg - 1))
        seg_ids = stream[start:start + seg]
        mask = _occ_mask(seg, rho, burst, rng)
        lives = np.array([live(m, seg_ids, mask, start_energy, rent, decay) for m in population])
        bi = int(np.argmax(lives)); champ = population[bi]
        if gen % 25 == 0 or gen == 1:
            hist.append((gen, int(lives[bi]), champ.M, round(rec_norm(champ), 3)))
            log(f"  [{tag} rho={rho} burst={burst}] gen {gen:>4} | life {lives[bi]:>5}/{seg} "
                f"| M {champ.M:>3} | rec_norm {rec_norm(champ):.2f} | avg {lives.mean():.1f}")
        population = reproduce(population, lives, rng)
    return {"stream": stream, "V": V, "rho": rho, "burst": burst, "champ": champ,
            "hist": hist, "seg": seg, "start_energy": start_energy, "rent": rent, "decay": decay}


def ablation_test(res, n_trials=40, seed=999, log=print):
    """Load-bearing proof: champion lifespan WITH vs WITHOUT the recurrent
    connection, same occlusion. Big drop => memory was doing real work."""
    m, seg, stream = res["champ"], res["seg"], res["stream"]
    rng = np.random.default_rng(seed)
    intact, ablated = [], []
    for _ in range(n_trials):
        start = int(rng.integers(0, len(stream) - seg - 1))
        s = stream[start:start + seg]
        mask = _occ_mask(seg, res["rho"], res["burst"], rng)
        intact.append(live(m, s, mask, res["start_energy"], res["rent"], res["decay"], kill_rec=False))
        ablated.append(live(m, s, mask, res["start_energy"], res["rent"], res["decay"], kill_rec=True))
    im, am = float(np.mean(intact)), float(np.mean(ablated))
    drop = round((im - am) / im, 3) if im > 0 else 0.0
    log(f"  ablation: intact life {im:.0f} | W_rec killed {am:.0f} | drop {drop:.1%}  "
        f"<- memory load-bearing if large")
    return {"intact": round(im, 1), "ablated": round(am, 1), "survival_drop": drop, "M": m.M}
