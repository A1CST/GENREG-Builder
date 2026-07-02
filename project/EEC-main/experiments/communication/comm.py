"""COMMUNICATION axis (multi-agent): the first law that requires another entity.

A hidden latent c_t (sticky symbol the LISTENER cannot see). A SPEAKER sees c_t and
emits a signal sig_t over a channel; the LISTENER sees only sig_t and must recover
c_t. SHARED FATE: the pair survives by the listener's accuracy, so the speaker only
lives if its signals are informative. No protocol is wired in; selection is pure
joint survival. A communication PROTOCOL (c -> sig -> c) must EMERGE.

Genome = a co-adapted (Speaker, Listener) pair, mutated together, selected by joint
lifespan. We read EMERGENT communication, not accuracy-as-goal: mutual information
MI(c; sig) (did a code form?) and listener accuracy vs a SEVERED-channel control
(is the channel load-bearing?). Plus a channel-capacity sweep over signal alphabet
size V_sig (1 = mute -> chance; >= V_c -> full).

Reachable (not search-walled): the task is a memoryless relay (align two lookup
maps), which mutation can find -- unlike conditional computation.
"""
# --- EEC path bootstrap ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
from multiprocessing import Pool
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "120"))
SEG = int(os.environ.get("EEC_SEGLR", "400"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "4"))))
VC = 4                       # hidden-meaning alphabet
P_SWITCH = 0.05
START = 120.0
WORLD = None


def world_latent(rng, n=120000):
    s = np.empty(n, np.int32); sw = rng.random(n) < P_SWITCH; s[0] = rng.integers(0, VC)
    for t in range(1, n):
        s[t] = rng.integers(0, VC) if sw[t] else s[t-1]
    return s


class Pair:
    """Speaker map (VC->Vsig logits) + Listener map (Vsig->VC logits)."""
    def __init__(self, vsig, rng):
        self.vsig = vsig
        self.Ws = rng.normal(0, 1.0, (VC, vsig)).astype(np.float32)
        self.Wl = rng.normal(0, 1.0, (vsig, VC)).astype(np.float32)

    def copy(self):
        g = Pair.__new__(Pair); g.vsig = self.vsig
        g.Ws = self.Ws.copy(); g.Wl = self.Wl.copy(); return g

    def mutate(self, rng):
        for p in (self.Ws, self.Wl):
            mask = rng.random(p.shape) < MUT_RATE
            p += mask*rng.normal(0, 1, p.shape).astype(np.float32)*(MUT_SCALE*(np.abs(p)+0.2))

    def signal(self, c):     # c: (T,) -> sig: (T,)
        return self.Ws[c].argmax(1)

    def decode(self, sig):
        return self.Wl[sig].argmax(1)


def life_of(pair, c, sever_rng=None):
    sig = pair.signal(c)
    if sever_rng is not None:
        sig = sever_rng.integers(0, pair.vsig, len(sig))   # channel cut: random signal
    chat = pair.decode(sig)
    hits = chat == c
    cum = np.cumsum((~hits).astype(np.float64))
    return len(c) if cum[-1] < START else int(np.searchsorted(cum, START)) + 1, hits, sig


def mutual_info(c, sig, vsig):
    J = np.zeros((VC, vsig))
    np.add.at(J, (c, sig), 1.0); J /= J.sum()
    pc = J.sum(1, keepdims=True); ps = J.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = J * (np.log2(J) - np.log2(pc) - np.log2(ps))
    return float(np.nansum(m))


def evolve(vsig, seed):
    rng = np.random.default_rng(seed)
    pop = [Pair(vsig, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        st = int(rng.integers(0, len(WORLD)-SEG-1)); c = WORLD[st:st+SEG]
        lives = np.array([life_of(m, c)[0] for m in pop])
        pop = reproduce(pop, lives, rng)
    return pop, rng


def cell(task):
    vsig, seed = task
    pop, rng = evolve(vsig, seed)
    st = int(rng.integers(0, len(WORLD)-SEG-1)); c = WORLD[st:st+SEG]
    best = max(pop, key=lambda m: life_of(m, c)[0])
    _, hits, sig = life_of(best, c)
    _, hits_sev, _ = life_of(best, c, sever_rng=np.random.default_rng(seed+5))
    return (vsig, dict(acc=float(hits.mean()), acc_sev=float(hits_sev.mean()),
                       mi=mutual_info(c, sig, vsig), used=int(len(set(sig.tolist())))))


def main():
    global WORLD
    WORLD = world_latent(np.random.default_rng(999))
    chance = 1.0/VC
    vsigs = [1, 2, 3, 4, 8]
    tasks = [(v, s) for v in vsigs for s in SEEDS]
    print(f"communication: VC={VC} (chance acc {chance:.3f}), channel sweep Vsig={vsigs}, "
          f"{len(SEEDS)} seeds (gens={GENS})", flush=True)
    with Pool(min(18, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for v, d in res:
        agg.setdefault(v, []).append(d)
    print("\n  Vsig  listener_acc   severed_acc   MI(c;sig) bits   signals_used")
    for v in vsigs:
        r = agg[v]
        acc = np.mean([d["acc"] for d in r]); sev = np.mean([d["acc_sev"] for d in r])
        mi = np.mean([d["mi"] for d in r]); us = np.mean([d["used"] for d in r])
        print(f"  {v:>3}   {acc:.3f}        {sev:.3f}        {mi:.3f}          {us:.1f}")
    full = agg[4]
    acc4 = np.mean([d["acc"] for d in full]); sev4 = np.mean([d["acc_sev"] for d in full])
    mi4 = np.mean([d["mi"] for d in full])
    print("\n===== DID COMMUNICATION EMERGE? =====")
    print(f"  Vsig=VC=4: listener acc {acc4:.3f} vs chance {chance:.3f} vs severed {sev4:.3f}")
    print(f"  MI(c;sig)={mi4:.3f} bits (max {np.log2(VC):.2f}).  "
          f"{'PROTOCOL EMERGED (channel load-bearing)' if acc4 > chance+0.15 and acc4 > sev4+0.15 else 'no protocol'}")
    print("  channel capacity: Vsig=1 (mute) -> chance; more signals -> more meanings distinguishable.")


if __name__ == "__main__":
    main()
