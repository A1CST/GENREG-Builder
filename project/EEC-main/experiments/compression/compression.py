"""COMPRESSION axis: internal processing has finite bandwidth -> forces ABSTRACTION.

The organism computes a rich M-dim activation each step, but the state it CARRIES
to the next step must pass through a K-dim bottleneck (K <= M). Computation is
wide; memory is narrow. To survive in a structured world the organism must encode
the survival-relevant structure into K dims -- i.e. ABSTRACT.

  h_t = tanh( emb_t W_in + z_{t-1} W_dec + b )     # wide computation (M dims)
  z_t = tanh( h_t W_enc )                           # narrow persistent state (K dims)
  output from h_t

Sweep the bottleneck width K. The test: does competence HOLD as K shrinks (the
organism found a low-dim abstraction of the world) or COLLAPSE (couldn't compress)?
The smallest K that still predicts = the world's intrinsic dimension, discovered
by the organism. Long-range is a periodic phase -> we expect ~2 dims (a phase
manifold) to suffice; K=1 should break.

Read internals (competence on clean segments, effective rank of the K-code).
Selection is pure survival.
"""
# --- EEC path bootstrap ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
import run_matrix as RM
from evolve import EMBED, MUT_RATE, MUT_SCALE, POP_SIZE
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "140"))
SEG = int(os.environ.get("EEC_SEGLR", "400"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "3"))))
M = 32                       # wide computation width (fixed)
E0 = 200.0
KS = [1, 2, 3, 4, 8, 16, 32]  # bottleneck widths to sweep
WORLD_KIND = os.environ.get("EEC_WORLD", "longrange")
LR = None; VOC = 6


def world_multiphase(rng, n=200000, V=12, p1=5, p2=7, noise=0.03):
    """Two independent phase counters combined -> intrinsic dim ~2 (must track BOTH
    to predict). token = (t%p1 + t%p2) mod V."""
    t = np.arange(n)
    tok = ((t % p1) + (t % p2)) % V
    mloc = rng.random(n) < noise
    tok[mloc] = rng.integers(0, V, mloc.sum())
    return tok.astype(np.int32), V


class Org:
    def __init__(self, V, K, rng):
        self.K = K
        self.E = rng.normal(0, 0.1, (V, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1/np.sqrt(EMBED), (EMBED, M)).astype(np.float32)
        self.W_dec = rng.normal(0, 1/np.sqrt(K), (K, M)).astype(np.float32)   # z_{t-1} -> h
        self.W_enc = rng.normal(0, 1/np.sqrt(M), (M, K)).astype(np.float32)   # h -> z
        self.b = np.zeros(M, dtype=np.float32)
        self.W_out = rng.normal(0, 1/np.sqrt(M), (M, V)).astype(np.float32)
        self.b_out = np.zeros(V, dtype=np.float32)

    def params(self):
        return [self.E, self.W_in, self.W_dec, self.W_enc, self.b, self.W_out, self.b_out]

    def copy(self):
        g = Org.__new__(Org); g.K = self.K
        (g.E, g.W_in, g.W_dec, g.W_enc, g.b, g.W_out, g.b_out) = [p.copy() for p in self.params()]
        return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            p += mask * rng.normal(0, 1, p.shape).astype(np.float32) * (MUT_SCALE*(np.abs(p)+1e-3))

    def run(self, seg):
        emb = self.E[seg]; drive = emb @ self.W_in; T = len(seg)
        H = np.empty((T, M), np.float32)
        z = np.zeros(self.K, np.float32)
        for t in range(T):
            h = np.tanh(drive[t] + z @ self.W_dec + self.b)
            z = np.tanh(h @ self.W_enc)
            H[t] = h
        return H


def hits(m, seg):
    H = m.run(seg)
    return (H @ m.W_out + m.b_out).argmax(1)[:-1] == seg[1:]


def life(h):
    cum = np.cumsum(0.005*M + (~h))
    return len(h) if cum[-1] < E0 else int(np.searchsorted(cum, E0)) + 1


def code_rank(m, seg):
    """effective dim actually used by the K-code (participation ratio of z variance)."""
    emb = m.E[seg]; drive = emb @ m.W_in
    z = np.zeros(m.K, np.float32); Z = []
    for t in range(len(seg)):
        h = np.tanh(drive[t] + z @ m.W_dec + m.b); z = np.tanh(h @ m.W_enc); Z.append(z.copy())
    Z = np.array(Z); v = Z.var(0)
    return float(v.sum()**2 / (np.sum(v**2) + 1e-9))   # participation ratio


def cell(task):
    K, seed = task
    rng = np.random.default_rng(seed)
    pop = [Org(VOC, K, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
        lives = np.array([life(hits(m, seg)) for m in pop])
        pop = reproduce(pop, lives, rng)
    st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
    best = pop[int(np.argmax([life(hits(m, seg)) for m in pop]))]
    return (K, dict(comp=float(hits(best, seg).mean()), rank=code_rank(best, seg)))


def main():
    global LR, VOC
    if WORLD_KIND == "multiphase":
        LR, VOC = world_multiphase(np.random.default_rng(999))
    else:
        LR, VOC = RM.world_longrange(np.random.default_rng(999))[0], 6
    tasks = [(K, s) for K in KS for s in SEEDS]
    print(f"compression sweep: K in {KS} x {len(SEEDS)} seeds, M={M}, "
          f"world={WORLD_KIND} V={VOC} (gens={GENS})", flush=True)
    with Pool(min(18, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for K, d in res:
        agg.setdefault(K, []).append(d)
    comp = [np.mean([d["comp"] for d in agg[K]]) for K in KS]
    rank = [np.mean([d["rank"] for d in agg[K]]) for K in KS]
    print("\n  K(bottleneck)  competence   code-rank(used dims)")
    for i, K in enumerate(KS):
        print(f"  {K:>3}            {comp[i]:.3f}        {rank[i]:.2f}")
    full = comp[-1]
    knee = next((K for i, K in enumerate(KS) if comp[i] >= 0.95*full), KS[-1])
    print(f"\nfull-bandwidth (K={M}) competence = {full:.3f}")
    print(f"smallest K within 5% of full = {knee}  -> the abstraction the organism found "
          f"(world's effective dimension)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(KS, comp, "o-", color="#1b5e9e", lw=2)
    ax[0].axhline(0.95*full, ls="--", color="#888", lw=1); ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("bottleneck width K (log2)"); ax[0].set_ylabel("competence")
    ax[0].set_title("Does prediction survive compression?"); ax[0].grid(alpha=.3)
    ax[0].plot(knee, comp[KS.index(knee)], "r*", ms=16)
    ax[1].plot(KS, rank, "s-", color="#2e8b57", lw=2)
    ax[1].plot([1, M], [1, M], ":", color="#aaa")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log", base=2)
    ax[1].set_xlabel("bottleneck width K"); ax[1].set_ylabel("effective dims used (participation ratio)")
    ax[1].set_title("How many dims does the code actually use?"); ax[1].grid(alpha=.3)
    fig.suptitle(f"COMPRESSION axis: finite bandwidth forces abstraction ({WORLD_KIND} world)", weight="bold")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"compression_{WORLD_KIND}.png")
    plt.savefig(out, dpi=115); print("saved", out)


if __name__ == "__main__":
    main()
