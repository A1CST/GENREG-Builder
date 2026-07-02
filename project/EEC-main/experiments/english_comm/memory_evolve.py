"""WHAT CONSTRAINT MAKES MEMORY EVOLVE?  (clean isolation)

Capability (substrate, not answer): a recurrent STATE register h (dim H); its read/write
weights (Wx input->state, Wh state->state, output decoders) are free and evolved. Nothing
says what to store.

Constraint (the world law): the environment presents a STIMULUS SPREAD OVER TIME -- a sequence
of L symbols, one per step, each GONE on the next step -- and survival depends on responding to
the WHOLE history (here: reproducing the sequence at the end). No single moment is sufficient.
The only bridge from the past to the response is internal state. So memory is the only way to
survive, and should evolve.

Decisive tests (because reproducing EARLY positions is impossible without carrying them forward):
  - memory ALLOWED (Wh free) vs BLOCKED (Wh forced 0 -> h only reflects the latest symbol)
  - per-position recall: blocked can only recover the LAST symbol; memory recovers earlier ones
  - ablation: sever an evolved organism's recurrence -> recall of the past collapses
  - capacity sweep H: does how-far-back-it-can-remember scale with register size?
"""
import numpy as np

L, V = 4, 8                          # sequence length, symbol alphabet


def new_pop(N, rng, H):
    return dict(Wx=rng.normal(0, 0.5, (N, H, V)),          # input -> state
                Wh=rng.normal(0, 0.5, (N, H, H)),          # state -> state  (THE MEMORY)
                b=rng.normal(0, 0.1, (N, H)),
                out=rng.normal(0, 0.5, (N, L, V, H)), H=H)  # one decoder per output position


def respond(g, j, seq, block=False):
    """read the sequence one symbol at a time; reproduce all L positions from the FINAL state."""
    H = g["H"]; h = np.zeros(H)
    for s in seq:
        x = np.zeros(V); x[s] = 1.0
        rec = 0 if (block or H == 0) else g["Wh"][j] @ h
        h = np.tanh(g["Wx"][j] @ x + rec + g["b"][j])
    return [int(np.argmax(g["out"][j, p] @ h)) for p in range(L)]


def play(g, rng, trials=10, block=False):
    N = len(g["spk"]) if "spk" in g else len(g["Wx"]); en = np.zeros(N)
    seqs = rng.integers(0, V, size=(trials, L))
    for seq in seqs:
        seq = list(seq)
        for j in range(N):
            out = respond(g, j, seq, block)
            en[j] += sum(int(out[p] == seq[p]) for p in range(L))   # survival = positions recalled
    return en


def recall_by_pos(g, rng, n=400, block=False):
    N = len(g["Wx"]); hit = np.zeros(L); tot = 0
    for _ in range(n):
        seq = list(rng.integers(0, V, size=L)); j = int(rng.integers(N))
        out = respond(g, j, seq, block)
        for p in range(L): hit[p] += (out[p] == seq[p])
        tot += 1
    return hit / tot


def evolve(gens=400, N=48, H=16, mut=0.22, cull=0.25, seed=1, block=False):
    rng = np.random.default_rng(seed); g = new_pop(N, rng, H)
    for t in range(gens):
        en = play(g, rng, block=block)
        Kc = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for key in ("Wx", "Wh", "b", "out"):
                m = rng.random(g[key][pa].shape) < 0.5
                g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"Stimulus over time: read {L} symbols (alphabet {V}) one per step; reproduce all {L} at the end.")
    print(f"chance per position = {1/V:.2f}. Without memory only the LAST position is recoverable.")
    print("=" * 72)
    print("T1 — memory ALLOWED vs BLOCKED: per-position recall (pos 0 = oldest, pos 3 = newest)")
    for block in [True, False]:
        rc = np.mean([recall_by_pos(evolve(gens=400, H=16, seed=s, block=block), rng, block=block)
                      for s in range(3)], 0)
        tag = "BLOCKED (no memory)" if block else "ALLOWED (memory)"
        print(f"  {tag:20}: recall by position = [{rc[0]:.2f} {rc[1]:.2f} {rc[2]:.2f} {rc[3]:.2f}]  mean={rc.mean():.2f}")
    print("=" * 72)
    print("T2 — ablation: evolve WITH memory, then sever the recurrence")
    g = evolve(gens=400, H=16, seed=1)
    on = recall_by_pos(g, rng, block=False); off = recall_by_pos(g, rng, block=True)
    print(f"  intact recall  = [{on[0]:.2f} {on[1]:.2f} {on[2]:.2f} {on[3]:.2f}]  mean={on.mean():.2f}")
    print(f"  severed recall = [{off[0]:.2f} {off[1]:.2f} {off[2]:.2f} {off[3]:.2f}]  mean={off.mean():.2f}")
    print("  (early positions collapse when memory is severed => it stored the past in the register)")
    print("=" * 72)
    print("T3 — capacity sweep: how far back can it remember vs register size H?")
    for H in [0, 1, 4, 8, 16, 32]:
        rc = np.mean([recall_by_pos(evolve(gens=350, H=H, seed=s), rng) for s in range(2)], 0)
        print(f"  H={H:2}: recall [{rc[0]:.2f} {rc[1]:.2f} {rc[2]:.2f} {rc[3]:.2f}]  mean={rc.mean():.2f}")
