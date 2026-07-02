"""CONVERSATION MEMORY -- context-dependent replies need memory across turns.

Real conversation: the right reply to a follow-up depends on what was said earlier. Turn 1 sets a
context c; turn 2 gives a prompt p; the correct reply h(c,p) depends on BOTH. A STATELESS organism
(replies only to the current prompt) cannot -- it has no access to c. A STATEFUL organism WRITES c to
a memory register at turn 1 and READS it at turn 2. We test that the stateful organism learns the
context-dependent reply where the stateless one is capped at ignoring context.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
C, P, M, RV = 6, 6, 6, 8; N = 100; BUDGET = 3000; SEEDS = 5
LOG = open(os.path.join(HERE, "conversation_memory_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def run(mode, seed):
    rng = np.random.default_rng(seed)
    h = rng.integers(RV, size=(C, P))                      # correct reply depends on BOTH context and prompt
    cc, pp = np.meshgrid(np.arange(C), np.arange(P), indexing="ij")
    cc, pp = cc.ravel(), pp.ravel(); tgt = h[cc, pp]       # all (c,p) situations
    if mode == "stateful":
        W = rng.integers(M, size=(N, C))                   # turn-1: write context -> memory id
        R = rng.integers(RV, size=(N, P, M))               # turn-2: (prompt, memory) -> reply
        genes = [("W", W, M), ("R", R, RV)]
        nix = np.arange(N)[:, None]
        def reply(): return R[nix, pp[None, :], W[:, cc]]  # (N, situations): R[n, prompt, mem(context)]
    else:                                                  # stateless: reply only from current prompt
        R = rng.integers(RV, size=(N, P)); genes = [("R", R, RV)]
        def reply(): return R[:, pp]
    G = {n: g for n, g, _ in genes}; hi = {n: h_ for n, _, h_ in genes}
    for t in range(BUDGET):
        en = (reply() == tgt[None]).sum(1).astype(float)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for n in G:
                m = rng.random(G[n][pa].shape) < 0.5; child = np.where(m, G[n][pa], G[n][pb])
                mm = rng.random(child.shape) < 0.06
                G[n][w] = np.where(mm, rng.integers(hi[n], size=child.shape), child)
    return float((reply() == tgt[None]).mean(1).max())     # best organism's accuracy over all situations


if __name__ == "__main__":
    out(f"CONVERSATION MEMORY: context-dependent reply h(context,prompt). C={C}, P={P}, mem M={M}. "
        f"pop {N}, budget {BUDGET}, {SEEDS} seeds.  chance {1/RV:.3f}; stateless ceiling ~ {1/C:.3f}.")
    for mode in ["stateless", "stateful"]:
        r = [run(mode, s) for s in range(SEEDS)]
        out(f"  {mode:>10}: context-dependent reply accuracy {ms(r)}")
    out("=" * 64)
    out("READING: stateful >> stateless (~1/C) => the organism must WRITE turn-1 context to memory and")
    out("READ it at turn-2 to reply correctly. Conversation needs memory; the organism evolves to use it.")
    out("done"); LOG.close()
