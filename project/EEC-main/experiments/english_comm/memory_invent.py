"""DOES THE GENOME INVENT ITS OWN MEMORY?  (not us giving it a register)

The organism is strictly FEEDFORWARD: its action at step t is a function ONLY of what it
perceives right now -- (current input symbol, current content of a world slot). It has NO
internal state and NO state-update rule. We do not give it memory.

The only thing that persists across time is the WORLD: there is a slot (think: a patch of
ground) that simply keeps whatever was last written to it. The organism's feedforward policy
emits, each step, (a response, whether to WRITE, what to write). Nothing says this slot is
"memory" -- writing/reading/preserving is not prescribed.

Task that demands the past: a CUE appears at step 0; steps 1..L-1 show distractors; at the last
step the organism must output the CUE -- which it can no longer see. With no internal state, the
ONLY way to bridge time is to invent a policy that stores the cue in the world slot and reads it
back. If that policy evolves, the genome built its OWN memory mechanism.

Controls:
  - slot WRITABLE vs slot DISABLED (writes ignored, always empty) -> proves the bridge is the slot
  - inspect what gets written and when -> shows the invented store/preserve/read policy
"""
import numpy as np

V, L = 6, 4                              # symbol alphabet, sequence length (cue + L-1 distractors)
IN = V + (V + 1)                         # perceive: current symbol (V) + slot content (V+1, +1=empty)


def new_pop(N, rng):
    return dict(Wo=rng.normal(0, 0.5, (N, V, IN)),        # -> response symbol
                Wv=rng.normal(0, 0.5, (N, V, IN)),        # -> value to write
                wg=rng.normal(0, 0.5, (N, IN)))           # -> write gate (scalar)


def perceive(x, slot):
    p = np.zeros(IN); p[x] = 1.0; p[V + slot] = 1.0       # slot in [0..V] ; V means "empty"
    return p


def run(g, j, seq, writable=True, trace=False):
    slot = V                                              # world slot starts EMPTY
    out = 0; writes = []
    for t, x in enumerate(seq):
        p = perceive(x, slot)
        out = int(np.argmax(g["Wo"][j] @ p))              # response this step
        if writable and (g["wg"][j] @ p) > 0:             # organism CHOOSES to write
            slot = int(np.argmax(g["Wv"][j] @ p))         # ...and what
            writes.append((t, slot))
        # if it doesn't write, the world simply KEEPS the slot (persistence)
    return (out, slot, writes) if trace else out


def trial(g, j, rng, writable=True):
    cue = int(rng.integers(V))
    seq = [cue] + [int(rng.integers(V)) for _ in range(L - 1)]   # cue, then distractors
    return int(run(g, j, seq, writable) == cue)                  # must output the cue at the end


def play(g, rng, trials=16, writable=True):
    N = len(g["Wo"]); en = np.zeros(N)
    for j in range(N):
        en[j] = sum(trial(g, j, rng, writable) for _ in range(trials))
    return en


def recall(g, rng, n=600, writable=True):
    N = len(g["Wo"])
    return np.mean([trial(g, int(rng.integers(N)), rng, writable) for _ in range(n)])


def evolve(gens=500, N=48, mut=0.22, cull=0.25, seed=1, writable=True):
    rng = np.random.default_rng(seed); g = new_pop(N, rng)
    for t in range(gens):
        en = play(g, rng, writable=writable)
        Kc = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for key in ("Wo", "Wv", "wg"):
                m = rng.random(g[key][pa].shape) < 0.5
                g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Feedforward organism (NO internal state). World has one persistent writable slot.")
    print(f"Task: output the CUE seen {L-1} steps earlier. chance = {1/V:.2f}")
    print("=" * 72)
    print("T1 — can the genome INVENT a store-and-recall policy?  (slot writable vs disabled)")
    for wr in [False, True]:
        rc = [recall(evolve(gens=500, seed=s, writable=wr), rng, writable=wr) for s in range(4)]
        tag = "slot WRITABLE" if wr else "slot DISABLED (no store possible)"
        print(f"  {tag:34}: recall = {np.mean(rc):.2f}")
    print("=" * 72)
    print("T2 — what mechanism did it invent? (trace an evolved organism)")
    g = evolve(gens=500, seed=1, writable=True)
    # pick the best organism
    en = play(g, np.random.default_rng(5), trials=40); j = int(np.argmax(en))
    for _ in range(5):
        cue = int(rng.integers(V)); seq = [cue] + [int(rng.integers(V)) for _ in range(L - 1)]
        out, slot, writes = run(g, j, seq, True, trace=True)
        wr = ", ".join(f"t{t}:wrote {s}" for t, s in writes) or "never wrote"
        print(f"    cue={cue} seq={seq} -> output={out} {'OK' if out==cue else 'x'}   [{wr}]")
    print("  (if it writes the cue at t0 and outputs it at the end, it built its own memory)")
    print("=" * 72)
    print("T3 — ablation: take the evolved organism, DISABLE the world slot")
    print(f"  recall with slot = {recall(g, rng, writable=True):.2f}   "
          f"slot disabled = {recall(g, rng, writable=False):.2f}  (collapse => it stored in the world)")
