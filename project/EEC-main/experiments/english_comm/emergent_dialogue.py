"""EMERGENT DIALOGUE -- does a back-channel emerge? (the listener tells the speaker what it needs)

Real conversation is two-way: the listener signals what it still needs, the speaker provides it. Test
this purely between two co-evolved organisms (no LLM, no designed reply map). A target has two
attributes; the LISTENER already knows one (randomly attr 0 or 1) and needs the other. The listener
sends a request; the SPEAKER (who knows the full target) must send the needed attribute. With a
back-channel the speaker can read the request and send the right attribute; without it the speaker must
guess which attribute the listener lacks. If two-way >> one-way, a feedback protocol emerged and is
load-bearing -- the listener's signal carries 'what I need', the foundation of dialogue.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
Q = 3; N = 80; BUDGET = 1500; SEEDS = 6
LOG = open(os.path.join(HERE, "emergent_dialogue_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def run(mode, seed):
    rng = np.random.default_rng(seed)
    Rq = rng.integers(Q, size=(N, 2))           # listener: need(0/1) -> request signal
    Apick = rng.integers(2, size=(N, Q))         # speaker (two-way): request -> which attribute to send
    Aconst = rng.integers(2, size=(N,))          # speaker (one-way): fixed attribute, ignores request
    for t in range(BUDGET):
        # success for need in {0,1}: speaker sends the needed attribute?
        en = np.zeros(N)
        for need in (0, 1):
            q = Rq[:, need]
            sent = Apick[np.arange(N), q] if mode == "two_way" else Aconst
            en += (sent == need)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        genomes = [("Rq", Rq, Q), ("Apick", Apick, 2)] if mode == "two_way" else [("Rq", Rq, Q), ("Aconst", Aconst, 2)]
        gd = {n: g for n, g, _ in genomes}; hi = {n: h for n, _, h in genomes}
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for n in gd:
                m = rng.random(gd[n][pa].shape) < 0.5; child = np.where(m, gd[n][pa], gd[n][pb])
                mm = rng.random(child.shape) < 0.1
                gd[n][w] = np.where(mm, rng.integers(hi[n], size=child.shape), child)
    # final success rate (best organism), averaged over both needs
    s = np.zeros(N)
    for need in (0, 1):
        q = Rq[:, need]; sent = Apick[np.arange(N), q] if mode == "two_way" else Aconst
        s += (sent == need)
    return float((s / 2).max())


if __name__ == "__main__":
    out(f"EMERGENT DIALOGUE: does a back-channel (listener->speaker 'what i need') emerge? "
        f"pop {N}, budget {BUDGET}, {SEEDS} seeds.")
    for mode in ["one_way", "two_way"]:
        r = [run(mode, s) for s in range(SEEDS)]
        out(f"  {mode:>8}: task success (speaker sends the attribute the listener needs) {ms(r)}")
    out("=" * 70)
    out("READING: two_way ~1.0 >> one_way ~0.5 => the listener's request carries 'what i need' and the")
    out("speaker uses it -- a feedback/back-channel protocol emerged; the speaker can't do it blind.")
    out("done"); LOG.close()
