"""H1/H2 test: is syntax-emergence a property of the WORLD (relations) or of the exact
reward slope I designed? Re-run the relational world under several fitness shapes -- from pure
ALL-OR-NOTHING (strictest world-consequence: the partner survives only if the WHOLE message is
understood) to various 'more-correct-helps' shapes -- and ask whether order STILL emerges and is
STILL load-bearing (scramble collapse). If it survives across shapes, the world drives it; if only
my hand-picked linear per-role works, it was reward-shaping. -> relational_reward_results.txt
"""
import numpy as np
import relational_syntax as RS

LOG = open("relational_reward_results.txt", "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def shaped_credit(nc, shape):
    if shape == "all_or_nothing": return 3 if nc == 3 else 0      # whole message or nothing
    if shape == "threshold2":     return 1 if nc >= 2 else 0      # coarse: 'mostly understood'
    if shape == "per_role":       return nc                       # the original linear grading
    if shape == "square":         return nc * nc                  # convex: getting more pays more
    if shape == "any":            return 1 if nc >= 1 else 0      # almost no gradient
    return nc


def evolve_shaped(shape, gens=600, N=48, mut=0.22, cull=0.25, seed=1):
    rng = np.random.default_rng(seed); g = RS.new_pop(N, rng)
    ev = RS.all_events()
    for t in range(gens):
        en = np.zeros(N)
        for _ in range(6):
            perm = rng.permutation(N)
            for k in range(0, N - 1, 2):
                i, j = perm[k], perm[k + 1]
                for sp, ls in ((i, j), (j, i)):
                    e = ev[rng.integers(len(ev))]
                    d = RS.decode(g, ls, RS.emit(g, sp, e))
                    nc = (d[0] == e[0]) + (d[1] == e[1]) + (d[2] == e[2])
                    c = shaped_credit(nc, shape); en[sp] += c; en[ls] += c
        Kc = max(1, int(cull * N)); order = np.argsort(en); worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for key in ("spk", "ag", "act", "tg"):
                m = rng.random(g[key][pa].shape) < 0.5
                g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


if __name__ == "__main__":
    out("=" * 66)
    out("SYNTAX under different fitness shapes (chance full-event = %.3f)" % (1.0 / len(RS.all_events())))
    out("does order EMERGE (full >> chance) and is it LOAD-BEARING (full >> scrambled)?")
    out("=" * 66)
    for shape in ["all_or_nothing", "any", "threshold2", "per_role", "square"]:
        full, scr = [], []
        for s in range(5):
            g = evolve_shaped(shape, gens=600, seed=s); rng = np.random.default_rng(900 + s)
            full.append(RS.role_acc(g, RS.all_events(), rng))
            scr.append(RS.role_acc(g, RS.all_events(), rng, scramble=True))
        fm, sm = np.mean(full), np.mean(scr)
        verdict = "ORDER EMERGES & load-bearing" if (fm > 0.08 and fm > 2 * sm) else \
                  ("emerges, weak order" if fm > 0.08 else "does NOT emerge (~chance)")
        out(f"  {shape:15}: full={ms(full)}  scrambled={ms(scr)}  -> {verdict}")
    out("\ndone")
    LOG.close()
