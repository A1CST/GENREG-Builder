"""High-seed re-tests of the headline contrasts (mean +/- std). Separates real effects
from 2-seed noise. -> retest_results.txt
"""
import numpy as np
import relational_syntax as RS
import memory_invent as MI
import spatial_seq as SS
import english_comm as EC

LOG = open("retest_results.txt", "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f} +/- {a.std():.3f}  (n={len(a)})"

S = 8
out("=" * 64)
out("HIGH-SEED RE-TESTS — mean +/- std")
out("=" * 64)

out("\n[B] English stability under channel noise (sex vs clone):")
for noise in [0.0, 0.15, 0.3]:
    for repro in ["clone", "sexual"]:
        vals = [EC.evolve(anchor=2.0, repro=repro, gens=250, noise=noise, seed=s)[1][-1][2] for s in range(S)]
        out(f"   noise={noise:<4} {repro:7}: english_usage {ms(vals)}")

out("\n[F] Syntax: order-dependence (scramble) + bag control:")
full, scr, bagf = [], [], []
for s in range(S):
    g = RS.evolve(gens=600, seed=s); rng = np.random.default_rng(100 + s)
    full.append(RS.role_acc(g, RS.all_events(), rng))
    scr.append(RS.role_acc(g, RS.all_events(), rng, scramble=True))
    gb = RS.evolve(gens=600, seed=s, bag=True)
    bagf.append(RS.role_acc(gb, RS.all_events(), np.random.default_rng(200 + s)))
out(f"   sequential full intact   : {ms(full)}")
out(f"   sequential full scrambled: {ms(scr)}   (collapse => order load-bearing)")
out(f"   BAG full                 : {ms(bagf)}  (vs sequential intact)")

out("\n[I] Invented external memory ablation (recall with vs without world slot):")
on, off = [], []
for s in range(S):
    g = MI.evolve(gens=500, seed=s); rng = np.random.default_rng(300 + s)
    on.append(MI.recall(g, rng, writable=True)); off.append(MI.recall(g, rng, writable=False))
out(f"   slot writable: {ms(on)}")
out(f"   slot disabled: {ms(off)}")

out("\n[J] Spatial sequence: W=1 vs W=3, freeze-head ablation:")
SS.NCUE = 3; w1, w3, frz = [], [], []
for s in range(6):
    g1 = SS.evolve(gens=450, W=1, seed=s); w1.append(SS.recall(g1, np.random.default_rng(400 + s)))
    g3 = SS.evolve(gens=450, W=3, seed=s); rng = np.random.default_rng(500 + s)
    w3.append(SS.recall(g3, rng)); frz.append(SS.recall(g3, rng, freeze=True))
out(f"   W=1 (one cell): {ms(w1)}")
out(f"   W=3 movable   : {ms(w3)}")
out(f"   W=3 frozen    : {ms(frz)}")

out("\n[K] Invented-memory recall vs delay (is it really delay-robust?):")
for L in [2, 4, 8, 12, 16, 24]:
    MI.L = L
    vals = [MI.recall(MI.evolve(gens=400, seed=s), np.random.default_rng(700 + s)) for s in range(5)]
    out(f"   delay L={L:<3}: {ms(vals)}")

out("\ndone")
LOG.close()
