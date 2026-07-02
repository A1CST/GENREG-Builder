import numpy as np, relational_syntax as RS
LOG=open("retest_bag_results.txt","w")
def out(s): print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
def ms(v): a=np.array(v,float); return f"{a.mean():.3f}+/-{a.std():.3f}"
out("FAIR bag vs sequential (both 1500 gens = near plateau); full-event AND swap (order-specific)")
out("swap = distinguishing 'A acts B' from 'B acts A'; the bag's TRUE ceiling on swap is ~chance")
for bag in [False, True]:
    fulls, swaps = [], []
    for s in range(5):
        g=RS.evolve(gens=1500, seed=s, bag=bag); rng=np.random.default_rng(s)
        fulls.append(RS.role_acc(g, RS.all_events(), rng)); swaps.append(RS.swap_acc(g, RS.all_events(), rng))
    out(f"  {'BAG' if bag else 'SEQUENTIAL':11}: full={ms(fulls)}  swap={ms(swaps)}")
out("done"); LOG.close()
