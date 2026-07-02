"""Does external memory's value SCALE with how much there is to remember? Pure food fitness,
no shaping. Sweep the number of food patches; measure the same-organism marks-ablation gap.
If the gap grows with #patches, spatial memory pays more when there's more to remember --
re-deriving the spatial result from a WORLD-CONSEQUENCE, not a designed per-position reward."""
import numpy as np, embodied_forage as EF
LOG=open("forage_scaling_results.txt","w")
def out(s): print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
def ms(v): a=np.array(v,float); return f"{a.mean():.2f}+/-{a.std():.2f}"
out("Foraging (pure food fitness): marks-ablation gap vs number of food patches")
out("on=food with marks, off=same organism marks disabled; gap = memory's value in food")
for F in [1, 2, 3, 5]:
    EF.F = F; ons, offs, gaps = [], [], []
    for s in range(3):
        g = EF.evolve(gens=450, seed=s); rng = np.random.default_rng(50+s)
        on = EF.evaluate(g, rng); off = EF.evaluate(g, rng, no_marks=True)
        ons.append(on); offs.append(off); gaps.append(on-off)
    out(f"  F={F}: marks_on={ms(ons)}  marks_off={ms(offs)}  GAP={ms(gaps)}")
out("done"); LOG.close()
