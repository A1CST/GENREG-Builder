"""Multi-seed A/B: sharpened moments (beta) on vs off, rel + rel_sep,
seeds 5/6/7 — significance before any more primitives."""
import json
import radial_stack as rs
CFG = dict(pop_size=64, gens=12, max_rounds=200,
           handoff="grid", grid_size=8, max_spaces=16)
res = {}
for beta_on in (True, False):
    rs.BETA_ON = beta_on
    for task, path in (("rel", "radial_data/u_rel.npz"),
                       ("rel_sep", "radial_data/u_rel_sep.npz")):
        for seed in (5, 6, 7):
            out = rs.run_stacked(seed=seed, data_npz=path,
                                 out_path=f"radial_data/ab_{task}_b{int(beta_on)}_s{seed}.json",
                                 **CFG)
            key = f"{task} beta={'on' if beta_on else 'off'} seed={seed}"
            res[key] = out["test_acc"]
            print(f"== {key}: {out['test_acc']}", flush=True)
json.dump(res, open("radial_data/beta_ab.json", "w"), indent=1)
print("BETA-AB COMPLETE:", json.dumps(res), flush=True)
