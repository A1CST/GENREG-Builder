"""Fat-R0 test (user: big R0s do amazing): R0 gets breathing room
(r0_cap=0.0002) while deep spaces keep the standard cap. rel + rel_sep,
seeds 5/6/7, beta on (default pool)."""
import json
import radial_stack as rs
CFG = dict(pop_size=64, gens=12, max_rounds=200,
           handoff="grid", grid_size=8, max_spaces=16, r0_cap=0.0002)
res = {}
for task, path in (("rel", "radial_data/u_rel.npz"),
                   ("rel_sep", "radial_data/u_rel_sep.npz")):
    for seed in (5, 6, 7):
        out = rs.run_stacked(seed=seed, data_npz=path,
                             out_path=f"radial_data/fatr0_{task}_s{seed}.json", **CFG)
        key = f"{task} seed={seed}"
        res[key] = {"test": out["test_acc"], "r0": out["space_caps"][0],
                    "spaces": out["space_caps"]}
        print(f"== {key}: {out['test_acc']}  R0={out['space_caps'][0]}  "
              f"spaces {out['space_caps']}", flush=True)
json.dump(res, open("radial_data/fatr0.json", "w"), indent=1)
print("FATR0 COMPLETE:", json.dumps(res), flush=True)
