"""UNIVERSAL SUITE v3: frozen config; the moment-stat family completed
(centroid comx/comy join mean/max/std). Location is now a readable quantity."""
import json
import radial_stack as rs
CFG = dict(pop_size=64, gens=12, max_rounds=200, seed=5,
           handoff="grid", grid_size=8, max_spaces=16)
TASKS = [("easy", "radial_data/u_easy.npz"), ("hard", "radial_data/u_hard.npz"),
         ("rel", "radial_data/u_rel.npz"), ("rel_sep", "radial_data/u_rel_sep.npz")]
summary = {}
for rule, path in TASKS:
    out = rs.run_stacked(data_npz=path,
                         out_path=f"radial_data/universal3_{rule}.json", **CFG)
    summary[rule] = {"test": out["test_acc"], "val": out["val_final"],
                     "spaces": out["space_caps"]}
    print(f"== {rule}: TEST {out['test_acc']}  spaces {out['space_caps']}", flush=True)
json.dump(summary, open("radial_data/universal3_suite.json", "w"), indent=1)
print("UNIVERSAL3 COMPLETE:", json.dumps(summary), flush=True)
