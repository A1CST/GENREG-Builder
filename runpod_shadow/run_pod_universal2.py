"""UNIVERSAL SUITE v2: identical frozen config; the residual-block
architecture gene (skip + bootstrap-no-op, per the resnet line) now competes
in the pool. Question: does the skip move the relational plateau?"""
import json
import radial_stack as rs

CFG = dict(pop_size=64, gens=12, max_rounds=200, seed=5,
           handoff="grid", grid_size=8, max_spaces=16)
TASKS = [("easy", "radial_data/u_easy.npz"), ("hard", "radial_data/u_hard.npz"),
         ("rel", "radial_data/u_rel.npz"), ("rel_sep", "radial_data/u_rel_sep.npz")]
summary = {}
for rule, path in TASKS:
    out = rs.run_stacked(data_npz=path,
                         out_path=f"radial_data/universal2_{rule}.json", **CFG)
    n_res = sum(1 for sp in out["spaces"] for _ in range(0))  # placeholder
    summary[rule] = {"test": out["test_acc"], "val": out["val_final"],
                     "spaces": out["space_caps"]}
    print(f"== {rule}: TEST {out['test_acc']}  spaces {out['space_caps']}", flush=True)
json.dump(summary, open("radial_data/universal2_suite.json", "w"), indent=1)
print("UNIVERSAL2 COMPLETE:", json.dumps(summary), flush=True)
