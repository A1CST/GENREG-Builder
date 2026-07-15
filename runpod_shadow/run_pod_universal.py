"""THE UNIVERSAL GRAMMAR SUITE — one frozen configuration, four tasks, zero
per-task toggles. Every primitive permanently in the gene pool (shifts,
raw-environment skips, gates, all scales, spatial hand-off); evolution alone
decides which primitives each task deserves. Config: grid 8, cap 0.0005,
max_spaces 16, pop 64, gens 12, seed 5."""
import json
import radial_synth
import radial_stack as rs

CFG = dict(pop_size=64, gens=12, max_rounds=200, seed=5,
           handoff="grid", grid_size=8, max_spaces=16)
TASKS = [("easy",    "radial_data/u_easy.npz"),
         ("hard",    "radial_data/u_hard.npz"),
         ("rel",     "radial_data/u_rel.npz"),
         ("rel_sep", "radial_data/u_rel_sep.npz")]
summary = {}
for rule, path in TASKS:
    radial_synth.make_data(rule=rule, path=path)
    out = rs.run_stacked(data_npz=path,
                         out_path=f"radial_data/universal_{rule}.json", **CFG)
    summary[rule] = {"test": out["test_acc"], "val": out["val_final"],
                     "spaces": out["space_caps"]}
    print(f"== {rule}: TEST {out['test_acc']}  spaces {out['space_caps']}",
          flush=True)
json.dump(summary, open("radial_data/universal_suite.json", "w"), indent=1)
print("UNIVERSAL SUITE COMPLETE:", json.dumps(summary), flush=True)
