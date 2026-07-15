"""Completeness push #2: the SHIFT primitive (relative-offset gene per term)
+ GRID=8 + patient cap. If octant-relation is now expressible, this should
approach the 100% ceiling."""
import radial_stack as rs
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5,
               handoff="grid", grid_size=8,
               data_npz="radial_data/synth_rel.npz",
               out_path="radial_data/rel_shift.json")
print("REL-SHIFT COMPLETE", flush=True)
