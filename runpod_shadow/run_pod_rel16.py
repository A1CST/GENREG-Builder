"""Completeness push #4: GRID=16 (2-px cells) + shift range +-8."""
import radial_stack as rs
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5,
               handoff="grid", grid_size=16, max_spaces=16,
               data_npz="radial_data/synth_rel.npz",
               out_path="radial_data/rel_grid16.json")
print("REL16 COMPLETE", flush=True)
