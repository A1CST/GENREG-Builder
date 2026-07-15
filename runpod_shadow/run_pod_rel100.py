"""Completeness push on the rel task: GRID=8 hand-off (4-px cells, finer than
the 8-px min separation) + cap 0.0005 (spaces grind further). Target: 100%."""
import radial_stack as rs
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5,
               handoff="grid", grid_size=8,
               data_npz="radial_data/synth_rel.npz",
               out_path="radial_data/rel_grid8.json")
print("REL100 COMPLETE", flush=True)
