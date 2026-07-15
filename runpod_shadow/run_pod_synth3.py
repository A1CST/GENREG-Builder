"""Hard-rule synth, SPATIAL-GRID hand-off (the fix): A/B vs run_synth2's
scalar hand-off on identical data."""
import radial_stack as rs
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5,
               data_npz="radial_data/synth_hard.npz",
               out_path="radial_data/radial_stack_synth_hard_grid.json")
