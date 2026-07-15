"""The REALIGNED discriminating test: positional relation task (octant of
satellite vs ring anchor, global translation, identity distractors, 1.5x
noise). Scalar hand-off arm, then grid hand-off arm — shared cached R0."""
import radial_synth
import radial_stack as rs

radial_synth.make_data(rule="rel", path="radial_data/synth_rel.npz")
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5,
               handoff="scalar",
               data_npz="radial_data/synth_rel.npz",
               out_path="radial_data/rel_scalar.json")
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5,
               handoff="grid",
               data_npz="radial_data/synth_rel.npz",
               out_path="radial_data/rel_grid.json")
print("REL A/B COMPLETE", flush=True)
