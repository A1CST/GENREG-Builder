"""Completeness push #5: no-overlap variant (Chebyshev sep >= 12) isolates
whether the residual ~9 pts are R0 perception on merged objects."""
import radial_synth
import radial_stack as rs
radial_synth.make_data(rule="rel_sep", path="radial_data/synth_rel_sep.npz")
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5,
               handoff="grid", grid_size=8, max_spaces=16,
               data_npz="radial_data/synth_rel_sep.npz",
               out_path="radial_data/rel_sep.json")
print("REL-SEP COMPLETE", flush=True)
