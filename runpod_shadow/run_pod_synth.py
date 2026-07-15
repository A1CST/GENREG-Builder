"""End-to-end stack test on the synthetic hierarchical task (trainable to
completion; the class exists only in the motif PAIRING, so deep spaces must
work). Full defaults, rotation on."""
import radial_synth
import radial_stack as rs

radial_synth.make_data()
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5, rot_deg=1.0,
               data_npz="radial_data/synth_hier.npz",
               out_path="radial_data/radial_stack_synth.json")
