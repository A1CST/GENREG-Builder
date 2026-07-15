"""Hard-rule synth: y = (t+b) mod 5 + 5*[t==b] — composition mandatory."""
import radial_synth
import radial_stack as rs
radial_synth.make_data(rule="hard", path="radial_data/synth_hard.npz")
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5, rot_deg=1.0,
               data_npz="radial_data/synth_hard.npz",
               out_path="radial_data/radial_stack_synth_hard.json")
