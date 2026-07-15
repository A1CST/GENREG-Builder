"""Completeness push #3: shift gene + GRID=8 + patient cap + NO depth guard
(max_spaces=16 — let the economy decide how deep)."""
import radial_stack as rs
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5,
               handoff="grid", grid_size=8, max_spaces=16,
               data_npz="radial_data/synth_rel.npz",
               out_path="radial_data/rel_deep.json")
print("REL-DEEP COMPLETE", flush=True)
