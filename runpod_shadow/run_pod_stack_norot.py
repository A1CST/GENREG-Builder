"""Rotation ablation arm: identical stacked run, rot_deg=0 (no frame rotation
between spaces). Compare vs radial_stack_cap14.json (rot_deg=1.0)."""
import radial_stack as rs
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5, rot_deg=0.0,
               out_path="radial_data/radial_stack_norot.json")
