"""Pod: full emergent-cap stacked run (radial grammar, resnet structure)."""
import radial_stack as rs
rs.run_stacked(pop_size=64, gens=12, max_rounds=200, seed=5, rot_deg=1.0)
