"""Pod runner: full grammar-v2 crossover run, seed 13 (independent replicate
of the local seed-7 run). Checkpoints in /workspace survive pod restarts;
STOP lever = touch /workspace/genreg-radial/radial_data/STOP_EVO."""
import radial_evo2 as e2

e2.run(rounds=400, pop_size=64, gens=12, freeze_top=8, seed=13, p_cross=0.5,
       ckpt_path="radial_data/evo2x_ckpt.json",
       out_path="radial_data/evo2x_cifar.json")
