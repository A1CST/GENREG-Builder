"""Pod ensemble-fodder: grammar v2 crossover run, seed 19."""
import radial_evo2 as e2
e2.run(rounds=400, pop_size=64, gens=12, freeze_top=8, seed=19, p_cross=0.5,
       ckpt_path="radial_data/evo2_s19_ckpt.json",
       out_path="radial_data/evo2_s19_cifar.json")
