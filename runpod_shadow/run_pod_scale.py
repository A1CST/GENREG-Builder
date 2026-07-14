"""Pod scale test: grammar v2, population 128 (2x), crossover 0.5, seed 17.
Question: does a bigger population per round buy a higher ceiling or just
faster rounds-to-ceiling? H100 has the headroom to ask."""
import radial_evo2 as e2

e2.run(rounds=400, pop_size=128, gens=12, freeze_top=8, seed=17, p_cross=0.5,
       ckpt_path="radial_data/evo2p128_ckpt.json",
       out_path="radial_data/evo2p128_cifar.json")
