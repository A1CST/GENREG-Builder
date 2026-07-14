"""Pod push-80 stage-2 on the seed-13 substrate, FRESH val window (slice 1)."""
import radial_push80 as p80

p80.run(rounds=300, pop_size=64, gens=12, seed=23, val_slice=1,
        stage1_ckpt="radial_data/evo2x_ckpt.json",
        ckpt_path="radial_data/push80_ckpt.json",
        out_path="radial_data/push80_cifar.json")
