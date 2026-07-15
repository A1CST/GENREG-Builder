import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import cifar_internal as ci
DEMO = os.path.join(ci.cp.ROOT, "demo")
if __name__ == "__main__":
    # #1 second independent private language (different seed, same config)
    print("\n########## ENCODER B (seed 101) ##########", flush=True)
    ci.evolve_encoder(M=16, d=16, pop=48, gens=2500, n_anchor=1500, V=4, N=64,
                      seed=101, out=os.path.join(DEMO, "cifar_encoder_seed101.pkl"),
                      log_every=250)
    # #2 scaled encoder: bigger code, more filters/views, harder augmentation
    print("\n########## SCALED ENCODER (d=32, M=24, V=6, hard aug) ##########", flush=True)
    ci.evolve_encoder(M=24, d=32, pop=48, gens=2500, n_anchor=1500, V=6, N=64,
                      seed=7, hard_aug=True,
                      out=os.path.join(DEMO, "cifar_encoder_scaled.pkl"),
                      log_every=250)
