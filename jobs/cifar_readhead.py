import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import cifar_internal as ci
DEMO = os.path.join(ci.cp.ROOT, "demo")
if __name__ == "__main__":
    print("\n########## LINEAR PROBE (16 -> 10) ##########", flush=True)
    ci.evolve_readhead(encoder_pkl="cifar_encoder_seed7.pkl", hidden=0,
                       pop=160, gens=2000, minibatch=400, n_head=1500, seed=13)
    shutil.copy(os.path.join(DEMO, "cifar_readhead.pkl"),
                os.path.join(DEMO, "cifar_readhead_linear.pkl"))
    print("\n########## EVOLVED MLP HEAD (16 -> 64 -> 10) ##########", flush=True)
    ci.evolve_readhead(encoder_pkl="cifar_encoder_seed7.pkl", hidden=64, H=64,
                       pop=160, gens=2000, minibatch=400, n_head=1500, seed=13)
    shutil.copy(os.path.join(DEMO, "cifar_readhead.pkl"),
                os.path.join(DEMO, "cifar_readhead_mlp.pkl"))
