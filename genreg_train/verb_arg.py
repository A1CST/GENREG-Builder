"""Verb-argument genome — ONE job: a verb should have a subject (an actor class) before
it. Kills "and ran of" — verbs appearing with no actor. A windowed class discriminator
that corrupts the LEFT slot of a window (the subject position): a real fragment
subject-verb-object vs one whose actor slot has been replaced. Biases the ORDER skeleton
so a verb-ish continuation is only favored when a plausible subject precedes it.

Approximation note: the induced classes aren't labelled subject/verb, so this learns the
distributional shape "valid left-context for this continuation" rather than a parsed
subject. Tested empirically in the battery; cut if it doesn't earn its place. ~900 params.
"""
from genreg_train import genelib as gl


def train_verbarg(n_classes=32, gens=2500, pop=200, seed=7, log=print):
    return gl.train_windowed(n_classes, corrupt_pos=0, name="verbarg",
                             gens=gens, pop=pop, seed=seed, log=log)


def bias_tensor(champ, nc, K=3, E=8):
    return gl.window_bias_tensor(champ, nc, K, E)
