"""Pronoun-reference genome — ONE job: a pronoun should have a plausible noun antecedent
nearby. Kills "the wall and he ran she" — pronouns with no referent. A windowed class
discriminator whose negative injects a pronoun-heavy class where a real antecedent (a noun
class) should be, so the genome learns that a pronoun position needs a preceding referent.

Approximation note: coreference proper is beyond a tiny gradient-free window genome; this
captures the local "a pronoun needs a recent nominal, not another pronoun/function class"
regularity. Tested in the battery; cut if it doesn't earn its place. ~900 params.
"""
from genreg_train import genelib as gl


def train_pronref(n_classes=32, gens=2500, pop=200, seed=7, log=print):
    # corrupt the middle slot (the antecedent between context and the pronoun position)
    return gl.train_windowed(n_classes, corrupt_pos=1, name="pronref",
                             gens=gens, pop=pop, seed=seed, log=log)


def bias_tensor(champ, nc, K=3, E=8):
    return gl.window_bias_tensor(champ, nc, K, E)
