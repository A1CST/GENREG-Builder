import os
import radial_kid

# kid_next.npz (50k/10k) already built by the first D run - reuse it so the
# only thing that changes vs D1 is the head rule.
if not os.path.exists(os.path.join(radial_kid.RD, "kid_next.npz")):
    radial_kid.make_next_d(n_train=50000, n_test=10000)

# GENOMES CARRY THE WEIGHT: head sees only frozen genome outputs; the word
# bag and ears are environment, not head inputs. warm=None - C's genomes
# measured negative (0.1703 -> 0.1699), nothing worth inheriting.
radial_kid.stage_d(head_mode="genomes", warm=None)
print('D2 DONE', flush=True)
