"""Complete the ablation grid: LENGTH with no ears.

The ablation falsified the "question-size law" (% of anchor fell with class
count WITH ears, but rises WITHOUT them - a size law would point the same way
in both). The surviving explanation is EAR SHARE of the anchor:

  question       ear share of anchor   evolution % of anchor
  first letter   0.0697/0.2283 = 31%          94.6%
  next word      0.1089/0.1703 = 64%          88.9%
  either, ears off             0%          106-122%

DLEN (length, 8 classes) is the missing cell: it scored 100.7% WITH ears, and
the ear-share story predicts its ears contribute LITTLE to length (how long the
next word is, is not what a continuation table encodes). If so its no-ears
anchor should sit only a little under 0.3186, and evolution should win here
too. If instead length's anchor collapses like next-word's did, ear share is
not the whole story either.
"""
import json

import radial_kid as K

out = K.stage_c(data="kid_next.npz", stage="DLENNE", ears=False,
                head_mode="genomes", target="length", pop_size=96,
                max_rounds=600, seed=13,
                task="next word's length, NO EARS - pixels only")
r = {k: out.get(k) for k in ("test_acc", "test_top5", "val_final",
                             "ref_anchor", "space_caps", "n_classes")}
print(f"ABLATE length_no_ears: {json.dumps(r)}", flush=True)
print("ABLATE2 DONE", flush=True)
