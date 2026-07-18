"""EAR ABLATION: is the kid's EYE contributing anything to language, or is
every language number just the `next` ear table read linearly?

Every D/C result so far sits on a bank of (word features from pixels) + (1536
ear channels from evolved corpus co-occurrence). The ears were justified as
"listening experience", but `embed_rs_next` separates words by what FOLLOWS
them - for next-word prediction that is close to a direct answer channel, the
evolved counterpart of a bigram continuation table. If removing the ears
collapses everything toward chance, the eye contributes ~nothing to language
and every number in this line is the ear.

Same pixels, same split, genome-only head, cached word features. The only
variable is ears on/off.

Contrast (ears ON, genome-only):
  next word   D2  test 0.1403  val 0.1514  anchor 0.1703
  first letter DFL test 0.2184  val 0.2159  anchor 0.2283
"""
import json

import radial_kid as K

res = {}

out = K.stage_c(data="kid_next.npz", stage="DNE", ears=False,
                head_mode="genomes", target="word", pop_size=96,
                max_rounds=600, seed=11,
                task="next word, NO EARS - pixels only (ear ablation)")
res["next_word_no_ears"] = {k: out.get(k) for k in
                            ("test_acc", "test_top5", "val_final",
                             "ref_anchor", "space_caps", "note")}
print(f"ABLATE next_word_no_ears: {json.dumps(res['next_word_no_ears'])}",
      flush=True)

out = K.stage_c(data="kid_next.npz", stage="DFLNE", ears=False,
                head_mode="genomes", target="first_letter", pop_size=96,
                max_rounds=600, seed=13,
                task="next word's first letter, NO EARS - pixels only")
res["first_letter_no_ears"] = {k: out.get(k) for k in
                               ("test_acc", "test_top5", "val_final",
                                "ref_anchor", "space_caps", "note")}
print(f"ABLATE first_letter_no_ears: "
      f"{json.dumps(res['first_letter_no_ears'])}", flush=True)

print("ABLATE SUMMARY: " + json.dumps(res), flush=True)
print("ABLATE DONE", flush=True)
