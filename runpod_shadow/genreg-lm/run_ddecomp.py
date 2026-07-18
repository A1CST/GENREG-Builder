"""Decomposed stage D: ask the answerable questions, then compose.

D2 found 140 genomes of signal instantly and then stalled 2.75 points short
of a plain ridge head - the signature of a question too big to answer in one
leap. So:

  DFL   next word's FIRST LETTER (26 classes)  - stage B already spells
  DLEN  next word's LENGTH (8 classes)
  DC    the full 500-way next word, with DFL+DLEN genomes as ENVIRONMENT

All three read the same kid_next.npz pixels and the same split; the labels
for DFL/DLEN are derived from the same next-word answers. Head is genome-only
throughout, so every stage earns its own model.

Anchors for DC: D2 = 0.1403 test / 0.1514 val; reference linear = 0.1703.
"""
import json

import radial_kid as K

res = {}
for target in ("first_letter", "length"):
    out = K.stage_d_part(target)
    res[target] = {k: out.get(k) for k in
                   ("test_acc", "test_top5", "n_classes", "chance",
                    "space_caps", "ref_anchor", "genome_share", "note")}
    print(f"PART {target} RESULT: {json.dumps(res[target])}", flush=True)

out = K.stage_d_compose()
res["compose"] = {k: out.get(k) for k in
                  ("test_acc", "test_top5", "n_classes", "space_caps",
                   "ref_anchor", "genome_share", "total_params", "note")}
print(f"COMPOSE RESULT: {json.dumps(res['compose'])}", flush=True)
print("DDECOMP SUMMARY: " + json.dumps(res), flush=True)
print("DDECOMP DONE", flush=True)
