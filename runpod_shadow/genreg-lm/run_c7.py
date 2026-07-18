"""C7: stage C (cloze) re-run with the GENOME-ONLY head, on C6's exact 100k
data. The question: how much of the C line was measuring the head bug?

C6 (legacy head, 100k): TEST 0.1538 / 0.3455, 2 genomes, and the conclusion
"data scaling is spent". If C7's genomes come alive the way D2's did, that
conclusion was about the readout, not about cloze or about data.
"""
import json
import os
import shutil

import radial_kid as K

RD = K.RD

# C6's artifacts are the baseline of record - never let a re-run eat them.
for f in ("kid_stageC3.json", "kid_modelC3.json"):
    src = os.path.join(RD, f)
    dst = os.path.join(RD, f.replace(".json", "_c6.json"))
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"backed up {f} -> {os.path.basename(dst)}", flush=True)

out = K.stage_c(ears=True, pop_size=96, max_rounds=600, head_mode="genomes")

# archive C7's outputs under their own name (the run writes the C3 names)
for f in ("kid_stageC3.json", "kid_modelC3.json"):
    src = os.path.join(RD, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(RD, f.replace(".json", "_c7.json")))
# restore C6's model as the canonical C3 model (C7 is an experiment, not the
# new stage-C model until it earns that on the numbers)
for f in ("kid_stageC3.json", "kid_modelC3.json"):
    bak = os.path.join(RD, f.replace(".json", "_c6.json"))
    if os.path.exists(bak):
        shutil.copy2(bak, os.path.join(RD, f))

print("C7 RESULT: " + json.dumps(
    {k: out.get(k) for k in ("test_acc", "test_top5", "n_spaces",
                             "space_caps", "ref_anchor", "genome_share",
                             "total_params", "note")}), flush=True)
print("C7 DONE", flush=True)
