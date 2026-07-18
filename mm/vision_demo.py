"""vision_demo.py — orchestrates the two staples of the /vision_demo page and
writes one payload the page reads.

  Staple 1 UNION           — mm_merge.run(): fuse the frozen SHAPE bank and the
                             frozen LETTER bank into one 36-class head.
  Staple 2 CONTINUED TRAIN — vision_continue.run(): grow the SHAPE model with new
                             evolved genomes until it also reads letters, one head,
                             no separate letter model.

Assembles radial_data/vision_demo.json = {union, continued, meta}, records the run
into runs/vision_demo/ (house rule 4) and alerts on completion (house rule 3).

    python mm/vision_demo.py            # full (union ~9s + continued ~2-3 min)
    python mm/vision_demo.py --smoke    # quick pipeline check
"""
import json
import os
import time

import os as _os, sys as _sys                     # repo-root shim (run-anywhere)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # mm/ for siblings
import genreg_paths                               # noqa: F401

import mm_merge
import vision_continue
import vision_samples

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")


def run(smoke=False):
    t0 = time.time()
    print("=== vision_demo: staple 1 — UNION (fuse shape + letter models) ===", flush=True)
    union = mm_merge.run(n_per=40 if smoke else 140)

    print("=== vision_demo: CONTINUED TRAINING (grow the shape model) ===", flush=True)
    cont = vision_continue.run(smoke=smoke, warm=True)

    print("=== vision_demo: FROM-SCRATCH control (transfer-efficiency A/B) ===", flush=True)
    scratch = vision_continue.run(smoke=smoke, warm=False, save=False, rounds=80)

    # transfer efficiency: genomes each method needs to reach a shared accuracy
    def genomes_to(curve, target):
        for c in curve:
            if c["overall"] >= target:
                return c["n_new"]
        return None
    tgt = round(min(cont["after"]["overall"], scratch["after"]["overall"]), 4)
    efficiency = {
        "target": tgt,
        "continued": {"genomes_to_target": genomes_to(cont["curve"], tgt),
                      "final": cont["after"]["overall"], "final_genomes": cont["after"]["n_new_genomes"]},
        "scratch": {"genomes_to_target": genomes_to(scratch["curve"], tgt),
                    "final": scratch["after"]["overall"], "final_genomes": scratch["after"]["n_new_genomes"]},
    }
    print(f"[vision_demo] efficiency @ {tgt}: continued {efficiency['continued']['genomes_to_target']} "
          f"vs scratch {efficiency['scratch']['genomes_to_target']} genomes", flush=True)

    print("=== vision_demo: animation samples (checkpoints identifying shapes/letters) ===", flush=True)
    vision_samples.run()                           # -> radial_data/vision_demo_samples.json

    payload = {
        "generated": None,                         # stamped by the caller shell if wanted
        "union": union,
        "continued": cont,
        "scratch": scratch,
        "efficiency": efficiency,
        "meta": {"n_classes": 36, "chance": round(1 / 36, 4),
                 "shapes": 10, "letters": 26,
                 "gradient_free": True, "test_touched_once": True,
                 "seconds": round(time.time() - t0)},
    }
    os.makedirs(RD, exist_ok=True)
    with open(os.path.join(RD, "vision_demo.json"), "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[vision_demo] wrote radial_data/vision_demo.json ({payload['meta']['seconds']}s)",
          flush=True)

    # --- record the run (house rules 3 & 4): five-file set + alert on end ---
    fused = next((r for r in union["results"] if r["bank"].startswith("FUSED")), {})
    cfg = {"union_n_per": 40 if smoke else 140, "continue_rounds": cont.get("curve") and
           len(cont["curve"]) - 1, "smoke": smoke, "n_classes": 36}
    hist = [{"round": c["n_new"], "fitness": c["overall"], "added": None, "n": c["n_new"]}
            for c in cont["curve"]]
    stats = {"union_fused": fused.get("overall"),
             "continued_before": cont["before"]["overall"],
             "continued_after": cont["after"]["overall"],
             "continued_letters_before": cont["before"]["letters"],
             "continued_letters_after": cont["after"]["letters"],
             "n_new_genomes": cont["after"]["n_new_genomes"],
             "seconds": payload["meta"]["seconds"]}
    try:
        import dot_runs
        dot_runs.record("vision_demo", cfg, hist, stats,
                        label=(f"vision demo: union fused {fused.get('overall')}, "
                               f"continued letters {cont['before']['letters']}->"
                               f"{cont['after']['letters']}"),
                        tags=["vision", "multimodal", "union", "continued-training"],
                        notify=True)
    except Exception as exc:
        print(f"[vision_demo] run record skipped: {exc}", flush=True)

    print(f"[vision_demo] DONE ({payload['meta']['seconds']}s)", flush=True)
    return payload


if __name__ == "__main__":
    import sys
    run(smoke=("--smoke" in sys.argv))
