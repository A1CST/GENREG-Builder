"""Cross-seed R0 ensemble experiment.

Train N R0s at different seeds, each pushed to ~VAL_TARGET val, then ask:
does ensembling the seeds raise val/test accuracy — and does it help MORE
before stacking (R0-only readout) or AFTER stacking (full spatial-grid tower)?

Ensemble = average of per-seed class probabilities (softmax of the ridge
logits), then argmax. Each seed's R0 is cached (per-seed file), so re-runs and
the stacking that follows are cheap.

Usage:  python resnet_ensemble.py [target] [seed1 seed2 seed3 ...]
"""
import json
import sys
import time

import numpy as np

import resnet_evo as r

TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.64
SEEDS = [int(x) for x in sys.argv[2:]] or [11, 23, 37]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def acc(probs, y):
    return float((np.mean(probs, 0).argmax(1) == y).mean())


def main():
    t0 = time.time()
    z = np.load(r"radial_data/cifar_full.npz") if False else \
        np.load(__import__("os").path.join(r._HERE, "radial_data", "cifar_full.npz"))
    ytr, yte = z["ytr"], z["yte"]
    n_fit = int(len(ytr) * 0.8)
    yv = ytr[n_fit:]

    r0_v, r0_t, st_v, st_t = [], [], [], []
    rows = []
    for s in SEEDS:
        print(f"\n===== seed {s} -> R0 to val {TARGET}, then stack =====", flush=True)
        out = r.run_stacked(seed=s, val_target=TARGET, reuse_r0=True, record=False,
                            verbose=True, out_path=fr"F:\Resnet\ens_seed{s}.json")
        L = out["_logits"]
        r0_v.append(softmax(L["r0_val"])); r0_t.append(softmax(L["r0_test"]))
        st_v.append(softmax(L["stack_val"])); st_t.append(softmax(L["stack_test"]))
        rows.append({"seed": s, "space_caps": out["space_caps"],
                     "r0_val": out["r0_val_acc"], "r0_test": None,   # filled below
                     "stack_val": out["stack_val_acc"], "stack_test": out["test_acc"]})
        # per-seed test accs from the logits (test-once, honest)
        rows[-1]["r0_test"] = float((r0_t[-1].argmax(1) == yte).mean())
        rows[-1]["stack_test"] = float((st_t[-1].argmax(1) == yte).mean())

    # ensembles (mean of probs across seeds)
    res = {
        "target": TARGET, "seeds": SEEDS,
        "per_seed": rows,
        "before_stacking": {
            "mean_val": round(float(np.mean([x["r0_val"] for x in rows])), 4),
            "mean_test": round(float(np.mean([x["r0_test"] for x in rows])), 4),
            "ensemble_val": round(acc(r0_v, yv), 4),
            "ensemble_test": round(acc(r0_t, yte), 4)},
        "after_stacking": {
            "mean_val": round(float(np.mean([x["stack_val"] for x in rows])), 4),
            "mean_test": round(float(np.mean([x["stack_test"] for x in rows])), 4),
            "ensemble_val": round(acc(st_v, yv), 4),
            "ensemble_test": round(acc(st_t, yte), 4)},
        "seconds": round(time.time() - t0)}
    res["before_stacking"]["ens_gain_test"] = round(
        res["before_stacking"]["ensemble_test"] - res["before_stacking"]["mean_test"], 4)
    res["after_stacking"]["ens_gain_test"] = round(
        res["after_stacking"]["ensemble_test"] - res["after_stacking"]["mean_test"], 4)
    with open(r"F:\Resnet\ensemble_summary.json", "w") as f:
        json.dump(res, f, indent=1)

    print("\n\n=================  R0 ENSEMBLE RESULT  =================", flush=True)
    print(f"target val {TARGET} | seeds {SEEDS}", flush=True)
    print(f"{'seed':>5} | {'caps':<18} | {'R0 val':>7} {'R0 test':>8} | "
          f"{'stk val':>7} {'stk test':>8}", flush=True)
    for x in rows:
        print(f"{x['seed']:>5} | {str(x['space_caps']):<18} | "
              f"{x['r0_val']:>7.4f} {x['r0_test']:>8.4f} | "
              f"{x['stack_val']:>7.4f} {x['stack_test']:>8.4f}", flush=True)
    b, a = res["before_stacking"], res["after_stacking"]
    print("\n            mean-of-seeds   ensemble    ens gain", flush=True)
    print(f"BEFORE stk  test {b['mean_test']:.4f}   {b['ensemble_test']:.4f}   "
          f"{b['ens_gain_test']:+.4f}", flush=True)
    print(f"AFTER  stk  test {a['mean_test']:.4f}   {a['ensemble_test']:.4f}   "
          f"{a['ens_gain_test']:+.4f}", flush=True)
    print(f"\nsingle-space ref 0.6593 | matured-single-stack 0.6638 | "
          f"{res['seconds']}s", flush=True)
    print("ENSEMBLE DONE", flush=True)


if __name__ == "__main__":
    main()
