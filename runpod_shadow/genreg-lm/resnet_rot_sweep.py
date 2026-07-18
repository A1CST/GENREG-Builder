"""Rotation-angle sweep for the stacked residual net. R0 is loaded from cache
(deterministic, unaffected by rotation), so each point only re-runs the
downstream rotated spaces.

Modes:
  block  — rotate every adjacent feature pair (~F/2 planes at once)
  embed  — FAITHFUL radial move: embed the feature set into its behavioral map
           (SVD principal axes) and rotate the dominant PC0-PC1 plane, a SINGLE
           axis, by the angle.

  rot_deg = per-space increment: R1 = rot_deg, R2 = 2*rot_deg, R3 = 3*rot_deg
Usage:  python resnet_rot_sweep.py [block|embed]
"""
import json
import sys
import time

import resnet_evo as r

MODE = sys.argv[1] if len(sys.argv) > 1 else "embed"
ANGLES = [0, 15, 30, 45, 60, 90]     # 0 = control (no rotation)
rows = []
t0 = time.time()
for deg in ANGLES:
    print(f"\n===== {MODE} rotation {deg}°/space =====", flush=True)
    out = r.run_stacked(rot_deg=deg, rot_mode=MODE, reuse_r0=True, record=False,
                        verbose=True, out_path=fr"F:\Resnet\sweep_{MODE}_rot{deg}.json")
    rows.append({"rot_deg": deg, "space_caps": out["space_caps"],
                 "n_spaces": out["n_spaces"], "val_final": out["val_final"],
                 "test_acc": out["test_acc"]})

summary = {"sweep": f"{MODE}_rotation_deg_per_space", "mode": MODE, "angles": ANGLES,
           "control_no_rotation": rows[0]["test_acc"],
           "single_space_ref": 0.6593, "rows": rows,
           "seconds": round(time.time() - t0)}
with open(fr"F:\Resnet\rot_sweep_{MODE}_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

print(f"\n\n=========  {MODE.upper()} ROTATION SWEEP RESULT  =========", flush=True)
print(f"{'rot/space':>10} | {'spaces':<22} | {'val':>7} | {'TEST':>7}", flush=True)
for x in rows:
    print(f"{str(x['rot_deg'])+chr(176):>10} | {str(x['space_caps']):<22} | "
          f"{x['val_final']:>7} | {x['test_acc']:>7}", flush=True)
print(f"\ncontrol (0°) test {rows[0]['test_acc']} | single-space ref 0.6593 | "
      f"{summary['seconds']}s", flush=True)
print("SWEEP DONE", flush=True)
