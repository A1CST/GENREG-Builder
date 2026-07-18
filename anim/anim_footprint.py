"""anim_footprint.py — compute the parameter footprint + on-disk size of every
model behind the animation page, so the page can show how TINY these
gradient-free, CPU-inference models are. Writes radial_data/anim_footprint.json.
"""
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import json
import os

from anim_infer import count_params

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RD = os.path.join(_HERE, "radial_data")

# (key, display name, role, checkpoint file)
MODELS = [
    ("motion_path", "Motion model (path)", "name the animation from motion (10-way)", "anim_model.json"),
    ("motion_shape", "Motion model (shape)", "name the shape from motion (10-way)", "anim_model_shape.json"),
    ("motion_color", "Motion model (random-color bg)", "the robustness / scaling base", "anim_model_color.json"),
    ("tracker", "Cursor tracker (Attention 1)", "localize the red cursor (x,y)", "dot_model.json"),
    ("classifier", "Shape classifier (Attention 1b)", "read the shape under the cursor (10)", "dot_shape_model.json"),
    ("interactive", "Interactive classifier", "the mouse demo (5 distinct shapes)", "dot_shape_sub_model.json"),
]


def _model_stats(fn):
    path = os.path.join(_RD, fn)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ck = json.load(f)
    disk = os.path.getsize(path)
    if "spaces" in ck:                          # motion model (list of evolved spaces)
        genomes = sum(len(sp) for sp in ck["spaces"])
        evolved = count_params(ck["spaces"])
        head = None                             # closed-form readout, refit at load
    else:                                       # dot model (genomes + stored readout W)
        genomes = len(ck["genomes"])
        evolved = count_params(ck["genomes"])
        head = len(ck["W"]) * len(ck["W"][0]) if "W" in ck else None
    total = evolved + (head or 0)
    return {"genomes": genomes, "evolved_params": evolved, "head_params": head,
            "total_params": total, "disk_bytes": disk}


def run():
    out = []
    for key, name, role, fn in MODELS:
        s = _model_stats(fn)
        if s:
            out.append({"key": key, "name": name, "role": role, "file": fn, **s})
    tot = {
        "n_models": len(out),
        "genomes": sum(m["genomes"] for m in out),
        "evolved_params": sum(m["evolved_params"] for m in out),
        "total_params": sum(m["total_params"] for m in out),
        "disk_bytes": sum(m["disk_bytes"] for m in out),
    }
    footprint = {"device": "CPU", "gradient_free": True, "models": out, "totals": tot}
    with open(os.path.join(_RD, "anim_footprint.json"), "w") as f:
        json.dump(footprint, f, indent=1)
    print(f"[footprint] {tot['n_models']} models, {tot['evolved_params']:,} evolved params, "
          f"{tot['disk_bytes']/1024:.0f} KB on disk", flush=True)
    return footprint


if __name__ == "__main__":
    run()
