"""Single import surface for the genreg-engine neuroevolution substrate.

The engine lives at ``project/genreg-engine-main`` and uses absolute imports
(``import genome``, ``from mutation import ...``) resolved by putting its own
directory on ``sys.path``. We do that here once, then re-export the pieces the
trainer uses, plus a couple of helpers that adapt a genome for the browser
(weight matrices for the Microscope, a compact summary for the HUD).

Nothing else in ``genreg_train`` should import the engine directly.
"""
import os
import sys

import numpy as np

ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "project", "genreg-engine-main",
)
if not os.path.isdir(ENGINE_DIR):
    raise RuntimeError(f"genreg-engine not found at {ENGINE_DIR}")
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

# --- engine pieces (see project/genreg-engine-main/ROADMAP.md) ---
from genome import Genome                                                   # 1
from evolver import Evolver, chain                                          # 4
from fitness import robust                                                  # 5
from constraints import shaped, weight_cost, size_cost, param_cost          # 6
from precision import init_precision, mutate_precision, mean_bits           # 7
from dimension import mutate_dimensions                                     # 8
from recurrence import enable, fresh_state, rstep, mutate_recurrence        # 9
from telemetry import snapshot, series                                     # 10
from checkpoint import save, load                                          # 11

__all__ = [
    "Genome", "Evolver", "chain", "robust", "shaped", "weight_cost", "size_cost",
    "param_cost", "init_precision", "mutate_precision", "mean_bits", "mutate_dimensions",
    "enable", "fresh_state", "rstep", "mutate_recurrence", "snapshot", "series",
    "save", "load", "genome_layers", "genome_summary",
]


def genome_layers(g, round_dp=4):
    """The genome's weight matrices as JSON-friendly layers for the Microscope.

    Returns a list of ``{"rows", "cols", "w"}`` where ``w`` is a flat row-major
    list of floats (length rows*cols). We expose the two feedforward matrices
    (input->hidden ``W1`` as [H, n_in]; hidden->output ``W2`` as [n_out, H]),
    oriented [out, in] to match the Microscope's row=neuron convention.
    """
    layers = []
    for W, rows, cols in ((g.W1, g.H, g.n_in), (g.W2, g.n_out, g.H)):
        M = np.asarray(W, np.float32).T          # engine stores [in, out]; want [out, in]
        assert M.shape == (rows, cols), (M.shape, (rows, cols))
        layers.append({
            "rows": int(rows), "cols": int(cols),
            "w": np.round(M.reshape(-1), round_dp).astype(float).tolist(),
        })
    return layers


def genome_summary(g):
    """Small dict for the HUD: width, memory leak, mean bit-depth."""
    out = {"H": int(g.H), "n_params": int(g.n_params())}
    if hasattr(g, "leak"):
        out["leak"] = round(float(g.leak), 3)
    if hasattr(g, "prec"):
        out["bits"] = round(float(mean_bits(g)), 2)
    return out
