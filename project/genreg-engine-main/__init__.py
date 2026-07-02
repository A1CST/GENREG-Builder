"""
GENREG engine — a gradient-free neuroevolution substrate, built piece by piece.

Public API (see ROADMAP.md for the twelve pieces and the principles). Import the whole thing:

    import baseline as gr
    world = gr.supervised(X, T)
    best  = gr.Evolver(n_in, n_out, gr.robust(world, 5), telemetry=gr.snapshot).run(200)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # let the modules' absolute imports resolve

from genome import Genome                                                   # 1
from mutation import mutate, relative_step, init_strategy                   # 2
from reproduction import copy, crossover                                    # 3
from evolver import Evolver, chain                                          # 4
from fitness import robust                                                  # 5
from constraints import shaped, weight_cost, size_cost, param_cost          # 6
from precision import (init_precision, mutate_precision, qforward,          # 7
                       precision_cost, mean_bits, quantize)
from dimension import resize_H, mutate_dimensions                           # 8
from recurrence import enable, fresh_state, rstep, mutate_recurrence        # 9
from telemetry import snapshot, series                                      # 10
from checkpoint import save, load                                           # 11
from worlds import supervised, classification, episodic                     # 12

__all__ = [
    "Genome", "mutate", "relative_step", "init_strategy", "copy", "crossover", "Evolver", "chain", "robust",
    "shaped", "weight_cost", "size_cost", "param_cost", "init_precision", "mutate_precision", "qforward",
    "precision_cost", "mean_bits", "quantize", "resize_H", "mutate_dimensions", "enable", "fresh_state",
    "rstep", "mutate_recurrence", "snapshot", "series", "save", "load", "supervised", "classification", "episodic",
]
