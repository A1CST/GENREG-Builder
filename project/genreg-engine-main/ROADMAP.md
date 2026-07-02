# GENREG engine — ground-up build

Building the gradient-free neuroevolution engine one piece at a time, so nothing is lost. Each piece is small,
self-contained, has its own self-test, and is reviewed before the next. This file is the contract: the ordered
pieces, plus the principles we already learned the hard way and must not re-break.

## Pieces (build order)

- [x] **1. Genome** — the network: parameters + a deterministic forward pass. Nothing else yet.
- [x] **2. Relative mutation** — perturb each weight by `N(0,1) * ms * (|w| + eps)`; self-adapting per-layer
      rate `mr` and scale `ms`. No fixed global step.
- [x] **3. Reproduction** — `copy()`, and gentle crossover (low-rate, like-shaped parents only).
- [x] **4. Evolver loop** — population, evaluate, select, elitism, next generation. Fixed seeds. Live
      sexual/asexual toggle (`ev.sexual`, default crossover).
- [x] **5. Fitness aggregation** — score a genome as the **median over N evaluations**, not the mean (a single
      lucky evaluation must not crown a genome).
- [x] **6. Constraints / world-shaping** — fitness = task score minus resource costs (size, precision, …). The
      constraints, not the experimenter, decide the surviving architecture.
- [x] **7. Evolvable precision** — per-neuron bit-depth gene + quantization + a precision cost. Default 8-bit is
      an exact no-op.
- [x] **8. Dimension evolution** — grow/shrink hidden width `H` (and memory `M`, perception `R`) with no fixed
      ceilings; weight matrices resize. Growth is function-preserving; size cost now bites.
- [x] **9. Recurrence / memory** — optional recurrent state with a leak gene, for sequential / control / agent
      tasks.
- [x] **10. Telemetry** — per-generation population averages (mean & median, not just best), bits, saturation,
      self-adapting rates.
- [x] **11. Checkpoints** — save/load named genomes; a rolling best; load to resume or to run inference.
- [x] **12. Task / world interface** — one clean way to plug in supervised, control, and agent worlds.

## Principles (do not re-break — each was learned by breaking it)

1. **No gradients, ever.** No backprop, no hybrid, and no designed reward-gradient or shaping. Fitness is a
   world consequence, never a score that points at the answer.
2. **Median, not mean, over evaluations.** High-variance worlds let a lucky single run inflate a genome; report
   and select on the median. The single best is not the learning signal — the population mean is.
3. **Crossover of neural weights is destructive.** Hidden units are not aligned between genomes; heavy mixing
   halves offspring skill. Keep crossover gentle (low rate) and only between like-shaped parents — or off.
4. **Relative mutation, self-adapting.** Steps proportional to weight magnitude; the search rate is itself a
   gene that evolves.
5. **Default precision = full = no-op.** Evolvable precision must not change behavior until a precision cost is
   turned on. Under a flat task it collapses to uniform minimum; mixed precision needs heterogeneous demand.
6. **The world shapes the model.** Architecture (size, precision, memory) is selected by resource costs, not set
   by hand.
7. **Verify before celebrating; document failure modes.** Check baselines, replay genomes, report where it
   fails. No overselling.

## Status

All 12 pieces done. Each module has a self-test (`python3 <module>.py`); `example.py` runs the whole engine on
one task; `README.md` is the guide.
