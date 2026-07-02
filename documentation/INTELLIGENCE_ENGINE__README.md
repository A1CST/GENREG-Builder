# INTELLIGENCE_ENGINE

A GENREG project to build a **self-replicating genome** — neural networks that produce mutated copies of themselves under metabolic pressure.

## Core Idea

Instead of training a network to perform a task, we train networks whose only job is to:

1. **Survive** — never let energy hit zero
2. **Reproduce** — produce a mutated copy of themselves that can also survive and reproduce

Both parent and offspring share the same goal. Selection is recursive: a genome only counts as "fit" if its offspring can also fit.

## The Six Signs of Life

The system instantiates all six signs of life rather than simulating them:

| Sign | Implementation |
|---|---|
| **Organization** | Structured genome (encoder + protein cascade + controller + output) |
| **Metabolism** | Energy budget drained by actions, restored by environment |
| **Growth** | Structural mutation can expand neurons/layers over generations |
| **Response to stimuli** | Environment signals feed protein cascade |
| **Reproduction** | Network outputs mutated weights when reproduce action fires |
| **Adaptation** | Selection pressure shapes lineages over generations |

The hypothesis: intelligence is what these six look like from the inside when running simultaneously under pressure.

## Architecture

```
INTELLIGENCE_ENGINE/
├── core/                          # GENREG infrastructure (copied)
│   ├── genreg_gpu_v5.py
│   ├── genreg_genome.py
│   ├── genreg_encoder_gpu.py
│   └── ...
├── self_replicating_genome.py     # Genome class — network that outputs its own mutated weights
├── code_mutator.py                # Mutation operators (value + structural)
├── reproduction_filter.py         # X-criteria selection filter
├── energy_environment.py          # Metabolic environment
├── lineage_tracker.py             # Parent-child tracking, evolution stats
├── intelligence_engine.py         # Main training loop
├── gui.py                         # Tkinter monitoring GUI
└── README.md
```

## Reproduction Filter

A genome only reproduces if it meets ANY of (configurable):
- Energy delta is positive across N steps
- Offspring viability test (offspring can produce its own valid child)
- Structural complexity increased without energy loss
- Code/weights shorter than parent with equal or better performance
- Sufficient divergence from parent (prevents clone stagnation)

The bar tightens dynamically — early generations accept any 1 of N criteria, later generations require more.

## NOT YET RUN

This project is built but not executed. Run from the project directory:

```bash
python intelligence_engine.py        # headless training
python gui.py                        # GUI monitor
```
