# GENREG engine

A gradient-free neuroevolution substrate, built from the ground up one piece at a time. No backpropagation: a
population of small networks is varied by mutation and selected by a task fitness. Each piece is a single,
self-tested module; together they form one engine that runs supervised, control, and agent worlds.

About 700 lines of NumPy, no ML framework. Every module has a self-test you can run directly
(`python3 genome.py`, `python3 mutation.py`, …). `ROADMAP.md` tracks the pieces and the principles.

## The pieces

| # | module | what it adds |
|---|---|---|
| 1 | `genome.py` | the network: parameters + a deterministic forward pass |
| 2 | `mutation.py` | relative mutation `N(0,1)·ms·(|w|+ε)`, self-adapting per-layer rate/scale |
| 3 | `reproduction.py` | `copy()` and gentle crossover (low-rate, like-shaped parents only) |
| 4 | `evolver.py` | the generational loop; live sexual/asexual toggle; hooks for the later pieces |
| 5 | `fitness.py` | `robust()` — score a genome by the **median** over N evaluations, not the mean |
| 6 | `constraints.py` | `shaped()` — fitness = task score − resource costs; the world shapes the model |
| 7 | `precision.py` | per-neuron bit-depth gene + quantization + a precision cost (8-bit is a no-op) |
| 8 | `dimension.py` | grow/shrink hidden width (function-preserving growth); the size cost now bites |
| 9 | `recurrence.py` | opt-in memory: recurrent weights + a `leak` gene, for sequential worlds |
| 10 | `telemetry.py` | per-generation population stats (mean/median fitness, bits, leak) |
| 11 | `checkpoint.py` | save/load a genome exactly (weights + all genes), atomic write |
| 12 | `worlds.py` | `supervised` / `classification` / `episodic` — one fitness type, three world kinds |

`example.py` uses all twelve at once; `__init__.py` exposes the public API.

## Usage

```python
import baseline as gr                       # (with the parent dir on the path)

X, T = ...                                  # a supervised task
world  = gr.supervised(X, T)
fitness = gr.robust(gr.shaped(world, gr.size_cost(0.001)), n=5)   # median, minus a size cost
best = gr.Evolver(n_in, n_out, fitness, telemetry=gr.snapshot).run(200)
gr.save(best, "best.pkl")
```

A control or agent world is a rollout that returns a scalar, wrapped the same way:
```python
fitness = gr.robust(gr.episodic(my_rollout), n=5)
```

## Principles (from `ROADMAP.md`)

1. No gradients, ever — fitness is a world consequence, not a score that points at the answer.
2. Median, not mean, over evaluations; the population mean, not the single best, is the progress signal.
3. Crossover of neural weights is destructive — keep it gentle and like-shaped, or off.
4. Relative, self-adapting mutation; the search rate is itself a gene.
5. Default precision is an exact no-op; precision only changes under a cost.
6. The world shapes the model — size, precision, and memory are selected by constraints, not set by hand.
7. A constraint must fit the world it shapes. `example.py` notes this directly: a size cost on a memory task
   strangles the capability before it forms, so it is left off there.

## Honest notes

Small scale, illustrative, not a benchmark. On smooth supervised tasks a tuned gradient baseline is competitive
or better; the gradient-free substrate's distinct value is non-differentiable objectives, evolving discrete
structure (precision, size), and worlds with no usable gradient. The `example.py` recall task reaches an error
of about 0.07 against a memoryless floor of ~0.45 — clearly using memory, not perfect. Results vary with seed,
population, and budget.
