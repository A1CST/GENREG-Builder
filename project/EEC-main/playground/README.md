# GENREG Playground

A real-time control room for the constraint-driven world: the **board**, the **PO cone**,
and a **communication chart**, with every law and population parameter toggleable live.

## Run

```bash
cd playground
python3 playground.py
```

(Needs a display. Deps: PyQt6, pyqtgraph, numpy — all already installed.)

## What you control

- **Transport**: Play / Pause / Step / Reset, and speed (sim steps per frame).
- **Laws of existence** (toggles, each with its own parameter sliders):
  Energy (survival) · Time/Occam (move cost) · Perception cost · Entropy (memory decay) ·
  Scarcity (shared food) · Communication (signals).
- **Population / world**: population size, food patches, mutation rate, selection rate
  (cull fraction), kin spread (how near offspring land — controls kin structure), exploration noise.

## What you watch

- **Board** — organisms coloured by energy; yellow-ringed = currently signalling; yellow
  lines = who is following whose signal; green squares = food (size = amount).
- **PO cone** — one ring per active law; `PO` = number of laws; the green cross-section
  width = how much strategy diversity survives (the cone closing as constraints stack).
- **Communication** — signalling vs following over time.

## Adding a new constraint (for Claude)

Everything is driven by the constraint registry in `sim_engine.py`. To add a law:

1. Subclass `Constraint` with a `key`, `label`, `color`, optional `desc`, and `PARAMS`
   (each a `dict(name, val, lo, hi, step)` — these become sliders automatically).
2. Override only what you need:
   - `cost(sim)` → per-organism energy cost this step (an additive pressure).
   - `enables_select = True` if the law creates death/selection.
   - For effects on core logic (eating, memory, signalling), gate them inside
     `Sim.step()` with `if self.on('yourkey')`.
3. Append it in `make_constraints()`.

The GUI auto-builds the toggle, sliders, and a cone ring. No GUI code to touch.
