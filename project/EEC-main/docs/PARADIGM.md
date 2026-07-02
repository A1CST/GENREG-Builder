# EEC — the paradigm (read this before touching the code)

This is NOT gradient training wearing an evolutionary costume. Do not grade
organisms by accuracy, loss, or "vs baseline." That is measuring how well the
organism approximates what backprop would find — i.e. implicitly evolving a
predictor. Every time we did that we called genuine emergence ("organism invents
selective perception under metabolic pressure") a failure because a number
didn't move.

## The two paradigms

- **Gradient model**: defined by what you ADD. Parameters, capacity, knobs.
  The model is the SUM of its weights. Complexity = parameter count (how many
  knobs it took to fit the curve). Noise is the enemy; backprop moves precisely
  down the error gradient.
- **GENREG / evolutionary**: defined by what you REMOVE. Each constraint (a law
  of existence) eliminates an infinite set of survival strategies. The model is
  the INTERSECTION of its constraints. Noise is the MEDIUM — the organism learns
  which signals in the noise matter to *its survival*.

## The PO metric (the real score)

The native unit of an evolutionary model is NOT parameters. It is **constraints**.

> PO = how many laws of existence are required to collapse infinite possible
> organisms down to the behavior we need.

Three laws (energy, time, perception) produce one class of survivor. Add memory
rent → four → the viable space shrinks again. The **minimum constraint set that
makes the target behavior inevitable is the model's true complexity.**

This also kills the "how does it compare to a 7B model" problem: apples/oranges.
One asks "how many knobs to fit this function." The other asks "how few laws to
make this function the only thing that survives."

## The geometry — a cone, not a cube

We are restricting INFINITY, so there are no edges. Mouth of the cone (top) =
every organism that could exist. Each constraint is a ring slicing the
cross-section smaller. The tip = the single surviving behavior. PO→0 is an
asymptote (you can always add another law; you never reach a point). 2D→square,
3D→sphere; with the constraints axis the sphere collapses into the cone. **The X
axis IS the metric. The shape of the graph IS the model's identity.** See
`po_cone.py` / `po_cone.png`.

## How GENREG reasons differently

It does not start from the output and work backward (backprop). It builds
**internal structure first**, because structure is what survives; the output is
whatever narrow action the organism can afford. Understanding comes before
expression. (Proof: the perception organism's output collapsed to `,` while its
hidden state self-organized a 31-dim syntactic world model — commas, sentence
boundaries, determiners — with no teacher. The plateau wasn't stagnation; it was
the organism building structure the output metric couldn't see.)

## Working rules

1. Never report accuracy/loss as success. Report what EMERGED and count the laws.
2. Read the STATE, not the token output.
3. A plateau is a place to LOOK INSIDE, not to stop.
4. Build the next constraint from what the organism actually did.
5. Laws of existence so far: ENERGY, TIME, PERCEPTION, MEMORY(rent).
   Pocket / candidates: SCARCITY (shared finite resource, carrying capacity),
   ENTROPY/DECAY (state degrades unless maintained), MORTALITY (finite life,
   generational turnover), REPRODUCTION COST.
