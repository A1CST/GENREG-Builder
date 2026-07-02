# VOLATILE
### A Gradient-Free Evolutionary System for Geometric Knowledge Representation

---

## What This Is

Volatile is a research system exploring a specific hypothesis: that **knowledge can be represented as a geometric shape**, and that **evolution is a valid mechanism for finding the stable configuration of that shape** against a real environment.

It is not a neural network trainer. It is not a reinforcement learning system. It uses no backpropagation anywhere. It is an evolutionary system where agents live, experience their environment, reproduce, and die — and the intelligence that emerges is whatever survives.

---

## The Core Theory

### The Three Signals

Every environment is modeled through three semantically distinct signals:

**What Is** — ground truth. The anchor. The immutable record of what actually happened. This is the hardest signal to get wrong. Errors here are penalized most heavily. In humans, a corrupted "what is" layer produces distorted perception of reality — the rest of the cognitive structure can be internally consistent and still be completely wrong, because it's orbiting a false anchor.

**What If** — the counterfactual. Not a raw prediction, but the *relationship* between the adjacent branch and the anchor. What if distance measures how far the world could have been from what actually was. The system doesn't predict what_if directly — it predicts the *deviation* from what_is, and how that deviation relates to the anchor signal.

**What Could Be** — forward projection. A plausible future extrapolated from what_is. Measured not independently but for its *geometric consistency* relative to both other signals. The question isn't "is this prediction accurate in isolation" but "does this projection sit in the right place relative to the anchor and the counterfactual."

Together, these three signals form a triangle in signal space. The genome must hold this triangle with correct proportions. This is not joint distribution fitting — it is geometric shape maintenance.

### The Perfect Shape

For any given environment and genome capacity, there exists a **perfect configuration** — the most accurate shape that capacity can hold. This is not an absolute truth. It is the limit of what expressiveness allows. A small genome holds a simpler shape. A larger genome can hold a richer one. The noise floor — irreducible uncertainty in the environment — sets the hard lower bound that no genome of any size can pass.

The gap between a genome's current energy and the noise floor is the gap between current understanding and the limit of what's knowable given the signals available.

### Energy as Shape Distance

A genome's **energy** is how far its predicted shape is from the perfect configuration. This is measured in three weighted terms:

- **Term 1 (weight 1.0):** Anchor error — mean and variance of what_is prediction
- **Term 2 (weight 0.7):** Counterfactual relationship error — how accurately the genome models the deviation of what_if from what_is, and how that deviation correlates with the anchor
- **Term 3 (weight 0.5):** Geometric consistency error — whether what_could_be sits in the right position relative to both other signals (measured via cross-correlations)

Lower energy = the genome's shape better matches the geometry of the real environment.

### The Ratchet

Evolution does not run against a fixed fitness target. The fitness target advances. Once the population's best genome holds its shape below a threshold for a sustained number of generations, **the ratchet advances** — the bar rises, and any genome that can't keep up with the new standard dies.

The ratchet is a one-way gate. It never retreats. This forces the population upward continuously, with no coasting. It is the selection pressure that gives the system direction.

### Volatile Lifecycle

Every genome is an **Agent** with a finite life. Death comes from three conditions:

1. **Age** — the genome reaches its maximum lifespan. Exception: the current population best cannot die of age. It is protected until something better replaces it.
2. **Struggle** — the genome's energy stays above the death line (ratchet + margin) for too many consecutive generations. It cannot survive in the current environment.
3. **Freeze** — the genome gets close enough to the noise floor that its shape is considered crystallized. It is removed from active evolution and preserved as a **frozen genome** — a permanent record of a configuration reality actually allows.

Before dying, an agent reproduces if its energy is good enough. The population is sustained entirely through lineage. There is no random injection of new genomes.

### Trust Inheritance and Crossover

When two agents reproduce, the child genome is produced through **trust-weighted crossover**:

- Each parent has a **trust score** — how well its energy compares to the current ratchet level. High trust = close to the ratchet, holding the shape well.
- The crossover samples each gene position from one parent or the other, with probability proportional to trust scores raised to an adaptive exponent.
- The exponent scales with the *difference* in trust scores: when parents are equal, crossover is roughly uniform. When one parent is much better than the other, that parent dominates the child genome.
- After crossover, mutation is applied. Mutation rate and scale increase when average trust is low — the population explores more aggressively when it's struggling.

This means good structure propagates efficiently when a clear winner exists, but diversity is maintained when the population is roughly equal.

### Frozen Genomes

A frozen genome is the output of the system. It represents a configuration that:
- Survived long enough in the environment to reach near the noise floor
- Holds all three signal relationships in geometric correspondence with reality
- Cannot be predicted away — it found what the environment actually allows

Frozen genomes are the **crystallized knowledge** the system produces. Each one found a slightly different stable configuration. The set of all frozen genomes is a library of valid shapes for this environment.

---

## How to Build It From Scratch

### Dependencies

```
python 3.10+
numpy
```

That's it. No deep learning frameworks. No GPU required (though larger hidden sizes benefit from one).

### Step 1: Define the Environment

Generate a reproducible data stream with three semantically distinct channels.

```python
def generate_stream(n=200, seed=42):
    rng = np.random.RandomState(seed)
    t   = np.linspace(0, 4 * np.pi, n)
    
    NOISE = 0.18
    
    # what_is: ground truth signal + irreducible noise
    clean_is      = np.sin(t)
    what_is       = clean_is + rng.normal(0, NOISE, n)
    
    # what_if: counterfactual — same signal, adjacent branch
    perturbation  = np.sin(t * 1.3 + 1.1) * 0.4
    what_if       = what_is * 0.85 + perturbation + rng.normal(0, NOISE, n)
    
    # what_could_be: smoothed forward projection + noise
    kernel        = np.ones(8) / 8
    smoothed      = np.convolve(clean_is, kernel, mode='same')
    what_could_be = smoothed * 1.1 + rng.normal(0, NOISE * 1.3, n)
    
    return np.stack([what_is, what_if, what_could_be], axis=1)
```

The key constraint: **the noise is seeded and fixed**. Same stream every run. The irreducible noise is part of the environment, not a training artifact. The perfect shape includes it.

### Step 2: Compute the Perfect Shape Targets

Before evolution starts, compute what the perfect shape looks like for this environment:

```python
stream = generate_stream()
WI, WIF, WCB = stream[:,0], stream[:,1], stream[:,2]

# Anchor targets
wi_mean, wi_std = WI.mean(), WI.std()

# Counterfactual relationship targets
diff         = WIF - WI
diff_mean    = diff.mean()
diff_std     = diff.std()
diff_corr_wi = np.corrcoef(WI, diff)[0, 1]

# Geometric consistency targets
wcb_corr_wi  = np.corrcoef(WI,  WCB)[0, 1]
wcb_corr_wif = np.corrcoef(WIF, WCB)[0, 1]
wcb_mean     = WCB.mean()
wcb_std      = WCB.std()
```

These targets don't change. They define what a perfect genome would predict. Evolution is the search for a genome that matches them.

### Step 3: Define the Genome and Energy Function

The genome is a small neural network: input(3) → hidden(N) → output(3). No activation on output. tanh on hidden layer.

```python
HIDDEN      = 24  # start here; 64 gets closer to noise floor
GENOME_SIZE = 3*HIDDEN + HIDDEN + HIDDEN*3 + 3

def forward(genome, x):
    i  = 0
    W1 = genome[i:i+3*HIDDEN].reshape(3, HIDDEN); i += 3*HIDDEN
    b1 = genome[i:i+HIDDEN];                       i += HIDDEN
    W2 = genome[i:i+HIDDEN*3].reshape(HIDDEN, 3);  i += HIDDEN*3
    b2 = genome[i:i+3]
    return np.tanh(x @ W1 + b1) @ W2 + b2

def calc_energy(genome):
    out  = forward(genome, stream)
    p_wi, p_wif, p_wcb = out[:,0], out[:,1], out[:,2]
    
    # Term 1: anchor fidelity (highest weight)
    e_wi = (abs(p_wi.mean() - wi_mean) + abs(p_wi.std() - wi_std)) * 1.0
    
    # Term 2: counterfactual relationship
    p_diff = p_wif - p_wi
    e_wif  = (abs(p_diff.mean() - diff_mean) +
               abs(p_diff.std()  - diff_std)  +
               abs(np.corrcoef(p_wi, p_diff)[0,1] - diff_corr_wi)) * 0.7
    
    # Term 3: geometric consistency of projection
    p_corr_wi  = np.corrcoef(p_wi,  p_wcb)[0,1] if p_wcb.std() > 1e-6 else 0.0
    p_corr_wif = np.corrcoef(p_wif, p_wcb)[0,1] if p_wcb.std() > 1e-6 else 0.0
    e_wcb = (abs(p_wcb.mean()   - wcb_mean)     +
              abs(p_wcb.std()    - wcb_std)       +
              abs(p_corr_wi  - wcb_corr_wi)       +
              abs(p_corr_wif - wcb_corr_wif)) * 0.5
    
    return e_wi + e_wif + e_wcb
```

### Step 4: Calibrate Thresholds

Sample 100 random genomes to understand the energy distribution. All thresholds are derived from this — never hardcoded.

```python
sample_energies = sorted([calc_energy(np.random.randn(GENOME_SIZE) * 0.3) for _ in range(100)])
initial_ratchet = np.percentile(sample_energies, 35)  # start at 35th percentile
freeze_thresh   = noise_floor + 0.08                   # near the actual floor
```

### Step 5: The Agent

Each agent wraps a genome with lifecycle state:

```python
class Agent:
    def __init__(self, genome, lifespan=30):
        self.genome   = genome
        self.lifespan = lifespan
        self.age      = 0
        self.energy   = calc_energy(genome)
        self.struggle = 0  # consecutive gens above death line
        self.alive    = True
        self.frozen   = False
    
    def step(self, ratchet, is_best=False):
        self.age    += 1
        self.energy  = calc_energy(self.genome)
        death_line   = ratchet + RATCHET_MARGIN  # 0.20
        
        self.struggle = self.struggle + 1 if self.energy > death_line else 0
        
        if   self.energy   <= freeze_thresh:    self.frozen = True;  self.alive = False
        elif self.struggle >= PATIENCE:          self.alive  = False  # 10 gens struggling
        elif self.age      >= self.lifespan and not is_best:
                                                 self.alive  = False
```

Key rule: **the current population best cannot die of age.** It is protected until something better replaces it.

### Step 6: Trust-Weighted Crossover

```python
def trust_score(energy, ratchet, margin=0.20):
    return max(0.0, 1.0 - (energy - ratchet) / margin)

def crossover(g1, g2, t1, t2):
    total = t1 + t2 + 1e-8
    exp   = 1.0 + abs(t1 - t2)   # adaptive: larger gap = stronger bias
    p1    = (t1 / total) ** exp
    p2    = (t2 / total) ** exp
    mask  = np.random.rand(len(g1)) < (p1 / (p1 + p2 + 1e-8))
    return np.where(mask, g1, g2)

def make_child(p1, p2, ratchet):
    t1, t2   = trust_score(p1.energy, ratchet), trust_score(p2.energy, ratchet)
    child_g  = crossover(p1.genome, p2.genome, t1, t2)
    avg_t    = (t1 + t2) / 2.0
    rate     = 0.15 + (1.0 - avg_t) * 0.15   # mutate more when struggling
    scale    = 0.10 + (1.0 - avg_t) * 0.10
    mask     = np.random.rand(len(child_g)) < rate
    child_g += mask * np.random.randn(len(child_g)) * scale
    return child_g
```

### Step 7: The Ratchet

The ratchet advances only after the population has sustained performance below the current level for `RATCHET_STABILITY` consecutive generations (5 is a good default). This prevents a lucky single genome from prematurely raising the bar.

```python
stable_gens = 0
RATCHET_STEP      = 0.03
RATCHET_STABILITY = 5

# Inside the generation loop:
if best_energy < ratchet - RATCHET_STEP:
    stable_gens += 1
    if stable_gens >= RATCHET_STABILITY:
        ratchet     = max(noise_floor + 0.02, best_energy + RATCHET_STEP * 0.5)
        stable_gens = 0
else:
    stable_gens = 0
```

### Step 8: The Main Loop

```python
population = [Agent(np.random.randn(GENOME_SIZE) * 0.3) for _ in range(30)]
frozen     = []

for gen in range(N_GENERATIONS):
    best_id = min(population, key=lambda a: a.energy).id
    
    # Step all agents
    for a in population:
        a.step(ratchet, is_best=(a.id == best_id))
    
    # Collect dead and frozen
    newly_frozen = [a for a in population if a.frozen]
    newly_dead   = [a for a in population if not a.alive and not a.frozen]
    frozen.extend(newly_frozen)
    
    # Reproduce — no random injection, lineage only
    reproducers = [a for a in newly_dead + newly_frozen
                   if a.energy < ratchet + RATCHET_MARGIN]
    
    children = []
    if len(reproducers) >= 2:
        reproducers.sort(key=lambda a: a.energy)
        for i in range(min(len(reproducers), 6)):
            p1 = reproducers[i]
            p2 = reproducers[(i+1) % len(reproducers)]
            g  = make_child(p1, p2, ratchet)
            children.append(Agent(g))
    
    population = [a for a in population if a.alive and not a.frozen]
    population.extend(children)
    
    # Advance ratchet if stable
    # ... (see Step 7)
```

**Critical constraint: no random genome injection.** If the population goes extinct, the run ends. The system lives or dies on its own lineage. This is not a flaw — it is the point. A system that can't maintain a viable population in its environment doesn't deserve to survive.

---

## Parameters and What They Do

| Parameter | Default | Effect |
|---|---|---|
| `HIDDEN` | 24 | Genome expressiveness. Hidden 6 ≈ 75% gap closed. Hidden 64 ≈ 99% gap closed. |
| `POP_SIZE` | 30 | Starting population. Shrinks naturally over time. |
| `PATIENCE` | 10 | Gens of struggle before death. Lower = harsher environment. |
| `RATCHET_MARGIN` | 0.20 | How far above ratchet a genome can survive. |
| `RATCHET_STEP` | 0.03 | How much ratchet advances each time. |
| `RATCHET_STABILITY` | 5 | Gens best must hold before ratchet advances. |
| `FREEZE_THRESH` | floor + 0.08 | Energy below which a genome crystallizes. |
| `IRREDUCIBLE_STD` | 0.18 | Noise floor of the environment. |

---

## What the Output Means

**Frozen genomes** are the primary output. Each one is a genome that:
- Evolved under real selection pressure
- Held a geometrically consistent shape against a real (noisy) environment
- Got close enough to the theoretical minimum energy to be crystallized

The frozen genome encodes — implicitly, in its weights — the statistical structure of the environment it survived in. It learned what_is by being penalized for getting it wrong. It learned the counterfactual relationship by being penalized for ignoring the deviation structure. It learned to project forward coherently by being penalized for geometric inconsistency.

It did not learn any of this from labels, rewards, or gradients. It learned because the environment demanded it, the ratchet raised the bar, and only the shapes reality actually allows survived long enough to freeze.

---

## Known Properties and Limits

**The gap is architecture-limited, not dynamics-limited.** A hidden-24 genome stalls around 0.20 energy. Hidden-64 gets within 0.001 of the noise floor. Increasing capacity closes the gap; the dynamics (ratchet, crossover, lifecycle) work correctly at any capacity.

**The mean population energy trends down when crossover bias is correctly calibrated.** If the mean plateaus while the best genome improves, crossover is too uniform — high-trust genomes aren't propagating their structure. Increase the adaptive exponent.

**The protected best genome is load-bearing.** Without it, the best configuration dies of age before it can propagate, and the population resets repeatedly. With it, the good structure anchors crossover for long enough to spread.

**No random injection means the system is honest.** If it goes extinct, the environment was too harsh or the population too small. Lower `PATIENCE` or increase `RATCHET_MARGIN` to give the population more room to survive ratchet transitions.

---

## Theoretical Grounding

This system is a practical implementation of the following claim:

> **The shape that survives is the shape reality allows.**

The environment is not a dataset. It is not a reward function. It is a geometric constraint on what configurations can exist stably. Evolution finds those configurations not by being told what they are, but by culling everything that can't hold the shape under pressure.

The three-signal framework (what_is / what_if / what_could_be) is the minimum geometry for knowledge that is temporally grounded. An anchor alone is just observation. An anchor plus counterfactual gives you causal structure. Add a coherent projection and you have a system that can reason about the future without losing contact with the past.

Distortion — in the pathological sense — occurs when the what_is anchor drifts. The rest of the shape can be internally coherent and still be completely wrong. The energy function's heaviest penalty on anchor fidelity is not arbitrary: it is the architectural acknowledgment that reality is the only fixed point.

---

## What to Build Next

1. **Expand the environment.** The current stream is synthetic. Real environments have regime changes, non-stationarity, and signals with genuine causal relationships. The ratchet threshold expansion mechanic — adding a new signal dimension when the current shape is mastered — is the natural next step.

2. **Multi-agent signal sources.** What_if could be generated by a second agent operating on the same what_is. What_could_be could be generated by a predictive module. The signals become dynamic rather than fixed.

3. **Freeze-and-stack.** Frozen genomes become the foundation for the next generation's environment. Stack frozen representations to build hierarchical understanding.

4. **Variable capacity evolution.** Let HIDDEN size itself be heritable and subject to selection. The system finds the minimum capacity required to hold the shape — no more, no less.

5. **Real environment integration.** Replace the synthetic stream with live sensor data, market data, or any time-series with genuine irreducible noise. The system should generalize — the theory doesn't depend on the specific signal domain.
