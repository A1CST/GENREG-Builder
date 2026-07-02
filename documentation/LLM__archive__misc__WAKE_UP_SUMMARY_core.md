# Overnight Session Summary — A_66 → A_70

Autonomous iteration to push word-level gradient-free LM toward usability thresholds.
Goals: top-5 ≥ 60% (candidate generator), top-1 ≥ 30% (primary predictor), drop < 5%.

## Results Table

| Run | Heldout top-1 | Heldout top-5 | Drop | Notes |
|---|---|---|---|---|
| A_66 (baseline) | 0.1993 | 0.4015 | 0.2% | bg=0.357 trigram=0.214 4gram=0.247 neural=0.104 uni=0.078 |
| A_67 (energy gate) | 0.1990 | 0.4013 | 0.3% | bg=0.342 tri=0.550 neural=0.064 uni=0.043 |
| A_68 (residual bootstrap) | 0.1992 | 0.4037 | 0.6% | bg=0.267 tri=0.346 neural=0.354 uni=0.033 |
| A_69 (hash output) | 0.1945 | 0.3910 | 0.8% | bg=0.134 tri=0.449 resid=0.323 hash=0.068 uni=0.025 |
| A_70 (per-neuron acts) | 0.2001 | 0.4015 | 0.2% | bg=0.285 tri=0.294 resid=0.238 hash=0.144 uni=0.039 |

**Best top-5: A_68 (residual bootstrap) at 0.4037**

## Threshold check

| Use case | Target | Best achieved |
|---|---|---|
| Candidate generator | top-5 ≥ 60% | 40.4% ❌ |
| Primary predictor   | top-1 ≥ 30% | 19.9% ❌ |

## What ran

### A_66 (baseline)
- Trust: bg=0.357 trigram=0.214 4gram=0.247 neural=0.104 uni=0.078

### A_67 (energy gate)
- Wall time: 702s
- Trust: bg=0.342 tri=0.550 neural=0.064 uni=0.043

### A_68 (residual bootstrap)
- Wall time: 659s
- Trust: bg=0.267 tri=0.346 neural=0.354 uni=0.033

### A_69 (hash output)
- Wall time: 840s
- Trust: bg=0.134 tri=0.449 resid=0.323 hash=0.068 uni=0.025

### A_70 (per-neuron acts)
- Wall time: 841s
- Trust: bg=0.285 tri=0.294 resid=0.238 hash=0.144 uni=0.039

## How to use the best model

```bash
# Inspect trust mix and activations
python inspect_model.py A_68_best.pkl

# Predict next token for an arbitrary prefix
python predict_next.py A_68_best.pkl 'in the year'
```

## Files

- `LM_AUTONOMOUS_LOG.md` — detailed log with diagnoses per iteration
- `genreg_lm_A_{67,68,69,70}.py` — the four iteration scripts
- `run_lm_A_{67,68,69,70}.log` — full training traces
- `A_{67,68,69,70}_best.pkl` — saved best genome per iteration
- `inspect_model.py`, `predict_next.py` — inference tooling
