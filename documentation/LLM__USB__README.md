# GENREG Wiki LLM — Portable Package

A gradient-free language model built entirely through evolutionary optimization (GENREG).
No backpropagation was used at any stage.

## Quick Start

```bash
# Generate text
python inference.py --prompt "the king sat on the"

# Interactive mode
python inference.py --interactive

# Run all benchmark prompts
python inference.py --batch

# Full benchmark with metrics
python benchmark.py
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA recommended)
- NumPy

## Architecture

```
tokens → Embedding(PPMI-SVD) → PosEnc(sinusoidal) → CausalAttn×2 → Bayesian(n-gram×cosine) → text
```

| Component | Method | Params |
|-----------|--------|--------|
| Embedding | Evolved PPMI-SVD + skip-residual | 231K evolved + 6.6M shared |
| Position | Evolved sinusoidal + per-dim gain | 397K |
| Attention L0 | Evolved Q/K/V/O + per-head activations | 2.36M |
| Attention L1 | Auto-stacked, independently evolved | 2.36M |
| Generation | Bayesian: 5-gram prior × embedding cosine | n/a |

Total model: ~49 MB frozen checkpoints + 228 MB n-gram tables

## Files

- `inference.py` — Self-contained inference script
- `benchmark.py` — Standardized benchmark suite
- `genreg_encoder_gpu.py` — Evolved activation functions
- `checkpoints/` — All frozen model weights
