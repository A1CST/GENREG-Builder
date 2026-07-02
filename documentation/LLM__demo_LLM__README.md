# A_77 — Gradient-Free Char-Level LLM (Demo)

A complete next-character language model trained entirely via evolutionary
selection. No gradients, no backprop. ~7,700 evolved parameters.

## Stats

| Quality | |
|---|---|
| Heldout top-1 | **54.96%** |
| Heldout top-5 | **87.07%** |
| Train/heldout drop | 0.18% |

| Size | |
|---|---|
| Trainable parameters | 7,700 |
| Model file | 49 KB |
| N-gram cache | 9.7 MB |
| Peak RAM at inference | 48 MB |

| Speed (CPU, single-threaded Python + NumPy) | |
|---|---|
| Load time | 442 ms |
| Generation throughput | **~5,650 tokens/sec** |
| Forward-only throughput | ~6,840 tokens/sec |
| Per-token latency | **0.18 ms** |

(Incremental cascade state — each new token is O(H) work, not O(N·H).
Roughly 220× faster than a naive rescan-each-step implementation.)

| Training | |
|---|---|
| Vocab | 32 chars (space + lowercase + `'`) |
| Generations | 2,500 |
| Population | 300 |
| Wall time | 28 min on one RTX 6000 |

Run `python benchmark.py` to reproduce on your machine.

## Contents of this folder

| File | Size | Purpose |
|---|---|---|
| `A_77_best.pkl` | 49 KB | The evolved model (weights + config + vocab) |
| `A_78_best.pkl` | 49 KB | Variant with multi-scale cascade (tied quality) |
| `_ngram_cache_V32_char.pkl` | 10 MB | Precomputed sparse n-gram tables (tri/4/5-gram) |
| `generate.py` | 21 KB | Autoregressive generator (sampling + beam + nucleus + rep/freq penalties) |
| `tokenizer.py` | 2 KB | Standalone char tokenizer (reads vocab from pkl) |
| `benchmark.py` | 3 KB | CPU benchmark (load + throughput + memory) |
| `ensemble.py` | 3 KB | Average logprobs across multiple models |
| `SAMPLES.md` | 4 KB | Gallery of generated text |
| `README.md` | this | |

The n-gram cache is derived from WikiText-103 training split and shipped
precomputed so you don't need the corpus to run the demo.

## Requirements

- Python 3.8+
- NumPy

That's it. CPU inference. No GPU, no PyTorch, no model server.

```bash
pip install numpy
```

## Usage

### Generate text

```bash
python generate.py A_77_best.pkl "the king and the queen " --tokens 250 --temp 0.7
```

Options:
- `--tokens N` (default 80): how many chars to generate
- `--temp T` (default 0.8): sampling temperature (lower = more deterministic)
- `--top_k K`: restrict sampling to top-K candidates
- `--top_p P`: nucleus sampling — keep smallest set covering P prob mass
- `--rep_penalty X` (>1): divide logits of tokens seen in `--rep_window` (default 32)
- `--freq_penalty X` (>0): subtract X × recency-count from logits
- `--greedy`: argmax each step (collapses to "the the the")
- `--seed N`: RNG seed (default 42)

**Coherence tips:** the best generation settings combine temperature with
repetition and nucleus sampling:
```bash
python generate.py A_77_best.pkl "the king and the queen " \
    --tokens 300 --temp 0.8 --top_p 0.9 --rep_penalty 1.3 --rep_window 20
```

### Tokenize independently

```bash
python tokenizer.py
```

Or from Python:

```python
from tokenizer import Tokenizer
tok = Tokenizer()
ids = tok.encode("Hello, world!")   # char indices, lowercase, drops non-vocab
text = tok.decode(ids)              # inverse
```

## Sample outputs

### Basic temperature sampling
Prefix: `"the king and the queen "` — temp 0.7, seed 42:
> the king and the queen comparally with a production rabbit the shortly
> deservice at the darth are and millian s decided to be built by the for
> and the traveling her with a third decided the game automobilitary

### With nucleus + repetition penalty (recommended)
Prefix: `"the king and the queen "` — temp 0.8, top_p 0.9, rep_penalty 1.3:
> the king and the queen comparis study in communited some of the
> previewer the first signed to resenterchase the defension by the later
> and more s seriences and she characters and around the game act is
> differents were asses both the minute the labout on the game french
> relation peak form a real life when the situal proble

### Top-k with frequency penalty
Prefix: `"in the year "` — temp 0.7, top_k 8, freq_penalty 0.3:
> in the year of the placed oncers and indicated to be proposed as and
> spaces inted into howeverally in the surfaced of the play moore
> supportedly after which was beforem in the million of the audied to
> and stated

Real English phrases emerge ("decided to be built by", "a real life when
the situal problem", "the surfaced of the play"). This is a 7,700-param
gradient-free model — no transformer, no backprop.

## How the model works

1. **Tokenize**: chars → 32-way integer ids
2. **Embed**: each id → 24-D SVD embedding of the bigram transition matrix
3. **Encode**: 24-D → 64-D hidden via evolved weights + per-neuron activation
   (8-function catalog: tanh, resonance, abs_gate, etc.)
4. **Cascade**: protein-inspired decay/momentum/integral state over sequence
5. **Produce 6 candidate distributions**:
   - bigram (lookup)
   - trigram (sparse lookup, 702 entries, 100% coverage)
   - 4-gram (sparse lookup, 9 K entries, 100% coverage)
   - 5-gram (sparse lookup, 51 K entries, 99.7% coverage)
   - neural residual (bigram + tanh(U @ V @ ctx))
   - hash channel (64-bit signature matching)
6. **Mix** via 7-channel trust (evolved per-input: bg+tri+4g+5g+resid+hash+uniform)
7. **Sample** next char

All weights are evolved via gradient-free tournament selection over 300
genomes for 2500 generations.

## Architecture details

- Vocab: V=32
- Embedding: D=24 (from SVD of bigram transition matrix)
- Hidden: H=64 with per-neuron activation
- Residual rank: R=16
- Hash bits: K=64
- Cascade: 3-way state (last, momentum, integral) with per-neuron evolved decay
- Trust mixer: 7 candidate channels → softmax gate

## Limitations

- Coherence is phrase-level, not sentence-level. You'll get real English
  words and short phrases but not meaningful paragraphs.
- Greedy decoding collapses to `"the state the state the state ..."`
  attractors. Always sample with temperature.
- Vocab is 32 chars — no uppercase, digits, or most punctuation.
- The n-gram cache (10 MB) does most of the heavy lifting at inference;
  the evolved params (7.7K) are the learned re-weighting on top.

## Interactive REPL

```bash
python repl.py               # loads A_77_best.pkl
python repl.py --model A_78_best.pkl
```

In the REPL:
- Type text to generate a continuation
- `/temp 0.7` — change sampling temperature
- `/top_p 0.9` — nucleus sampling
- `/rep 1.3` — repetition penalty
- `/tokens 300` — generation length
- `/model A_78_best.pkl` — switch model
- `/quit` — exit

## Ensemble

Average log-probs from multiple models:

```bash
python ensemble.py --models A_77_best.pkl A_78_best.pkl --prefix "the king" --tokens 200
```

## Benchmark

```bash
python benchmark.py --tokens 1000
```

Reports load time, tokens/sec (generation + forward-only), memory, and
sample output.
