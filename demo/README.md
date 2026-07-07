# WordPipe visual demo

A pygame window that shows **how the evolved language pipeline builds up, one
genome at a time** — both how each specialist *trains* (live fitness curves) and
how the generated text *transforms* as you stack the layers.

```powershell
cd $HOME\Documents\GENREG
pip install pygame          # first time only
python demo\demo.py
```

**First run** trains the four genomes (~4 min) while drawing their fitness curves,
then caches the champions to `demo/genomes.pkl`. Every run after that starts
instantly. (To force a retrain, delete `genomes.pkl`. To pre-build the cache
headless: `python demo\build_cache.py`.)

## What you see

Left — the **genome stack**. Each layer shows its status, a live fitness
sparkline as it evolves, and a toggle. Click a layer to turn it on/off.

Right — the **generated output**, regenerated instantly as you toggle layers, so
you watch the language appear capability by capability:

| stack | output |
|---|---|
| nothing | random letters — no words |
| + Vocabulary | real words, random order |
| + Order | words follow a grammatical class skeleton |
| + Selection | context-fit word choice (previous word), or **Bidirectional** (both neighbours) |
| + Boundary | real sentences — periods and capitals |

`SPACE` regenerates with a new seed. The **Selection** layer is three-state:
Off → Prev-word → Bidirectional.

## The genomes (all gradient-free, ~7K params total)

- **Order** — next grammatical class given the last few classes (dense log-prob).
- **Selection** — fill a class slot with the word that fits the previous word
  (fixed distributional features + a tiny bilinear head).
- **Bidirectional selection** — fits the previous word *and* the next class.
- **Boundary** — per-position P(sentence ends), from class + sentence position.

See `documentation/WORDPIPE_FINDINGS.md` for the full method and results.
