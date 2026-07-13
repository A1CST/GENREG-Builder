# evolang_v1 — archived 2026-07-09

The WordPipe gradient-free specialist-genome language pipeline (`/evolang`),
archived after its fluency ceiling stopped moving across every architecture
variant tried (forward, meaning-first, intent-first/backward, crystallized,
clause-obligation-tracked). Full findings, every genome's verdict, every
number, and rebuild recommendations: `documentation/WORDPIPE_FIELD_NOTES.pdf`
(generator: `documentation/WORDPIPE_FIELD_NOTES.py`).

## What's here

- `genreg_train/` — every genome module, the GA training engine
  (`wordpipe.py`, `genelib.py`, `evolang.py` — these three are tightly
  coupled to each other and to this pipeline specifically, not standalone
  infra), and every `run_*.py` experiment driver.
- `templates/`, `static/` — the `/evolang` and `/evolang/layers` pages.
- `demo/` — the pygame visual demo (`demo.py`) and the trained genome
  artifacts (`genomes.pkl`, `genomes_wiki.pkl`, `genomes_archaic_backup.pkl`,
  `chunks_wiki.pkl`) from each corpus generation.
- `corpora_artifacts/` — the combined-corpus (Wikipedia + Cornell Movie
  Dialogs) trained genome artifacts (`combined_genomes.pkl` and siblings).

## What was deliberately left in place (NOT archived)

Per explicit instruction — these are datasets/reusable infra, not pipeline
code:
- `corpora/wikipedia/wiki_corpus.txt`, `corpora/wikipedia/wiki_feats.npz`
- `corpora/combined/combined_corpus.txt`,
  `corpora/combined/build_combined_corpus.py`
- `project/conversational/cornell movie-dialogs corpus/`
- The I2 job-dispatch system (`i2_node.py`, `push_to_primary.py`)

## Known dangling references (harmless, not fixed)

A handful of one-off historical build scripts still import the moved
modules and will fail if re-run as-is: `genreg_train/eec_memory.py`,
`corpora/wikipedia/build/semantic_batch2.py`,
`corpora/wikipedia/build/register_build.py`,
`corpora/wikipedia/build/export_champions.py`. These already completed
their jobs (they produced artifacts that still exist); they were not
re-pointed at the archive because they're not part of any active code path.
