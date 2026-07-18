#!/bin/bash
# Fork A (recency-preserving temporal) + Fork B (real-vs-shuffled structural
# task), both on the already-built char stream. Detached; sentinel.
cd /workspace/genreg-lm
rm -f bothforks.exit
{
  echo "[forks] $(date -u) FORK A: recency-preserving temporal (char stream)"
  python -u radial_temporal.py --cache wf_char_stream.pt --recency
  cp radial_data/temporal_result_char.json radial_data/temporal_result_char_recency.json 2>/dev/null
  echo "[forks] $(date -u) FORK B: real-vs-shuffled structural task (char stream)"
  python -u radial_temporal_shuffle.py --cache wf_char_stream.pt
  echo "[forks] $(date -u) done"
} > bothforks.log 2>&1
code=$?
echo "exit=${code}" > bothforks.exit
