#!/bin/bash
# The GRAMMAR discriminator: build a word-level stream (words as evolved
# embeddings), then real-vs-LOCAL-shuffle at several strengths. Local word
# swaps break grammatical transitions while preserving absolute position, so
# only relative-time (grammar) composition can win. Detached; sentinel.
cd /workspace/genreg-lm
rm -f grammar.exit radial_data/wf_word_stream.pt
{
  echo "[grammar] $(date -u) build word stream (embed_rs word vectors)"
  python -u build_word_stream.py                                         || exit 2
  for n in 1 2 4; do
    echo "[grammar] $(date -u) real-vs-LOCAL-shuffle, word swaps = $n"
    python -u radial_temporal_shuffle.py --cache wf_word_stream.pt --local $n || exit 3
  done
  echo "[grammar] $(date -u) also FULL scramble (for reference)"
  python -u radial_temporal_shuffle.py --cache wf_word_stream.pt          || exit 4
  echo "[grammar] $(date -u) done"
} > grammar.log 2>&1
code=$?
echo "exit=${code}" > grammar.exit
