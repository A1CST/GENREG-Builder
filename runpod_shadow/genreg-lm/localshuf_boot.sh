#!/bin/bash
# Harden fork B: real-vs-LOCAL-shuffle at several swap strengths. Local
# (adjacent) swaps preserve absolute position but break local transitions, so
# the position-stats shortcut (which made the full scramble linearly easy)
# collapses and only relative-time detection can win. Sweep strengths to see
# the trend. Detached; sentinel.
cd /workspace/genreg-lm
rm -f localshuf.exit
{
  for n in 2 4 8 16; do
    echo "[localshuf] $(date -u) local swaps = $n"
    python -u radial_temporal_shuffle.py --cache wf_char_stream.pt --local $n
  done
  echo "[localshuf] $(date -u) done"
} > localshuf.log 2>&1
code=$?
echo "exit=${code}" > localshuf.exit
