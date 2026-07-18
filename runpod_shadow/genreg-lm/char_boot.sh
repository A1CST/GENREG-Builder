#!/bin/bash
# CHAR-level temporal experiment: build the 32-step char stream from the
# stage-A letter eye, then run temporal-vs-static on it. Detached; sentinel.
cd /workspace/genreg-lm
rm -f char_boot.exit radial_data/wf_char_stream.pt
{
  echo "[char-boot] $(date -u) build char stream (515 letter-eye genomes)"
  python -u build_char_stream.py                        || exit 2
  echo "[char-boot] $(date -u) run temporal experiment on the char stream"
  python -u radial_temporal.py --cache wf_char_stream.pt || exit 3
  echo "[char-boot] $(date -u) done"
} > char_boot.log 2>&1
code=$?
oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
echo "exit=${code} oom_kill=${oom}" > char_boot.exit
