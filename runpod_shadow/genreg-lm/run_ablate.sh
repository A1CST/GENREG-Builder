#!/bin/bash
# EAR ABLATION: ears on-off, genome-only head, cached features.
# Waits for C7 to release the GPU, then runs. Detached (setsid) so a dropped
# SSH cannot SIGHUP it; the exit code is always written to run_ablate.exit
# because a cgroup OOM SIGKILLs python and no in-process path survives that.
cd /workspace/genreg-lm
rm -f run_ablate.exit

# chain behind C7 rather than contend for the one GPU
waited=0
while pgrep -f 'run_ablate_none.py' > /dev/null 2>&1; do
  sleep 30
  waited=$((waited + 30))
  if [ $waited -gt 7200 ]; then
    echo "gave up waiting for C7 after 2h" > run_ablate.exit
    exit 1
  fi
done
echo "C7 released the GPU after ${waited}s wait; starting decomposed D" \
  > run_ablate.log

python -u run_ablate.py >> run_ablate.log 2>&1
code=$?
oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)
echo "exit=${code} oom_kill=${oom} peak_bytes=${peak}" > run_ablate.exit
