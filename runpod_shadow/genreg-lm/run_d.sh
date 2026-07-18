#!/bin/bash
# Stage D: autoregression (next word) at 50k contexts, warm-started on C.
# Detached (setsid) so a dropped SSH cannot SIGHUP it; the exit code is
# always written to run_d.exit because a cgroup OOM SIGKILLs python and no
# in-process completion path survives that (AGENTS.md rule 3).
cd /workspace/genreg-lm
rm -f run_d.exit
python -u run_d.py > run_d.log 2>&1
code=$?
oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)
echo "exit=${code} oom_kill=${oom} peak_bytes=${peak}" > run_d.exit
