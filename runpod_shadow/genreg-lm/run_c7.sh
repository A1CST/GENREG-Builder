#!/bin/bash
# C7: stage C cloze re-run with the GENOME-ONLY head on C6 100k data.
# Detached (setsid) so a dropped SSH cannot SIGHUP it; the exit code is
# always written to run_c7.exit because a cgroup OOM SIGKILLs python and no
# in-process completion path survives that (AGENTS.md rule 3).
cd /workspace/genreg-lm
rm -f run_c7.exit
python -u run_c7.py > run_c7.log 2>&1
code=$?
oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)
echo "exit=${code} oom_kill=${oom} peak_bytes=${peak}" > run_c7.exit
