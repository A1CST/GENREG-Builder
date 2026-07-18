#!/bin/bash
# C6: cloze at 100k phrases. Detached (setsid) so a dropped SSH session
# cannot SIGHUP the run, and the exit code is always written to
# run_c6.exit - a cgroup OOM SIGKILLs python, so no in-process completion
# path can be trusted to report the ending (AGENTS.md rule 3).
cd /workspace/genreg-lm
rm -f run_c6.exit
python -u run_c6.py > run_c6.log 2>&1
code=$?
# cgroup OOM counter at exit: proves an OOM kill vs a clean/other death
oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)
echo "exit=${code} oom_kill=${oom} peak_bytes=${peak}" > run_c6.exit
