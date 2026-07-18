#!/bin/bash
# Bootstrap the temporal experiment on a fresh pod: regenerate the data the
# dead pod took with it, build the word-feature cache (the eye sweep), then
# run the temporal->static experiment. Detached; sentinel on every exit.
cd /workspace/genreg-lm
rm -f temporal_boot.exit
# CRITICAL: remove the synthetic smoke cache or stage_c will load it instead
# of building the real one (it only checks existence).
rm -f radial_data/wf_kid_next_g8_A515_B297_N50000x10000.pt radial_data/kid_next.npz
{
  echo "[boot] $(date -u) regen kid_words.npz (deterministic vocab)"
  python -c "import radial_kid; radial_kid.make_words_b()"                || exit 2
  echo "[boot] $(date -u) build kid_next.npz (50k/10k)"
  python -c "import radial_kid; radial_kid.make_next_d(50000,10000)"      || exit 3
  echo "[boot] $(date -u) build word-feature cache via minimal stage_d (eye sweep ~21min)"
  python -c "import radial_kid; radial_kid.stage_d(ears=False, max_rounds=1, max_spaces=1)" || exit 4
  echo "[boot] $(date -u) run temporal experiment (full)"
  python -u radial_temporal.py                                           || exit 5
  echo "[boot] $(date -u) done"
} > temporal_boot.log 2>&1
code=$?
oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
echo "exit=${code} oom_kill=${oom}" > temporal_boot.exit
