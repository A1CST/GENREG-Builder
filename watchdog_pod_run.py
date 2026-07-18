"""Watch a detached pod run and post an Agent-panel notice when it ends.

AGENTS.md rule 3 says every training run must alarm when it ends - finished
OR stopped/crashed. A pod run cannot do this itself: a cgroup OOM SIGKILLs
python (no in-process completion path survives it), and agent_notify writes
to the LOCAL agent_store/notices.jsonl, so a pod-side notice would land in
the pod's file and never reach the panel. So the watchdog runs here and
polls the pod's exit sentinel (written by the run's shell wrapper).

    python watchdog_pod_run.py --run c6 [--every 120]

Exits after posting. Safe to leave running; it only reads.
"""
import argparse
import subprocess
import time

import agent_board

POD = "root@216.243.220.170"
PORT = "19544"
KEY = "~/.ssh/id_ed25519"
WORK = "/workspace/genreg-lm"


def sh(cmd, timeout=60):
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-o",
            "ConnectTimeout=15", "-p", PORT, "-i", KEY, POD, cmd]
    try:
        p = subprocess.run(full, capture_output=True, text=True,
                           timeout=timeout)
        return p.stdout.strip()
    except Exception as e:                     # network blip: keep polling
        return f"__ERR__ {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="c6", help="run tag (e.g. c6)")
    ap.add_argument("--every", type=int, default=120, help="poll seconds")
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()

    tag = a.run
    exit_f = f"{WORK}/run_{tag}.exit"
    log_f = f"{WORK}/run_{tag}.log"
    t0 = time.time()
    misses = 0

    # Bracket the first char so the pattern cannot match THIS ssh command's
    # own shell: `pgrep -f run_c6` inside a bash -c whose cmdline contains
    # "run_c6" matches itself and always reports ALIVE, which silently made
    # the vanish-detection below dead code. "run_[c]6" matches the real
    # process but not the literal text in our own cmdline.
    # Match the WRAPPER too, not just the .py: a run can legitimately have no
    # python yet (e.g. run_ddecomp.sh waiting for the GPU), and that is not
    # death.
    pat = f"run_[{tag[0]}]{tag[1:]}"

    while time.time() - t0 < a.max_hours * 3600:
        out = sh(f"cat {exit_f} 2>/dev/null; echo '@@'; "
                 f"pgrep -f '{pat}' >/dev/null && echo ALIVE || echo DEAD")
        if out.startswith("__ERR__"):
            misses += 1                        # unreachable != finished
            time.sleep(a.every)
            continue
        misses = 0
        sentinel, _, alive = out.partition("@@")
        sentinel, alive = sentinel.strip(), alive.strip()

        if sentinel:                           # wrapper wrote the exit line
            tail = sh(f"tail -25 {log_f} 2>/dev/null")
            code = "?"
            for part in sentinel.split():
                if part.startswith("exit="):
                    code = part.split("=", 1)[1]
            ok = code == "0"
            agent_board.post(
                f"LM {tag.upper()} {'finished' if ok else 'ENDED BAD'} "
                f"({sentinel})",
                f"pod {POD}:{PORT}\n{sentinel}\n\n--- tail {log_f} ---\n{tail}",
                kind="run" if ok else "alert", source="claude")
            print(f"posted: {sentinel}")
            return

        if alive == "DEAD":                    # gone without a sentinel
            tail = sh(f"tail -25 {log_f} 2>/dev/null")
            agent_board.post(
                f"LM {tag.upper()} VANISHED - no exit sentinel",
                f"pod {POD}:{PORT}\nProcess is gone but the wrapper never "
                f"wrote run_{tag}.exit - the wrapper itself was killed "
                f"(container OOM kills the whole group, or the pod died).\n\n"
                f"--- tail {log_f} ---\n{tail}",
                kind="alert", source="claude")
            print("posted: vanished")
            return

        time.sleep(a.every)

    agent_board.post(
        f"LM {tag.upper()} watchdog timed out after {a.max_hours}h",
        f"Run may still be going on {POD}:{PORT}; watchdog stopped watching.",
        kind="alert", source="claude")


if __name__ == "__main__":
    main()
