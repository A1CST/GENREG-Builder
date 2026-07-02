"""Headless training driver — the primary automated verification.

Trains a genome to play snake or 2048 without the browser and prints the fitness
curve alongside a random-policy baseline, so we can confirm the engine actually
learns the game (and that constraints don't break it).

    python run_headless.py snake --gens 60 --pop 120
    python run_headless.py snake --gens 60 --constraints energy,time
    python run_headless.py 2048  --gens 60
"""
import argparse
import time

import numpy as np

from trainer import Trainer, parse_config
from envs import SnakeEnv, Game2048Env


def random_baseline(env_name, snake, n=200, seed=123):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        env = SnakeEnv(snake["w"], snake["h"], rng) if env_name == "snake" else Game2048Env(rng)
        env.reset()
        while env.is_alive() and env.steps_taken() < 2000:
            va = env.valid_actions()
            if not va:
                break
            env.step(va[int(rng.integers(len(va)))])
        scores.append(env.base_score())
    return float(np.mean(scores)), float(np.max(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("environment", choices=["snake", "2048"])
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--constraints", default="", help="comma-separated constraint values")
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cons = [c.strip() for c in args.constraints.split(",") if c.strip()]
    cfg = parse_config(dict(
        environment=args.environment, population=args.pop, generations=args.gens,
        constraints=cons, hidden=args.hidden, seed=args.seed,
    ))

    base_mean, base_max = random_baseline(args.environment, cfg.snake)
    print(f"random baseline: mean base-score {base_mean:.3f}, max {base_max:.3f}")
    print(f"config: env={cfg.environment} pop={cfg.population} gens={cfg.generations} "
          f"constraints={cons or 'none'} hidden={cfg.hidden}")

    first = {"v": None}
    step = max(1, args.gens // 12)

    def emit(ev):
        t = ev.get("type")
        if t == "started":
            if ev.get("notes"):
                for n in ev["notes"]:
                    print("  note:", n)
        elif t == "generation":
            g, f, b = ev["gen"], ev["fitness"], ev["best"]
            if first["v"] is None:
                first["v"] = b["base"]
            if g == 1 or g % step == 0 or g == args.gens:
                print(f"  gen {g:3d}/{args.gens}  best {f['best']:.3f}  mean {f['mean']:.3f}  "
                      f"champ score {b['score']} base {b['base']:.2f}  H{b['H']} leak {b.get('leak')}")
        elif t == "done":
            print("done:", ev["reason"], "final champ:", ev["best"])
        elif t == "error":
            print("ERROR:", ev["message"])
            print(ev.get("trace", ""))

    t0 = time.time()
    Trainer(cfg, emit).run()
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
