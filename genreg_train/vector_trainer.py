"""VectorTrainer: run the vectorized (GPU-capable) engine and emit the same
events as the CPU Trainer, so the browser (board / Microscope / HUD) is unchanged.

The population is evolved as batched tensors (vector_engine); each generation the
champion is pulled off the device into a plain CPU genome so the Microscope layers
and the board replay reuse the existing CPU path.

Current vectorized coverage: snake, no constraints, fixed H. Anything outside that
(2048, any constraint) is handled by the CPU Trainer — the factory decides.
"""
import numpy as np
import torch

from engine_api import Genome, genome_layers, genome_summary
from envs import SnakeEnv, Game2048Env
from agent import rollout, STEP_CAP
from constraints_map import build_constraints
import vector_engine as ve


def _cpu_genome(champ):
    """Build a genreg-engine Genome from extracted numpy weights (for replay/scope)."""
    g = Genome.__new__(Genome)
    g.n_in, g.H = int(champ["W1"].shape[0]), int(champ["W1"].shape[1])
    g.n_out = int(champ["W2"].shape[1])
    g.W1 = champ["W1"].astype(np.float32); g.b1 = champ["b1"].astype(np.float32)
    g.W2 = champ["W2"].astype(np.float32); g.b2 = champ["b2"].astype(np.float32)
    g.W_rec = champ["Wrec"].astype(np.float32); g.leak = float(champ["leak"])
    return g


class VectorTrainer:
    def __init__(self, cfg, emit):
        self.cfg = cfg
        self.emit = emit
        self._stop = False
        self.dev = ve.device_of(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        self.last_champ = None

    def stop(self):
        self._stop = True

    def champion(self):
        return _cpu_genome(self.last_champ) if self.last_champ is not None else None

    def _make_replay_env(self):
        if self.cfg.environment == "2048":
            return Game2048Env(rng=self.rng)
        mh = STEP_CAP if getattr(self, "C_cpu", None) and self.C_cpu.relax_hunger else None
        return SnakeEnv(self.cfg.snake["w"], self.cfg.snake["h"], self.rng, max_hunger=mh)

    def _evaluate(self, pop, gen):
        cfg = self.cfg
        if cfg.environment == "2048":
            return ve.evaluate_2048(pop, cfg.population, cfg.hidden, self.dev,
                                    episodes=4, base_seed=cfg.seed + gen, C=self.C)
        return ve.evaluate_snake(pop, cfg.population, cfg.snake["h"], cfg.snake["w"], cfg.hidden,
                                 self.dev, episodes=4, base_seed=cfg.seed + gen, C=self.C)

    def run(self):
        try:
            cfg = self.cfg
            Hd = cfg.hidden
            P = cfg.population
            n_par = max(2, int(P * cfg.parent_frac))
            elite = min(cfg.elite, max(0, P - 1))
            nin = ve.V2048.N_IN if cfg.environment == "2048" else ve.VSnake.N_IN
            nout = ve.V2048.N_OUT if cfg.environment == "2048" else ve.VSnake.N_OUT

            # vectorized constraints for the batched eval + matching CPU constraints for replay
            self.C = ve.build_vconstraints(cfg.constraints, cfg.params, self.dev) if cfg.constraints else None
            self.C_cpu, _ = build_constraints(cfg.constraints, cfg.params)

            self.emit({
                "type": "started", "environment": cfg.environment,
                "generations": cfg.generations, "population": P,
                "n_in": nin, "n_out": nout,
                "constraints": cfg.constraints, "po": len(cfg.constraints),
                "notes": [f"vectorized engine on {self.dev.type.upper()} (fixed H={Hd})"],
            })

            pop = ve.init_pop(P, nin, Hd, nout, self.dev, cfg.seed)

            for gen in range(cfg.generations):
                if self._stop:
                    break
                fit = self._evaluate(pop, gen)
                order = torch.argsort(fit, descending=True)
                self.last_champ = ve.champion_numpy(pop, int(order[0]))
                self.emit(self._gen_event(gen, fit, self.last_champ))

                parents = order[:n_par]
                pidx = parents[torch.randint(n_par, (P,), device=self.dev)]
                child = ve.gather_pop(pop, pidx)
                child = ve.mutate_pop(child, self.dev)
                elites = ve.gather_pop(pop, order[:elite])
                for k in child:
                    child[k][:elite] = elites[k]
                pop = child

            summary = {}
            if self.last_champ is not None:
                g = _cpu_genome(self.last_champ)
                r = rollout(g, self._make_replay_env(), self.C_cpu, self.rng)
                summary = {"score": r.stats.get("score", 0), "base": round(r.base_score, 4),
                           **genome_summary(g)}
            self.emit({"type": "done", "reason": "stopped" if self._stop else "finished",
                       "gen": self.cfg.generations, "best": summary})
        except Exception as exc:
            import traceback
            self.emit({"type": "error", "message": f"{exc}", "trace": traceback.format_exc()})

    def _gen_event(self, gen, fit, champ):
        g = _cpu_genome(champ)
        replay = rollout(g, self._make_replay_env(), self.C_cpu, self.rng, record=True)
        return {
            "type": "generation", "gen": gen + 1, "generations": self.cfg.generations,
            "fitness": {"best": round(float(fit.max()), 4),
                        "mean": round(float(fit.mean()), 4),
                        "median": round(float(fit.median()), 4)},
            "best": {"score": replay.stats.get("score", 0), "steps": replay.stats.get("steps", 0),
                     "base": round(replay.base_score, 4), **genome_summary(g)},
            "po": len(self.cfg.constraints),
            "genome": {"layers": genome_layers(g)},
            "replay": {"env": self.cfg.environment, "frames": replay.frames, "meta": replay.stats},
        }
