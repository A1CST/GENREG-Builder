"""Trainer: assemble the engine for a game world and run generations.

Builds a constraint-shaped, common-seed episodic fitness over Snake/2048 and an
Evolver with recurrence + precision (+ dimension when a size cost is present),
then steps generations, emitting one event per generation with:
  * telemetry (best/mean/median fitness)
  * the champion genome's weight matrices (for the Microscope)
  * a replay of the champion playing a FRESH (unseen) episode (for the board)
  * the PO count (checked constraints)

Fair evaluation: within a generation every genome is scored on the *same* set of
episode seeds, so selection compares genomes on identical challenges (far less
noise than independent draws). Non-stationarity resamples those seeds every K
generations — the world drifts. One Trainer runs one job; `stop()` is thread-safe.
"""
from dataclasses import dataclass, field

import numpy as np

from engine_api import (
    Genome, Evolver, chain, snapshot,
    enable, init_precision, mutate_recurrence, mutate_precision, mutate_dimensions,
    genome_layers, genome_summary,
)
from envs import SnakeEnv, Game2048Env
from agent import rollout, STEP_CAP
from constraints_map import build_constraints

EPISODES_PER_EVAL = 4          # median over N episodes (robust, noise-tolerant)
_SEED_MASK = 0x7FFFFFFF


@dataclass
class TrainConfig:
    environment: str = "snake"
    population: int = 100
    generations: int = 60
    constraints: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    snake: dict = field(default_factory=lambda: {"w": 20, "h": 15})
    hidden: int = 24
    seed: int = 0
    # evolution controls
    evolve_hidden: bool = False       # let H grow/shrink vs fixed
    sexual: bool = True               # crossover on (False = mutation-only / asexual)
    elite: int = 2                    # keep the best N unchanged each generation
    parent_frac: float = 0.25         # breed from the top fraction of the population
    device: str = "cpu"               # "cpu" (engine) | "gpu" (vectorized) | "auto"


def _clamp(v, lo, hi, default):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clampf(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def parse_config(d):
    d = d or {}
    env = d.get("environment", "snake")
    if env not in ("snake", "2048"):
        env = "snake"
    snake_in = d.get("snake") or {}
    params = dict(d.get("params") or {})
    pop = _clamp(d.get("population", 100), 4, 1000, 100)
    return TrainConfig(
        environment=env,
        population=pop,
        generations=_clamp(d.get("generations", 60), 1, 1000, 60),
        constraints=list(d.get("constraints") or []),
        params=params,
        snake={"w": _clamp(snake_in.get("w", 20), 5, 60, 20),
               "h": _clamp(snake_in.get("h", 15), 5, 60, 15)},
        hidden=_clamp(params.get("hidden", d.get("hidden", 24)), 4, 128, 24),
        seed=_clamp(d.get("seed", 0), 0, _SEED_MASK, 0),
        evolve_hidden=bool(d.get("evolve_hidden", False)),
        sexual=bool(d.get("sexual", True)),
        elite=_clamp(d.get("elite", 2), 0, max(0, pop - 1), 2),
        parent_frac=_clampf(d.get("parent_frac", 0.25), 0.02, 1.0, 0.25),
        device=(d.get("device") if d.get("device") in ("cpu", "gpu", "auto") else "cpu"),
    )


# vectorized (GPU) mode currently covers snake with no constraints; everything
# else runs on the CPU engine. `auto` uses the GPU only when it actually helps.
AUTO_VEC_MIN_POP = 2000


def create_trainer(cfg_dict, emit):
    """Pick the CPU engine trainer or the vectorized trainer for a config.

    The vectorized path covers snake/2048 with fixed H and the subset of
    constraints in vector_engine.VEC_SUPPORTED; anything else uses the CPU engine.
    """
    cfg = parse_config(cfg_dict)
    use_vec = False
    if cfg.device in ("gpu", "auto") and cfg.environment in ("snake", "2048") and not cfg.evolve_hidden:
        try:
            import vector_engine as ve
            constraints_ok = all(c in ve.VEC_SUPPORTED for c in cfg.constraints)
            if constraints_ok:
                if cfg.device == "gpu":
                    use_vec = True
                else:  # auto: only when the GPU actually helps
                    use_vec = ve.torch.cuda.is_available() and cfg.population >= AUTO_VEC_MIN_POP
        except Exception:
            use_vec = False
    if use_vec:
        from vector_trainer import VectorTrainer
        return VectorTrainer(cfg, emit)
    return Trainer(cfg, emit)


class Trainer:
    def __init__(self, config, emit):
        self.cfg = config if isinstance(config, TrainConfig) else parse_config(config)
        self.emit = emit
        self._stop = False
        self.C, self.params = build_constraints(self.cfg.constraints, self.cfg.params)

        master = np.random.default_rng(self.cfg.seed)
        # fixed evaluation seeds (resampled only under non-stationarity)
        self._eval_seeds = [int(master.integers(_SEED_MASK)) for _ in range(EPISODES_PER_EVAL)]
        self._seed_rng = np.random.default_rng(int(master.integers(_SEED_MASK)))   # for resampling
        self._replay_rng = np.random.default_rng(int(master.integers(_SEED_MASK)))  # fresh replays
        self._probe = self._make_env(0)         # shape probe (n_in/n_out only)
        self.ev = None
        self._last_best = None                  # champion from the last fully-evaluated generation

    # -- env construction (deterministic per seed) ------------------------
    def _make_env(self, seed):
        rng = np.random.default_rng(int(seed) & _SEED_MASK)
        if self.cfg.environment == "snake":
            mh = STEP_CAP if self.C.relax_hunger else None
            return SnakeEnv(self.cfg.snake["w"], self.cfg.snake["h"], rng=rng, max_hunger=mh)
        return Game2048Env(rng=rng)

    @property
    def n_in(self):
        return self._probe.N_IN

    @property
    def n_out(self):
        return self._probe.N_OUT

    # -- fitness: median over the shared episode seeds, minus engine costs --
    def _fitness(self, g):
        # Stop fast: once stopped, every remaining genome resolves instantly, so
        # the in-flight generation wraps up in ~one episode instead of grinding
        # through the whole population.
        if self._stop:
            return -1e9
        C = self.C
        scores = []
        for s in self._eval_seeds:
            if self._stop:
                break
            env = self._make_env(s)
            crng = np.random.default_rng((int(s) ^ 0x9E3779B9) & _SEED_MASK)
            try:
                scores.append(rollout(g, env, C, crng).score)
            except Exception:
                scores.append(-1e9)                          # a broken genome must not win
        if not scores:
            return -1e9
        base = float(np.median(scores))
        cost = sum(c(g) for c in C.engine_costs)
        val = base - cost
        return val if np.isfinite(val) else -1e9

    def _build_evolver(self):
        C = self.C
        H0 = self.cfg.hidden

        def make_genome(rng):
            g = Genome(self.n_in, self.n_out, H0, rng)
            enable(g, rng)            # recurrence: control worlds need memory
            init_precision(g)         # precision genes (exact no-op until a precision cost bites)
            return g

        hooks = [mutate_recurrence, mutate_precision]
        # H evolves when the user asks for it OR a size cost (memory-rent) needs it to bite
        if self.cfg.evolve_hidden or C.want_dimension:
            hooks.append(mutate_dimensions)

        # user population knobs are the base; constraint laws (scarcity /
        # reproduction-cost) override them when present.
        evo_kwargs = {
            "sexual": self.cfg.sexual,
            "elite": min(self.cfg.elite, max(0, self.cfg.population - 1)),
            "parent_frac": self.cfg.parent_frac,
        }
        evo_kwargs.update(C.evolver_kwargs)

        self.ev = Evolver(
            self.n_in, self.n_out, self._fitness,
            pop=self.cfg.population, H0=H0, seed=self.cfg.seed,
            make_genome=make_genome, mutate_hook=chain(*hooks),
            telemetry=snapshot, **evo_kwargs,
        )

    # -- run loop ---------------------------------------------------------
    def stop(self):
        self._stop = True

    def champion(self):
        """The best genome from the last fully-evaluated generation (for checkpointing)."""
        return self._last_best

    def run(self):
        try:
            self._build_evolver()
            self.emit({
                "type": "started",
                "environment": self.cfg.environment,
                "generations": self.cfg.generations,
                "population": self.cfg.population,
                "n_in": self.n_in, "n_out": self.n_out,
                "constraints": self.cfg.constraints,
                "po": len(self.cfg.constraints),
                "notes": self.C.notes,
            })

            for gen in range(self.cfg.generations):
                if self._stop:
                    break
                if self.C.shift_every and gen > 0 and gen % self.C.shift_every == 0:
                    self._eval_seeds = [int(self._seed_rng.integers(_SEED_MASK))
                                        for _ in range(EPISODES_PER_EVAL)]      # world drifts
                self.ev.step()
                if self._stop:            # stopped mid-generation: don't emit a half-evaluated gen
                    break
                self.emit(self._generation_event(gen))
                self._last_best = self.ev.best     # remember the last fully-evaluated champion

            best = self._last_best
            if best is None:
                summary = {}
            elif self._stop:
                summary = genome_summary(best)     # stopped: skip the extra replay episode
            else:
                summary = self._replay(best, record=False)[1]
            self.emit({
                "type": "done",
                "reason": "stopped" if self._stop else "finished",
                "gen": self.ev.gen if self.ev else 0,
                "best": summary,
            })
        except Exception as exc:                              # never take down the WS
            import traceback
            self.emit({"type": "error", "message": f"{exc}", "trace": traceback.format_exc()})

    # -- champion replay on a FRESH episode -------------------------------
    def _replay(self, genome, record):
        seed = int(self._replay_rng.integers(_SEED_MASK))
        env = self._make_env(seed)
        crng = np.random.default_rng((seed ^ 0x9E3779B9) & _SEED_MASK)
        r = rollout(genome, env, self.C, crng, record=record)
        summary = {
            "score": r.stats.get("score", 0),
            "steps": r.stats.get("steps", 0),
            "base": _round(r.base_score),
            **genome_summary(genome),
        }
        return r, summary

    def _generation_event(self, gen):
        best = self.ev.best
        rec = self.ev.history[-1] if self.ev.history else {}
        replay, summary = self._replay(best, record=True)
        return {
            "type": "generation",
            "gen": gen + 1,
            "generations": self.cfg.generations,
            "fitness": {
                "best": _round(rec.get("best")),
                "mean": _round(rec.get("mean")),
                "median": _round(rec.get("median")),
            },
            "best": summary,
            "po": len(self.cfg.constraints),
            "genome": {"layers": genome_layers(best)},
            "replay": {"env": self.cfg.environment, "frames": replay.frames, "meta": replay.stats},
        }


def _round(v, dp=4):
    if v is None:
        return None
    try:
        return round(float(v), dp)
    except (TypeError, ValueError):
        return None
