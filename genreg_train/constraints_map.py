"""Turn the GUI's checked constraints into real effects on the world.

Faithful to the EEC axis catalogue (project/EEC-main/docs/CONSTRAINTS.md): each
constraint is a *cost or law imposed on existence*, never reward shaping toward a
target. A constraint touches exactly one of five integration points:

  obs_transform   perturb what the organism sees   (occlusion, noise)
  state_decay     leak the recurrent memory        (entropy)
  survival        a per-step life/death budget      (energy, mortality, perception-cost)
  post_score      reshape the world-consequence     (time, perception-cost fallback)
  engine_costs    parsimony costs on the network    (efficiency, memory-rent)
  evolver_kwargs  turnover / selection pressure     (scarcity, reproduction-cost)
  shift_every     drift the world over generations  (non-stationarity)

Same-axis instantiations are handled per EEC ("swap, don't stack"): energy vs
mortality (survival), occlusion vs noise (observation).
"""
from dataclasses import dataclass, field

import numpy as np

from engine_api import weight_cost, size_cost

DEFAULTS = {
    "energy_budget": 120.0,
    "food_energy": 60.0,          # snake: energy restored per apple eaten
    "merge_energy": 1.0,          # 2048: multiplier on merge-restored energy (1.0 = 2+2 gives 2)
    "step_cost": 0.01,            # energy drained per step (both snake & 2048)
    "hazard": 0.02,
    "time_budget": 400.0,
    "cost_strength": 0.01,
    "decay": 0.15,
    "occlusion_p": 0.25,
    "noise": 0.15,
    "perception_cost": 1.0,
    "shift_every": 15,
}


@dataclass
class Constraints:
    obs_transform: object = None          # (obs, rng) -> obs
    state_decay: float = 0.0              # entropy: multiply recurrent state by (1 - decay)
    survival: object = None               # Survival instance or None
    post_score: object = None             # (base, stats) -> score
    engine_costs: list = field(default_factory=list)
    evolver_kwargs: dict = field(default_factory=dict)
    want_dimension: bool = False          # memory-rent needs H to be able to change
    relax_hunger: bool = False            # energy budget replaces the env's built-in starvation
    shift_every: int = None               # non-stationarity: reseed world every K gens
    names: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class Survival:
    """A per-step life/death budget. mode 'energy' (additive) or 'mortality' (hazard)."""

    def __init__(self, mode, energy_budget=0.0, food_energy=0.0, hazard=0.0,
                 step_cost=1.0, per_step_extra=0.0, merge_mult=1.0):
        self.mode = mode
        self.energy_budget = float(energy_budget)
        self.food_energy = float(food_energy)         # snake: energy per apple
        self.merge_mult = float(merge_mult)           # 2048: multiplier on merge-restored energy
        self.hazard = float(hazard)
        self.step_cost = float(step_cost)             # energy drained per step
        self.per_step_extra = float(per_step_extra)   # perception-cost drains extra energy/step

    def start(self, env):
        return {"energy": self.energy_budget}

    def step(self, st, env, reward, info, rng):
        """Return (state, dead). Energy restore is environment-specific."""
        if self.mode == "energy":
            st["energy"] -= (self.step_cost + self.per_step_extra)
            if getattr(env, "NAME", "") == "2048":
                # a move's gain = sum(2v) over merges; energy restored = sum(v) = gain/2
                # so 2+2 restores 2, 4+4 restores 4, ... (× merge_mult)
                if reward > 0:
                    st["energy"] += (reward / 2.0) * self.merge_mult
            else:
                # snake: eating an apple restores a flat food_energy
                if reward >= 0.5:
                    st["energy"] += self.food_energy
            return st, st["energy"] <= 0.0
        if self.mode == "mortality":
            mh = getattr(env, "max_hunger", 0) or 0
            frac = (getattr(env, "hunger", 0) / mh) if mh else 0.0
            return st, bool(rng.random() < self.hazard * frac)
        return st, False


def _make_obs_transform(occ_p, noise):
    def transform(obs, rng):
        if occ_p and rng.random() < occ_p:
            return np.zeros_like(obs)                       # sensory blackout
        if noise:
            return (obs + rng.normal(0.0, noise, obs.shape)).astype(np.float32)
        return obs
    return transform


def _make_post_score(time_budget, perception_penalty):
    def post(base, stats):
        s = base
        if time_budget:
            s = s / (1.0 + stats["steps"] / time_budget)   # Occam: fewer steps favored
        if perception_penalty:
            s = s - perception_penalty * stats["steps"]
        return s
    return post


def build_constraints(names, params=None):
    """names: iterable of checkbox values; params: overrides for DEFAULTS.

    Returns (Constraints, resolved_params).
    """
    p = {**DEFAULTS, **(params or {})}
    names = list(names or [])
    have = set(names)
    C = Constraints(names=names)

    # --- survival axis (energy | mortality) — swap, don't stack ---
    if "energy" in have:
        extra = p["perception_cost"] if "perception-cost" in have else 0.0
        C.survival = Survival("energy", p["energy_budget"], p["food_energy"],
                              step_cost=p["step_cost"], per_step_extra=extra,
                              merge_mult=p["merge_energy"])
        C.relax_hunger = True
        if "mortality" in have:
            C.notes.append("energy & mortality are the same (survival) axis — using energy; swap don't stack.")
    elif "mortality" in have:
        C.survival = Survival("mortality", hazard=p["hazard"])

    # perception-cost without an energy budget to drain -> a small step penalty
    perception_penalty = 0.0
    if "perception-cost" in have and "energy" not in have:
        perception_penalty = p["perception_cost"] / 1000.0
        C.notes.append("perception-cost applied as a per-step penalty (no energy budget present).")

    # --- observation axis (occlusion | noise) ---
    occ = p["occlusion_p"] if "occlusion" in have else 0.0
    noise = p["noise"] if "noise" in have else 0.0
    if occ and noise:
        C.notes.append("occlusion & noise are the same (observation) axis — both applied (redundant).")
    if occ or noise:
        C.obs_transform = _make_obs_transform(occ, noise)

    # --- active maintenance ---
    if "entropy" in have:
        C.state_decay = float(p["decay"])

    # --- parsimony (engine costs) ---
    if "efficiency" in have:
        C.engine_costs.append(weight_cost(p["cost_strength"]))
    if "memory-rent" in have:
        C.engine_costs.append(size_cost(p["cost_strength"]))
        C.want_dimension = True

    # --- time / perception post-score ---
    time_budget = p["time_budget"] if "time" in have else 0.0
    if time_budget or perception_penalty:
        C.post_score = _make_post_score(time_budget, perception_penalty)

    # --- selection / turnover axes ---
    if "scarcity" in have:
        C.evolver_kwargs["elite"] = 1
        C.evolver_kwargs["parent_frac"] = 0.5
        C.notes.append("scarcity: steady-state turnover (elite=1, parent_frac=0.5).")
    if "reproduction-cost" in have:
        C.evolver_kwargs["parent_frac"] = min(C.evolver_kwargs.get("parent_frac", 0.25), 0.15)
        C.notes.append("reproduction-cost: only the top ~15% reproduce.")

    # --- plasticity ---
    if "non-stationarity" in have:
        C.shift_every = int(p["shift_every"])

    return C, p
