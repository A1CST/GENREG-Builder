"""Genome <-> environment glue: run one recurrent episode and score it.

A control/agent world in the engine is "a rollout that returns a scalar". This
module is that rollout, with memory (the recurrent state carries across steps)
and with the per-step / post-episode hooks that the checked constraints inject
(see ``constraints_map.Constraints``).
"""
from dataclasses import dataclass, field

import numpy as np

from engine_api import fresh_state, rstep

STEP_CAP = 2000        # hard bound on episode length so evaluation stays fast
MAX_FRAMES = 240       # replay frames streamed to the browser per episode


@dataclass
class RolloutResult:
    score: float                       # shaped score used for selection
    base_score: float                  # raw world-consequence (apples / tile score)
    frames: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _choose_action(env, out):
    """Highest-logit action among the env's currently valid actions.

    Snake always allows all three; 2048 masks to moves that change the board, so
    an empty valid set means the game is over.
    """
    valid = env.valid_actions()
    if not valid:
        return None
    return max(valid, key=lambda a: float(out[a]))


def rollout(genome, env, C=None, rng=None, record=False):
    """Drive `genome` through one episode of `env`. Returns a RolloutResult.

    `C` is a Constraints bundle (or None for the bare world). `rng` supplies any
    stochastic constraint effects (mortality, noise). `record=True` collects
    render_state() frames (downsampled to MAX_FRAMES) for the browser replay.
    """
    if rng is None:
        rng = np.random.default_rng()

    obs = env.reset()
    state = fresh_state(genome)
    frames = [env.render_state()] if record else None
    survival = C.survival.start(env) if (C and C.survival) else None

    while env.is_alive() and env.steps_taken() < STEP_CAP:
        obs_in = C.obs_transform(obs, rng) if (C and C.obs_transform) else obs
        out, state = rstep(genome, state, obs_in)
        if C and C.state_decay:
            state = state * (1.0 - C.state_decay)            # entropy: state leaks

        action = _choose_action(env, out)
        if action is None:                                   # no legal move -> game over
            break
        obs, reward, done, info = env.step(action)

        if survival is not None:
            survival, dead = C.survival.step(survival, env, reward, info, rng)
            if dead:
                done = True

        if record:
            frames.append(env.render_state())
        if done:
            break

    base = float(env.base_score())
    stats = {
        "score": int(getattr(env, "score", 0)),
        "steps": int(env.steps_taken()),
        "base": base,
    }
    if survival is not None:
        stats["energy_left"] = float(survival.get("energy", 0.0))

    score = C.post_score(base, stats) if (C and C.post_score) else base
    if not np.isfinite(score):
        score = -1e9                                         # guard: never let NaN/inf win
    if record and frames:
        frames = _downsample(frames, MAX_FRAMES)
    return RolloutResult(score=float(score), base_score=base, frames=frames or [], stats=stats)


def _downsample(frames, k):
    if len(frames) <= k:
        return frames
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    # keep last frame (the terminal state) explicitly
    out = [frames[i] for i in idx]
    if out[-1] is not frames[-1]:
        out[-1] = frames[-1]
    return out
