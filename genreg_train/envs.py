"""Game environments for training genomes: Snake and 2048.

Pure game logic, no engine imports. Each env exposes the same small interface so
``agent.rollout`` can drive any of them:

    env.reset() -> obs (np.float32 vector, length env.N_IN)
    env.step(action) -> (obs, reward, done, info)
    env.observe() -> obs
    env.render_state() -> JSON-serialisable dict for the browser canvas
    env.valid_actions() -> list[int]           (all actions, unless masked like 2048)
    class attrs: NAME, N_IN, N_OUT

Rewards are *world consequences* (apples, merged tile value), never a designed
gradient toward a target — see EEC docs.
"""
import numpy as np

# ---------------------------------------------------------------- Snake
# Directions indexed 0..3 = up, right, down, left (y grows downward).
_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class SnakeEnv:
    NAME = "snake"
    N_IN = 11
    N_OUT = 3                    # relative: 0=turn-left, 1=straight, 2=turn-right
    APPLE = 1.0
    STEP_REWARD = 0.01

    def __init__(self, w=20, h=15, rng=None, max_hunger=None):
        self.w = int(max(5, min(60, w)))
        self.h = int(max(5, min(60, h)))
        self.rng = rng if rng is not None else np.random.default_rng()
        # implicit metabolism so episodes are finite even with no energy constraint
        self.max_hunger = int(max_hunger) if max_hunger else 2 * (self.w + self.h)
        self.reset()

    def reset(self):
        cx, cy = self.w // 2, self.h // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]   # head first
        self.dir = 1                                          # moving right
        self.alive = True
        self.score = 0
        self.steps = 0
        self.hunger = 0
        self._place_food()
        return self.observe()

    def _place_food(self):
        occupied = set(self.snake)
        empty = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in occupied]
        self.food = empty[int(self.rng.integers(len(empty)))] if empty else None

    def valid_actions(self):
        return [0, 1, 2]

    def step(self, action):
        if not self.alive:
            return self.observe(), 0.0, True, {"cause": "dead"}
        if action == 0:
            self.dir = (self.dir - 1) % 4
        elif action == 2:
            self.dir = (self.dir + 1) % 4
        # action 1 = straight

        dx, dy = _DIRS[self.dir]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy
        self.steps += 1
        self.hunger += 1

        # wall collision
        if nx < 0 or nx >= self.w or ny < 0 or ny >= self.h:
            self.alive = False
            return self.observe(), 0.0, True, {"cause": "wall"}

        eating = self.food is not None and (nx, ny) == self.food
        # body collision (the tail cell is free next step unless we grow)
        occupied = set(self.snake) if eating else set(self.snake[:-1])
        if (nx, ny) in occupied:
            self.alive = False
            return self.observe(), 0.0, True, {"cause": "self"}

        self.snake.insert(0, (nx, ny))
        if eating:
            self.score += 1
            self.hunger = 0
            reward = self.APPLE
            self._place_food()
            if self.food is None:                            # board full = win
                self.alive = False
                return self.observe(), reward, True, {"cause": "win"}
        else:
            self.snake.pop()
            reward = self.STEP_REWARD

        if self.hunger > self.max_hunger:
            self.alive = False
            return self.observe(), reward, True, {"cause": "starved"}
        return self.observe(), reward, False, {"cause": None}

    def observe(self):
        hx, hy = self.snake[0]
        d = self.dir
        body = set(self.snake[:-1])                           # tail vacates next step

        def blocked(dir_idx):
            dx, dy = _DIRS[dir_idx]
            x, y = hx + dx, hy + dy
            return 1.0 if (x < 0 or x >= self.w or y < 0 or y >= self.h or (x, y) in body) else 0.0

        danger = [blocked(d), blocked((d - 1) % 4), blocked((d + 1) % 4)]  # straight, left, right
        dir_onehot = [1.0 if d == i else 0.0 for i in range(4)]           # up, right, down, left
        fx, fy = self.food if self.food else (hx, hy)
        food = [
            1.0 if fy < hy else 0.0,   # up
            1.0 if fx > hx else 0.0,   # right
            1.0 if fy > hy else 0.0,   # down
            1.0 if fx < hx else 0.0,   # left
        ]
        return np.array(danger + dir_onehot + food, np.float32)

    def render_state(self):
        return {
            "env": "snake", "w": self.w, "h": self.h,
            "snake": [[int(x), int(y)] for (x, y) in self.snake],
            "food": [int(self.food[0]), int(self.food[1])] if self.food else None,
            "score": int(self.score), "steps": int(self.steps), "alive": bool(self.alive),
        }

    def is_alive(self):
        return self.alive

    def steps_taken(self):
        return self.steps

    def base_score(self):
        # world consequence: apples plus a little for staying alive
        return self.score * self.APPLE + self.steps * self.STEP_REWARD


# ---------------------------------------------------------------- 2048
class Game2048Env:
    NAME = "2048"
    SIZE = 4
    N_IN = 16
    N_OUT = 4                    # 0=up, 1=right, 2=down, 3=left

    def __init__(self, rng=None, **_):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.grid = [[0] * self.SIZE for _ in range(self.SIZE)]
        self.score = 0
        self.moves = 0
        self.over = False
        self._spawn()
        self._spawn()
        return self.observe()

    def _spawn(self):
        empty = [(r, c) for r in range(self.SIZE) for c in range(self.SIZE) if self.grid[r][c] == 0]
        if not empty:
            return False
        r, c = empty[int(self.rng.integers(len(empty)))]
        self.grid[r][c] = 2 if self.rng.random() < 0.9 else 4
        return True

    _SLIDE_CACHE = {}

    @classmethod
    def _slide_row(cls, row):
        """Slide a single row toward index 0, merging equal neighbours once.

        Memoized on the row contents (the inner loop dominates 2048 rollouts).
        Returns a *fresh* list each call so the caller may store/mutate it (the
        grid gets mutated on spawn) without corrupting the cache.
        """
        key = tuple(row)
        cached = cls._SLIDE_CACHE.get(key)
        if cached is None:
            vals = [v for v in row if v != 0]
            out, gain, i = [], 0, 0
            while i < len(vals):
                if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                    merged = vals[i] * 2
                    out.append(merged)
                    gain += merged
                    i += 2
                else:
                    out.append(vals[i])
                    i += 1
            out += [0] * (len(row) - len(out))
            cached = (tuple(out), gain)
            cls._SLIDE_CACHE[key] = cached
        return list(cached[0]), cached[1]

    def _apply(self, grid, action):
        """Return (new_grid, gain, changed) for an action without mutating `grid`."""
        if action == 3:      # left
            g = [r[:] for r in grid]
        elif action == 1:    # right
            g = [r[::-1] for r in grid]
        elif action == 0:    # up
            g = [list(c) for c in zip(*grid)]
        elif action == 2:    # down
            g = [list(c)[::-1] for c in zip(*grid)]
        else:
            raise ValueError(f"bad action {action}")

        slid, gain = [], 0
        for row in g:
            nr, ga = self._slide_row(row)
            slid.append(nr)
            gain += ga

        if action == 3:
            res = slid
        elif action == 1:
            res = [r[::-1] for r in slid]
        elif action == 0:
            res = [list(c) for c in zip(*slid)]
        else:                # down
            res = [list(c) for c in zip(*[r[::-1] for r in slid])]
        return res, gain, (res != grid)

    def valid_actions(self):
        return [a for a in range(4) if self._apply(self.grid, a)[2]]

    def step(self, action):
        if self.over:
            return self.observe(), 0.0, True, {"cause": "over"}
        res, gain, changed = self._apply(self.grid, action)
        if not changed:
            # invalid move: no state change, no reward; over if nothing is valid
            self.over = not self.valid_actions()
            return self.observe(), 0.0, self.over, {"cause": "invalid"}
        self.grid = res
        self.score += gain
        self.moves += 1
        self._spawn()
        self.over = not self.valid_actions()
        return self.observe(), float(gain), self.over, {"gain": gain, "max_tile": self.max_tile()}

    def max_tile(self):
        return max(max(row) for row in self.grid)

    def observe(self):
        flat = [self.grid[r][c] for r in range(self.SIZE) for c in range(self.SIZE)]
        return np.array([0.0 if v == 0 else np.log2(v) / 11.0 for v in flat], np.float32)

    def render_state(self):
        return {
            "env": "2048",
            "grid": [[int(v) for v in row] for row in self.grid],
            "score": int(self.score), "moves": int(self.moves),
            "over": bool(self.over), "max_tile": int(self.max_tile()),
        }

    def is_alive(self):
        return not self.over

    def steps_taken(self):
        return self.moves

    def base_score(self):
        return float(self.score)


ENVS = {SnakeEnv.NAME: SnakeEnv, Game2048Env.NAME: Game2048Env}


# --- self-test: run `python envs.py` ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Snake: random policy should end (wall/self/starve) and observations are 11-dim in [0,1]
    s = SnakeEnv(10, 10, rng)
    obs = s.reset()
    assert obs.shape == (11,) and obs.min() >= 0 and obs.max() <= 1
    steps = 0
    while s.alive and steps < 10000:
        s.step(int(rng.integers(3)))
        steps += 1
    assert not s.alive and s.steps > 0
    # food never sits on the snake
    s.reset()
    for _ in range(200):
        assert s.food is None or tuple(s.food) not in set(s.snake)
        if not s.alive:
            s.reset()
        s.step(int(rng.integers(3)))

    # 2048: merges work; valid-move masking; game eventually ends
    g = Game2048Env(rng)
    assert g.observe().shape == (16,)
    merged, gain = Game2048Env._slide_row([2, 2, 4, 0])
    assert merged == [4, 4, 0, 0] and gain == 4, (merged, gain)
    merged, gain = Game2048Env._slide_row([2, 2, 2, 2])
    assert merged == [4, 4, 0, 0] and gain == 8, (merged, gain)     # merge once per move
    moves = 0
    while not g.over and moves < 10000:
        va = g.valid_actions()
        if not va:
            break
        g.step(va[int(rng.integers(len(va)))])
        moves += 1
    assert g.score > 0 and g.max_tile() >= 4
    print(f"envs: snake ended in {s.steps} steps; 2048 random score {g.score} maxtile {g.max_tile()} — OK")
