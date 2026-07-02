"""Vectorized neuroevolution engine (PyTorch, device-selectable).

The whole population is a batch of tensors, so all `P` genomes step `P` parallel
games in lockstep and mutate/select as batched tensor ops. This runs on CPU or
CUDA; at small P the CPU engine (genreg-engine) is faster (tiny sequential ops),
but this scales to large populations and heavier (vectorizable) environments
where the GPU wins.

Scope/limits vs the CPU engine (batching needs uniform shapes):
  * fixed hidden width H (no dimension evolution)
  * full precision (no per-neuron bit-depth genes)
  * recurrence + relative self-adapting mutation are kept

Weights live in a dict of tensors [P, ...]; the champion is extracted back to a
plain numpy genome (see `champion_numpy`) so the Microscope/replay reuse the CPU
path unchanged.
"""
import numpy as np
import torch

EPS = 1e-3
TAU = 0.20
RATE_BOUNDS = (0.02, 0.70)
SCALE_BOUNDS = (0.02, 0.80)
WEIGHT_KEYS = ("W1", "b1", "Wrec", "W2", "b2")


def device_of(name):
    if name in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------- population
def init_pop(P, nin, H, nout, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def rn(*shape, sc):
        return (torch.randn(*shape, generator=g) * sc).to(device)

    return {
        "W1": rn(P, nin, H, sc=1.0 / nin ** 0.5), "b1": torch.zeros(P, H, device=device),
        "Wrec": rn(P, H, H, sc=0.1 / H ** 0.5), "leak": torch.full((P,), 0.5, device=device),
        "W2": rn(P, H, nout, sc=1.0 / H ** 0.5), "b2": torch.zeros(P, nout, device=device),
        "mr": torch.full((P,), 0.2, device=device), "ms": torch.full((P,), 0.2, device=device),
    }


def gather_pop(pop, idx):
    return {k: v[idx].clone() for k, v in pop.items()}


def mutate_pop(pop, device):
    """Batched relative, self-adapting mutation (mirrors mutation.py)."""
    P = pop["ms"].shape[0]
    pop["ms"] = (pop["ms"] * torch.exp(TAU * torch.randn(P, device=device))).clamp(*SCALE_BOUNDS)
    pop["mr"] = (pop["mr"] * torch.exp(TAU * torch.randn(P, device=device))).clamp(*RATE_BOUNDS)
    for k in WEIGHT_KEYS:
        W = pop[k]
        view = (P,) + (1,) * (W.dim() - 1)
        mr = pop["mr"].view(view)
        ms = pop["ms"].view(view)
        mask = (torch.rand_like(W) < mr).float()
        pop[k] = W + mask * torch.randn_like(W) * ms * (W.abs() + EPS)
    pop["leak"] = (pop["leak"] + 0.10 * torch.randn(P, device=device)).clamp(0.0, 0.99)
    return pop


def bforward(obs, state, pop):
    """Batched recurrent step. obs [P,nin], state [P,H] -> (out [P,nout], state [P,H])."""
    cand = torch.tanh(torch.bmm(obs.unsqueeze(1), pop["W1"]).squeeze(1)
                      + torch.bmm(state.unsqueeze(1), pop["Wrec"]).squeeze(1) + pop["b1"])
    leak = pop["leak"].unsqueeze(1)
    state = leak * state + (1.0 - leak) * cand
    out = torch.bmm(state.unsqueeze(1), pop["W2"]).squeeze(1) + pop["b2"]
    return out, state


# ----------------------------------------------------------------- vectorized snake
_DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]]   # up, right, down, left (dr, dc)


class VSnake:
    """P parallel snake games on a grid. Rules match envs.SnakeEnv closely.

    Reward is a world-consequence: base = apples*1.0 + steps*0.01 (per game).
    """
    N_IN, N_OUT = 11, 3
    APPLE, STEP_REWARD = 1.0, 0.01

    def __init__(self, P, h, w, device, seed=0, max_hunger=None):
        self.P, self.h, self.w, self.device = P, h, w, device
        self.MAX = h * w + 3
        self.max_hunger = int(max_hunger) if max_hunger else 2 * (h + w)
        self.DIRS = torch.tensor(_DIRS, device=device, dtype=torch.long)
        self.g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.reset()

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g).to(self.device)

    def reset(self):
        P, h, w, dev = self.P, self.h, self.w, self.device
        cy, cx = h // 2, w // 2
        self.head = torch.stack([torch.full((P,), cy, device=dev), torch.full((P,), cx, device=dev)], 1).long()
        self.dir = torch.full((P,), 1, device=dev, dtype=torch.long)   # right
        self.occ = torch.zeros(P, h, w, dtype=torch.bool, device=dev)
        self.length = torch.full((P,), 3, device=dev, dtype=torch.long)
        self.body = torch.zeros(P, self.MAX, 2, dtype=torch.long, device=dev)
        ar = torch.arange(P, device=dev)
        for i, dx in enumerate((-2, -1, 0)):                            # tail..head
            cell = torch.stack([torch.full((P,), cy, device=dev), torch.full((P,), cx + dx, device=dev)], 1)
            self.body[:, i] = cell
            self.occ[ar, cell[:, 0], cell[:, 1]] = True
        self.ptr = torch.full((P,), 2, device=dev, dtype=torch.long)
        self.alive = torch.ones(P, dtype=torch.bool, device=dev)
        self.hunger = torch.zeros(P, dtype=torch.long, device=dev)
        self.score = torch.zeros(P, dtype=torch.long, device=dev)
        self.steps = torch.zeros(P, dtype=torch.long, device=dev)
        self.food = torch.zeros(P, 2, dtype=torch.long, device=dev)
        self._place_food(torch.ones(P, dtype=torch.bool, device=dev))
        return self.obs()

    def _place_food(self, mask):
        P, h, w = self.P, self.h, self.w
        r = self._rand(P, h, w)
        r[self.occ] = -1.0
        flat = r.view(P, -1).argmax(1)
        newf = torch.stack([flat // w, flat % w], 1)
        self.food = torch.where(mask.unsqueeze(1), newf, self.food)

    def _blocked(self, cells):
        r, c = cells[:, 0], cells[:, 1]
        oob = (r < 0) | (r >= self.h) | (c < 0) | (c >= self.w)
        occ = self.occ[torch.arange(self.P, device=self.device), r.clamp(0, self.h - 1), c.clamp(0, self.w - 1)]
        return (oob | occ).float()

    def obs(self):
        d = self.dir
        dang = torch.stack([self._blocked(self.head + self.DIRS[d]),
                            self._blocked(self.head + self.DIRS[(d - 1) % 4]),
                            self._blocked(self.head + self.DIRS[(d + 1) % 4])], 1)
        onehot = torch.nn.functional.one_hot(d, 4).float()
        fr, fc, hr, hc = self.food[:, 0], self.food[:, 1], self.head[:, 0], self.head[:, 1]
        food = torch.stack([(fr < hr).float(), (fc > hc).float(), (fr > hr).float(), (fc < hc).float()], 1)
        return torch.cat([dang, onehot, food], 1)

    def step(self, actions):
        P, dev = self.P, self.device
        ar = torch.arange(P, device=dev)
        self.dir = (self.dir + torch.tensor([-1, 0, 1], device=dev)[actions]) % 4
        nh = self.head + self.DIRS[self.dir]
        inb = (nh[:, 0] >= 0) & (nh[:, 0] < self.h) & (nh[:, 1] >= 0) & (nh[:, 1] < self.w)
        eating = inb & (nh[:, 0] == self.food[:, 0]) & (nh[:, 1] == self.food[:, 1])

        not_eat = self.alive & ~eating
        tail_idx = (self.ptr - self.length + 1) % self.MAX
        tail = self.body[ar, tail_idx]
        self.occ[ar[not_eat], tail[not_eat, 0], tail[not_eat, 1]] = False

        rc, cc = nh[:, 0].clamp(0, self.h - 1), nh[:, 1].clamp(0, self.w - 1)
        occ_hit = self.occ[ar, rc, cc] & inb
        collide = self.alive & (~inb | occ_hit)
        move = self.alive & ~collide

        self.ptr = torch.where(move, (self.ptr + 1) % self.MAX, self.ptr)
        self.body[ar[move], self.ptr[move]] = nh[move]
        self.occ[ar[move], nh[move, 0], nh[move, 1]] = True
        self.head = torch.where(move.unsqueeze(1), nh, self.head)

        eat = move & eating
        self.length = torch.where(eat, self.length + 1, self.length)
        self.score = torch.where(eat, self.score + 1, self.score)
        self.hunger = torch.where(eat, torch.zeros_like(self.hunger), self.hunger + 1)
        self.steps = torch.where(move, self.steps + 1, self.steps)
        if eat.any():
            self._place_food(eat)

        self.alive = self.alive & ~collide & (self.hunger <= self.max_hunger)

    def base(self):
        return self.score.float() * self.APPLE + self.steps.float() * self.STEP_REWARD


# ----------------------------------------------------------------- vectorized 2048
class V2048:
    """P parallel 2048 boards. Batched slide/merge; per-board action each step."""
    N_IN, N_OUT = 16, 4          # obs = log2 tiles; actions 0=up,1=right,2=down,3=left

    def __init__(self, P, device, seed=0):
        self.P, self.device = P, device
        self.g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.reset()

    def reset(self):
        self.grid = torch.zeros(self.P, 4, 4, dtype=torch.long, device=self.device)
        self.score = torch.zeros(self.P, dtype=torch.long, device=self.device)
        self.moves = torch.zeros(self.P, dtype=torch.long, device=self.device)
        self.over = torch.zeros(self.P, dtype=torch.bool, device=self.device)
        self._spawn(torch.ones(self.P, dtype=torch.bool, device=self.device))
        self._spawn(torch.ones(self.P, dtype=torch.bool, device=self.device))
        return self.obs()

    def _spawn(self, mask):
        P = self.P
        empty = self.grid == 0
        r = torch.rand(P, 4, 4, generator=self.g).to(self.device)
        r[~empty] = -1.0
        flat = r.view(P, -1).argmax(1)
        val = torch.where(torch.rand(P, generator=self.g).to(self.device) < 0.9,
                          torch.full((P,), 2, device=self.device), torch.full((P,), 4, device=self.device))
        do = mask & empty.view(P, -1).any(1)
        ar = torch.arange(P, device=self.device)
        self.grid[ar[do], (flat // 4)[do], (flat % 4)[do]] = val[do].long()

    @staticmethod
    def _slide_left(rows):
        # compress (nonzeros left, stable), merge once L->R, compress again
        order = torch.argsort((rows == 0).long(), dim=1, stable=True)
        c = torch.gather(rows, 1, order)
        gain = torch.zeros(rows.shape[0], dtype=torch.long, device=rows.device)
        for i in range(3):
            eq = (c[:, i] == c[:, i + 1]) & (c[:, i] != 0)
            c[:, i] = torch.where(eq, c[:, i] * 2, c[:, i])
            gain = gain + torch.where(eq, c[:, i], torch.zeros_like(gain))
            c[:, i + 1] = torch.where(eq, torch.zeros_like(c[:, i + 1]), c[:, i + 1])
        order2 = torch.argsort((c == 0).long(), dim=1, stable=True)
        return torch.gather(c, 1, order2), gain

    def _apply_dir(self, d):
        grid = self.grid
        if d == 3:   rows = grid.reshape(-1, 4)
        elif d == 1: rows = grid.flip(2).reshape(-1, 4)
        elif d == 0: rows = grid.transpose(1, 2).reshape(-1, 4)
        else:        rows = grid.transpose(1, 2).flip(2).reshape(-1, 4)   # down
        c, gain = self._slide_left(rows)
        c = c.reshape(grid.shape)
        if d == 3:   res = c
        elif d == 1: res = c.flip(2)
        elif d == 0: res = c.transpose(1, 2)
        else:        res = c.flip(2).transpose(1, 2)
        changed = (res != grid).flatten(1).any(1)
        return res, gain.reshape(self.P, 4).sum(1), changed

    def dir_results(self):
        res, gain, changed = [], [], []
        for d in range(4):
            r, g, ch = self._apply_dir(d)
            res.append(r); gain.append(g); changed.append(ch)
        return torch.stack(res), torch.stack(gain), torch.stack(changed)   # [4,P,..],[4,P],[4,P]

    def valid_mask(self, changed):
        return changed.transpose(0, 1)                                     # [P,4]

    def step(self, res, gain, changed, actions):
        ar = torch.arange(self.P, device=self.device)
        newgrid = res[actions, ar]
        got = gain[actions, ar]
        did = changed[actions, ar] & ~self.over
        self.grid = torch.where(did.view(self.P, 1, 1), newgrid, self.grid)
        self.score = self.score + torch.where(did, got, torch.zeros_like(got))
        self.moves = self.moves + did.long()
        if did.any():
            self._spawn(did)
        self.over = ~self.dir_results()[2].transpose(0, 1).any(1)          # no valid move left
        return got * did                                                   # per-board reward this step

    def obs(self):
        g = self.grid.float()
        o = torch.where(g > 0, torch.log2(g.clamp(min=1)) / 11.0, torch.zeros_like(g))
        return o.reshape(self.P, 16)

    def base(self):
        return self.score.float()

    def all_over(self):
        return bool(self.over.all())


def evaluate_2048(pop, P, Hd, device, episodes=4, cap=800, base_seed=0, C=None):
    scores = []
    for e in range(episodes):
        env = V2048(P, device, seed=base_seed * 1000 + e)
        state = torch.zeros(P, Hd, device=device)
        energy = torch.full((P,), C["budget"], device=device) if (C and C["energy"]) else None
        for _ in range(cap):
            obs = _apply_obs(env.obs(), C, device) if C else env.obs()
            logits, state = bforward(obs, state, pop)
            if C and C["decay"]:
                state = state * (1.0 - C["decay"])
            res, gain, changed = env.dir_results()
            masked = logits.masked_fill(~env.valid_mask(changed), -1e9)
            reward = env.step(res, gain, changed, masked.argmax(1))    # per-board gain this move
            if C and C["energy"]:
                energy = energy - C["step_cost"] + (reward.float() / 2.0) * C["merge_mult"]
                env.over = env.over | (energy <= 0)
            if env.all_over():
                break
        base = env.base()
        if C and C["time_budget"]:
            base = base / (1.0 + env.moves.float() / C["time_budget"])
        scores.append(base)
    fit = torch.stack(scores, 1).median(1).values
    if C and C["weight_k"]:
        fit = fit - C["weight_k"] * mean_abs_weights(pop)
    return fit


# ----------------------------------------------------------------- evaluation + evolution
# ----------------------------------------------------------------- vectorized constraints
# Subset of the EEC constraints that map cleanly to batched ops. Others (memory-rent,
# scarcity, reproduction-cost, non-stationarity, perception-cost) still use the CPU engine.
VEC_SUPPORTED = {"energy", "mortality", "time", "noise", "occlusion", "entropy", "efficiency"}


def build_vconstraints(names, params, device):
    from constraints_map import DEFAULTS
    p = {**DEFAULTS, **(params or {})}
    have = set(names or [])
    return {
        "noise": float(p["noise"]) if "noise" in have else 0.0,
        "occ": float(p["occlusion_p"]) if "occlusion" in have else 0.0,
        "decay": float(p["decay"]) if "entropy" in have else 0.0,
        "energy": "energy" in have,
        "mortality": ("mortality" in have) and ("energy" not in have),
        "budget": float(p["energy_budget"]), "step_cost": float(p["step_cost"]),
        "food_energy": float(p["food_energy"]), "merge_mult": float(p["merge_energy"]),
        "hazard": float(p["hazard"]),
        "time_budget": float(p["time_budget"]) if "time" in have else 0.0,
        "weight_k": float(p["cost_strength"]) if "efficiency" in have else 0.0,
    }


def _apply_obs(obs, C, device):
    if C["occ"]:
        mask = torch.rand(obs.shape[0], 1, device=device) < C["occ"]
        obs = torch.where(mask, torch.zeros_like(obs), obs)
    if C["noise"]:
        obs = obs + torch.randn_like(obs) * C["noise"]
    return obs


def mean_abs_weights(pop):
    P = pop["W1"].shape[0]
    tot = torch.zeros(P, device=pop["W1"].device)
    for k in WEIGHT_KEYS:
        tot = tot + pop[k].abs().flatten(1).mean(1)
    return tot / len(WEIGHT_KEYS)


def evaluate_snake(pop, P, gh, gw, Hd, device, episodes=4, cap=600, base_seed=0, C=None):
    """Median base-score per genome over `episodes` batched games (optional constraints)."""
    scores = []
    relax = int(cap) if (C and (C["energy"] or C["mortality"])) else None
    for e in range(episodes):
        env = VSnake(P, gh, gw, device, seed=base_seed * 1000 + e, max_hunger=relax)
        state = torch.zeros(P, Hd, device=device)
        energy = torch.full((P,), C["budget"], device=device) if (C and C["energy"]) else None
        for _ in range(cap):
            obs = _apply_obs(env.obs(), C, device) if C else env.obs()
            out, state = bforward(obs, state, pop)
            if C and C["decay"]:
                state = state * (1.0 - C["decay"])
            prev = env.score.clone()
            env.step(out.argmax(1))
            if C and C["energy"]:
                energy = energy - C["step_cost"] + (env.score > prev).float() * C["food_energy"]
                env.alive = env.alive & (energy > 0)
            elif C and C["mortality"]:
                frac = env.hunger.float() / max(1, env.max_hunger)
                env.alive = env.alive & ~(torch.rand(P, device=device) < C["hazard"] * frac)
            if not env.alive.any():
                break
        base = env.base()
        if C and C["time_budget"]:
            base = base / (1.0 + env.steps.float() / C["time_budget"])
        scores.append(base)
    fit = torch.stack(scores, 1).median(1).values
    if C and C["weight_k"]:
        fit = fit - C["weight_k"] * mean_abs_weights(pop)
    return fit


def champion_numpy(pop, idx):
    """Extract genome `idx` as numpy arrays (engine [in,out] orientation)."""
    i = int(idx)
    g = {k: pop[k][i].detach().cpu().numpy() for k in ("W1", "b1", "Wrec", "W2", "b2", "leak")}
    return g


def evolve_snake(P=256, gens=30, gh=12, gw=12, Hd=24, device="cpu", seed=0,
                 elite=2, parent_frac=0.25, episodes=4, log=None):
    dev = device_of(device)
    pop = init_pop(P, VSnake.N_IN, Hd, VSnake.N_OUT, dev, seed)
    n_par = max(2, int(P * parent_frac))
    history = []
    best_idx = 0
    for gen in range(gens):
        fit = evaluate_snake(pop, P, gh, gw, Hd, dev, episodes, base_seed=seed + gen)
        order = torch.argsort(fit, descending=True)
        best_idx = int(order[0])
        history.append((float(fit[best_idx]), float(fit.mean())))
        if log:
            log(gen, float(fit[best_idx]), float(fit.mean()))
        parents = order[:n_par]
        pidx = parents[torch.randint(n_par, (P,), device=dev)]
        child = gather_pop(pop, pidx)
        child = mutate_pop(child, dev)
        elites = gather_pop(pop, order[:elite])
        for k in child:
            child[k][:elite] = elites[k]
        pop = child
    # final eval to pick champion
    fit = evaluate_snake(pop, P, gh, gw, Hd, dev, episodes, base_seed=seed + gens)
    best_idx = int(torch.argmax(fit))
    return pop, best_idx, history


# ----------------------------------------------------------------- self-test / benchmark
if __name__ == "__main__":
    import time
    print("cuda:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    # 1. does vectorized snake LEARN? (cpu, modest pop)
    def logline(gen, best, mean):
        if gen % 4 == 0 or gen == 29:
            print(f"  gen {gen:3d}  best {best:.2f}  mean {mean:.2f}")
    print("== learning test (cpu, P=256, 30 gens) ==")
    t0 = time.time()
    pop, bi, hist = evolve_snake(P=256, gens=30, device="cpu", seed=0, log=logline)
    print(f"  first-gen best {hist[0][0]:.2f} -> last {hist[-1][0]:.2f}   ({time.time()-t0:.1f}s)")

    # 2. GPU vs CPU per-generation time across population sizes
    print("== per-generation wall time: cpu vs cuda ==")
    for P in (256, 1024, 4096, 16384):
        tc = time.time(); evolve_snake(P=P, gens=2, device="cpu", seed=1); tcpu = (time.time() - tc) / 2
        line = f"  P={P:6d}  cpu {tcpu*1000:8.1f} ms/gen"
        if torch.cuda.is_available():
            evolve_snake(P=P, gens=1, device="cuda", seed=1)  # warmup
            tg = time.time(); evolve_snake(P=P, gens=2, device="cuda", seed=1); tgpu = (time.time() - tg) / 2
            line += f"  |  cuda {tgpu*1000:8.1f} ms/gen  ({tcpu/tgpu:.2f}x)"
        print(line)
