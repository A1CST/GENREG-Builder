"""GENREG playground simulation engine.

A 2D world of evolvable organisms under a stack of toggleable LAWS OF EXISTENCE.
Everything the GUI shows (board, PO cone, communication) reads from a Sim instance.

============================  ADDING A NEW CONSTRAINT  ========================
A constraint is one small class subclassing `Constraint`. Give it a key/label/colour
and override only the hooks you need:

  cost(sim)        -> per-organism energy cost this step  (float or (N,) array)
  desire(sim)      -> (N,2) movement contribution added to where organisms want to go
  enables_select   -> class attr; True if this law makes death/selection exist
  STRUCTURAL flag  -> for effects that change core logic (eat sharing, memory decay,
                      signalling), gate them inside Sim.step() with `if self.on('key')`.

Then append an instance to CONSTRAINTS. The GUI auto-builds a toggle + parameter
sliders from `.params`. That's it.
=============================================================================
"""
import numpy as np

GENES = ["w_food", "w_mem", "w_sep", "w_sig", "vision", "speed"]


# ----------------------------- constraints ---------------------------------
class Constraint:
    key = ""; label = ""; color = "#888"; desc = ""
    enables_select = False

    def __init__(self):
        self.enabled = False
        # params: list of dicts {name, val, lo, hi, step}
        self.params = [dict(p) for p in getattr(self, "PARAMS", [])]

    def p(self, name):
        for pr in self.params:
            if pr["name"] == name:
                return pr["val"]
        return None

    def cost(self, sim):     # per-organism energy cost
        return 0.0

    def desire(self, sim):   # (N,2) movement contribution, or None
        return None


class Energy(Constraint):
    key = "energy"; label = "Energy (survival)"; color = "#e74c3c"
    desc = "Metabolism drains energy; run out -> die. This is what makes selection exist."
    enables_select = True
    PARAMS = [dict(name="metabolism", val=0.4, lo=0.1, hi=2.0, step=0.05)]
    def cost(self, sim):
        return self.p("metabolism")


class Time(Constraint):
    key = "time"; label = "Time / Occam (move cost)"; color = "#9b59b6"
    desc = "Moving costs energy -> efficient, direct paths are selected."
    PARAMS = [dict(name="move_cost", val=0.12, lo=0.0, hi=0.5, step=0.01)]
    def cost(self, sim):
        return self.p("move_cost") * np.clip(sim.geno[:, 5], 0.2, 2.2)


class Perception(Constraint):
    key = "percep"; label = "Perception cost (looking)"; color = "#3498db"
    desc = "Wider vision costs energy -> vision shrinks to only what's worth seeing."
    PARAMS = [dict(name="cost", val=0.006, lo=0.0, hi=0.03, step=0.001)]
    def cost(self, sim):
        return self.p("cost") * np.clip(sim.geno[:, 4], 4, 40) ** 2 / 10.0


class Entropy(Constraint):
    key = "entropy"; label = "Entropy (memory decays)"; color = "#e08a3a"
    desc = "Remembered food locations decay -> organisms revisit to refresh."
    PARAMS = [dict(name="decay", val=0.90, lo=0.5, hi=0.99, step=0.01)]


class Scarcity(Constraint):
    key = "scarce"; label = "Scarcity (shared food)"; color = "#f1c40f"
    desc = "A patch's food splits among everyone on it -> spread out / territory."
    PARAMS = [dict(name="regen", val=0.004, lo=0.0, hi=0.05, step=0.002)]


class Communication(Constraint):
    key = "comm"; label = "Communication (signals)"; color = "#2ecc71"
    desc = "Organisms on food emit a signal; others move toward signallers."
    PARAMS = [dict(name="range", val=42.0, lo=10.0, hi=80.0, step=2.0)]


class Climate(Constraint):
    key = "climate"; label = "Climate (rotating season)"; color = "#1ab6c4"
    desc = ("Board split in 4 quadrants; the growing season rotates round-robin. Food grows "
            "seedling->mature only in the active quadrant (and only mature food feeds you); "
            "elsewhere it exhausts. Forces the population to migrate with the bloom.")
    # structural: all effects live in Sim.step() gated by self.on('climate')
    PARAMS = [dict(name="period", val=140, lo=20, hi=400, step=10),     # steps per season
              dict(name="growth", val=0.04, lo=0.005, hi=0.15, step=0.005),  # seedling -> mature
              dict(name="decay",  val=0.008, lo=0.0, hi=0.1, step=0.002),    # off-season exhaustion
              dict(name="ripe",   val=0.55, lo=0.2, hi=0.95, step=0.05)]     # maturity needed to feed


class Reproduction(Constraint):
    key = "repro"; label = "Sexual reproduction (mating)"; color = "#ff6fae"
    desc = ("Turns OFF clone-the-best. Evolution takes over: a genome reproduces ONLY with a "
            "nearby MATE, and only if BOTH parents are at >=50% energy. Offspring = genetic "
            "crossover of the two + mutation; both parents pay an energy cost. Population now "
            "floats freely with real births and deaths (no fixed N).")
    # structural: handled in Sim.step(); replaces the cull-and-clone operator entirely
    PARAMS = [dict(name="mate_radius", val=12, lo=4, hi=40, step=2),
              dict(name="repro_cost", val=8, lo=2, hi=20, step=1),
              dict(name="max_pop", val=140, lo=20, hi=300, step=10)]


# the registry the GUI reads. Append new constraints here.
def make_constraints():
    return [Energy(), Time(), Perception(), Entropy(), Scarcity(), Communication(), Climate(), Reproduction()]


# ------------------------------- the world ---------------------------------
class Sim:
    def __init__(self, seed=7):
        self.seed = seed
        self.constraints = make_constraints()
        self.cmap = {c.key: c for c in self.constraints}
        # population / world controls (the GUI edits these live)
        self.ctrl = dict(N=32, NF=46, L=100.0, hear=42.0, mut=0.2,
                         cull_frac=0.08, food_amt0=0.8, kin_spread=4.0, noise=0.45)
        self.ecap = 40.0          # energy capacity; mating needs both parents >= 50% of this
        self.reset()

    # ---- constraint access ----
    def on(self, key):
        c = self.cmap.get(key)
        return bool(c and c.enabled)

    def po(self):
        return sum(c.enabled for c in self.constraints)

    # ---- lifecycle ----
    def reset(self):
        r = np.random.default_rng(self.seed)
        self.rng = r
        N = int(self.ctrl["N"]); NF = int(self.ctrl["NF"]); L = self.ctrl["L"]
        g = r.normal(0, 1.0, (N, 6))
        g[:, 4] = r.uniform(10, 30, N)      # vision
        g[:, 5] = r.uniform(0.6, 1.8, N)    # speed
        self.geno = g
        self.pos = np.clip(L/2 + r.normal(0, L/6, (N, 2)), 0, L)   # start central: reachable to any quadrant
        self.energy = np.full(N, 14.0)
        self.mem = r.uniform(0, L, (N, 2)); self.memc = np.zeros(N)
        # food spread EVENLY across the 4 quadrants (so every climate season has food)
        per = NF // 4; corners = [(0, 0), (L/2, 0), (L/2, L/2), (0, L/2)]
        xs, ys = [], []
        for qi, (x0, y0) in enumerate(corners):
            n = per if qi < 3 else NF - 3 * per
            xs.append(r.uniform(x0 + 4, x0 + L/2 - 4, n)); ys.append(r.uniform(y0 + 4, y0 + L/2 - 4, n))
        self.food_xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.food_amt = r.uniform(0.4, self.ctrl["food_amt0"], NF)
        # climate: per-patch maturity (seedling 0 -> mature 1). Quadrant 0 starts ripe.
        self.food_mat = r.uniform(0.1, 0.4, NF)
        q0 = self._food_quads() == 0
        self.food_mat[q0] = r.uniform(0.6, 1.0, int(q0.sum()))
        self.active_quadrant = 0
        self.signalers = np.zeros(N, bool)
        self.edges = []                      # (speaker, listener) this step
        self.t = 0; self.births = 0
        self.spread0 = self._spread()
        self.comm_hist = []                  # follow-events (info acted on) over time
        self.sig_hist = []                   # signaller count over time

    def _food_quads(self):
        """quadrant index 0..3 per food patch, ordered clockwise: BL,BR,TR,TL."""
        L = self.ctrl["L"]; right = self.food_xy[:, 0] >= L/2; top = self.food_xy[:, 1] >= L/2
        return np.where(~top & ~right, 0, np.where(~top & right, 1, np.where(top & right, 2, 3)))

    def _spread(self):
        if len(self.geno) < 2:
            return 0.0
        g = self.geno
        rel = g.std(0) / (np.abs(g.mean(0)) + 0.6)
        return float(np.tanh(rel).mean())

    def spread_frac(self):
        return self._spread() / max(self.spread0, 1e-6)

    # ---- one tick ----
    def step(self):
        if len(self.geno) == 0:                 # population went extinct (mating mode)
            self.t += 1; self.comm_hist.append(0); self.sig_hist.append(0)
            if len(self.comm_hist) > 300: self.comm_hist.pop(0); self.sig_hist.pop(0)
            return
        r = self.rng; N = len(self.geno); L = self.ctrl["L"]
        pos, geno, fx, fa, en = self.pos, self.geno, self.food_xy, self.food_amt, self.energy
        self.signalers = np.zeros(N, bool); self.edges = []
        comm_on = self.on("comm"); hear = self.cmap["comm"].p("range") if comm_on else 0
        # CLIMATE: only ripe food in/around the active season is real food
        climate_on = self.on("climate")
        if climate_on:
            cc = self.cmap["climate"]; per = max(1, int(cc.p("period")))
            self.active_quadrant = int((self.t // per) % 4)
            vis_mask = (fa > 0.05) & (self.food_mat >= 0.25)            # SEE growing food (migrate toward bloom)
            eat_mask = (fa > 0.05) & (self.food_mat >= cc.p("ripe"))   # only MATURE food feeds you
        else:
            vis_mask = eat_mask = (fa > 0.05)
        # --- sense + decide ---
        desired = np.zeros((N, 2))
        on_food_now = np.zeros(N, bool)
        for i in range(N):
            vis = np.clip(geno[i, 4], 4, 40)
            d = fx - pos[i]; dist = np.linalg.norm(d, axis=1)
            seen_eat = (dist < vis) & eat_mask          # ripe & edible -> first choice
            seen_see = (dist < vis) & vis_mask          # growing/seedling -> migrate toward it
            if seen_eat.any():
                j = int(np.where(seen_eat)[0][np.argmin(dist[seen_eat])])
            elif seen_see.any():
                j = int(np.where(seen_see)[0][np.argmin(dist[seen_see])])
            else:
                j = -1
            if j >= 0:
                desired[i] += geno[i, 0] * (d[j] / max(dist[j], 1e-6))
                if dist[j] < 4 and eat_mask[j]:
                    self.mem[i] = fx[j]; self.memc[i] = 1.0; on_food_now[i] = True
            if self.memc[i] > 0.05:
                dm = self.mem[i] - pos[i]; dd = np.linalg.norm(dm)
                desired[i] += geno[i, 1] * self.memc[i] * (dm / max(dd, 1e-6))
        self.signalers = on_food_now & comm_on
        # separation (scarcity) + signal-following (comm) via desire hooks
        if self.on("scarce"):
            for i in range(N):
                dn = pos[i] - pos; dnn = np.linalg.norm(dn, axis=1)
                near = (dnn > 1e-6) & (dnn < 16)
                if near.any():
                    desired[i] += geno[i, 2] * (dn[near] / (np.linalg.norm(dn[near], axis=1, keepdims=True)+1e-6)).sum(0)
        if comm_on and self.signalers.any():
            src = pos[self.signalers]; src_idx = np.where(self.signalers)[0]
            for i in range(N):
                if on_food_now[i]:
                    continue
                ds = src - pos[i]; dss = np.linalg.norm(ds, axis=1); k = int(np.argmin(dss))
                if dss[k] < hear:
                    desired[i] += geno[i, 3] * (ds[k] / max(dss[k], 1e-6))
                    self.edges.append((int(src_idx[k]), i))
        desired += self.ctrl["noise"] * r.normal(0, 1, (N, 2))
        # --- move ---
        nd = np.linalg.norm(desired, axis=1, keepdims=True)
        step_dir = desired / np.where(nd < 1e-6, 1, nd)
        spd = np.clip(geno[:, 5], 0.2, 2.2)
        self.pos = np.clip(pos + step_dir * spd[:, None], 0, L); pos = self.pos
        # --- eat (only AVAILABLE food: has amount, and if climate on, is ripe) ---
        D = np.linalg.norm(fx[None, :, :] - pos[:, None, :], axis=2)   # (N, NF)
        if not eat_mask.all():
            D[:, ~eat_mask] = np.inf
        nearest = D.argmin(1); ndist = D[np.arange(N), nearest]
        onf = ndist < 4
        scarce = self.on("scarce")
        for i in range(N):
            if onf[i]:
                j = nearest[i]
                share = int((onf & (nearest == j)).sum()) if scarce else 1
                b = min(0.5, fa[j]) / share; en[i] += b * 3.0
                if scarce:
                    fa[j] -= b
                if climate_on:
                    self.food_mat[j] = max(0.0, self.food_mat[j] - 0.18)   # eating ripens-down to seedling
        if scarce:
            self.food_amt[:] = np.minimum(1.0, fa + self.cmap["scarce"].p("regen"))
        # --- climate: grow the active quadrant, exhaust the rest ---
        if climate_on:
            ins = self._food_quads() == self.active_quadrant
            self.food_mat[ins] = np.minimum(1.0, self.food_mat[ins] + cc.p("growth"))
            self.food_mat[~ins] = np.maximum(0.0, self.food_mat[~ins] - cc.p("decay"))
        # --- memory decay (entropy) ---
        self.memc *= (self.cmap["entropy"].p("decay") if self.on("entropy") else 0.995)
        # --- costs from every enabled constraint ---
        total = np.zeros(N)
        for c in self.constraints:
            if c.enabled:
                total = total + np.asarray(c.cost(self))
        self.energy = en - total
        # --- reproduction ---
        self.energy[:] = np.clip(self.energy, -2, self.ecap)
        if self.on("repro"):
            self._reproduce_mates(self.cmap["repro"])               # MATING: evolution in control
        elif any(c.enabled and c.enables_select for c in self.constraints):
            K = max(1, int(self.ctrl["cull_frac"] * N))             # steady-state clone-the-best
            order = np.argsort(self.energy); worst = order[:K]; top = order[N - max(2, N//3):]
            med = float(np.median(self.energy)); mut = self.ctrl["mut"]
            for w in worst:
                p = int(top[r.integers(len(top))])
                self.geno[w] = self.geno[p] + r.normal(0, mut, 6) * (np.abs(self.geno[p]) + 0.3)
                self.geno[w, :4] = np.clip(self.geno[w, :4], -6, 6)
                self.geno[w, 4] = np.clip(self.geno[w, 4], 4, 40)
                self.geno[w, 5] = np.clip(self.geno[w, 5], 0.2, 2.2)
                self.pos[w] = np.clip(self.pos[p] + r.normal(0, self.ctrl["kin_spread"], 2), 0, L)
                self.energy[w] = max(med, 4.0); self.memc[w] = 0
            self.births += K
            self.energy[:] = np.clip(self.energy, -2, self.ecap)
        self.t += 1
        self.comm_hist.append(len(self.edges))
        self.sig_hist.append(int(self.signalers.sum()))
        if len(self.comm_hist) > 300:
            self.comm_hist.pop(0); self.sig_hist.pop(0)

    # ---- mate-based reproduction (variable population) ----
    def _keep(self, mask):
        remap = np.full(len(mask), -1); remap[mask] = np.arange(int(mask.sum()))
        for k in ("geno", "pos", "energy", "mem", "memc", "signalers"):
            setattr(self, k, getattr(self, k)[mask])
        self.edges = [(int(remap[a]), int(remap[b])) for a, b in self.edges if mask[a] and mask[b]]

    def _append(self, genos, poss, birth_e):
        self.geno = np.vstack([self.geno, genos]); self.pos = np.vstack([self.pos, poss])
        self.energy = np.concatenate([self.energy, np.full(len(genos), birth_e)])
        self.mem = np.vstack([self.mem, poss]); self.memc = np.concatenate([self.memc, np.zeros(len(genos))])
        self.signalers = np.concatenate([self.signalers, np.zeros(len(genos), bool)])

    def _reproduce_mates(self, repro):
        r = self.rng; L = self.ctrl["L"]; thr = 0.5 * self.ecap
        # deaths: anyone out of energy is gone
        alive = self.energy > 0
        if not alive.all():
            self._keep(alive)
        N = len(self.geno)
        if N < 2:
            return
        en = self.energy; pos = self.pos
        mate_r = repro.p("mate_radius"); cost = repro.p("repro_cost"); maxpop = int(repro.p("max_pop"))
        elig = np.where(en >= thr)[0]                      # both parents need >= 50% energy
        if len(elig) < 2:
            return
        elig_set = set(int(x) for x in elig)
        order = list(elig); r.shuffle(order)
        used = np.zeros(N, bool); bg, bp = [], []
        for i in order:
            if used[i] or N + len(bg) >= maxpop:
                continue
            d = np.linalg.norm(pos - pos[i], axis=1)
            best, bestd = -1, mate_r                       # nearest eligible unused mate in range
            for j in elig_set:
                if j != i and not used[j] and d[j] < bestd:
                    bestd, best = d[j], j
            if best < 0:
                continue
            j = best; used[i] = used[j] = True
            mask = r.random(6) < 0.5                        # genetic CROSSOVER of the two parents
            child = np.where(mask, self.geno[i], self.geno[j]) + r.normal(0, self.ctrl["mut"] * 0.6, 6)
            child[:4] = np.clip(child[:4], -6, 6)
            child[4] = np.clip(child[4], 4, 40); child[5] = np.clip(child[5], 0.2, 2.2)
            bg.append(child); bp.append((pos[i] + pos[j]) / 2 + r.normal(0, 2, 2))
            en[i] -= cost; en[j] -= cost                    # both parents pay
        if bg:
            self._append(np.array(bg), np.clip(np.array(bp), 0, L), float(cost))
            self.births += len(bg)

    # ---- readouts for the GUI ----
    def generation(self):
        return self.births // max(1, len(self.geno))
