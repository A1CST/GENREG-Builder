"""radial_lensmap.py — LIVE, OPEN-ENDED, COMPOUNDING exploration of the lens map.

The lens space is INFINITE. A lens is a small program in the activation algebra
over the feature axes:
  * combine any number of axes with coefficients  (proj = sum c_k * axis_k)
  * compose activations to any depth               (tanh(sin(proj)))
  * optionally multiply two sub-lenses             (l1 * l2)
So there is no fixed grid — exploration is a generator that keeps producing
ever-richer lenses (growing combinatorial order + composition depth) and never
"finishes". You Stop it when you want. Each lens has a deterministic address
(its index seeds its program), so the map is still a fixed coordinate system —
just an unbounded one. It is laid out radially: complexity = radius, so simple
lenses fill the centre and richer ones spiral outward.

COMPOUNDING (Extend): freeze a layer's encoder and generate the next layer's
lenses over its OUTPUT. RESIDUAL, so extending only adds.

Self-contained: reads only radial_data/cifar_radial.npz. Honest by construction —
the encoder-accuracy curve is computed live and will plateau even as the infinite
map keeps growing (infinite lenses, finite useful information).
"""
import os
import threading
import time
import numpy as np

from radial_lens import ACTS, _zscore

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPZ = os.path.join(_HERE, "radial_data", "cifar_radial.npz")
ACT_NAMES = ["sin", "cos", "tanh", "relu", "gauss", "abs", "sq"]
BASE_DIM = 64
K = 64
BANK_CAP = 2600          # reservoir size for the encoder (bounded RAM)
SOFT_CAP = 20000         # safety stop; Stop button ends earlier
GOLD = np.pi * (3 - np.sqrt(5))

_DATA = {}
STACK, HISTORY, STATE = [], [], {}
_LOCK = threading.Lock()


def _rand_pca(X, k, seed=0):
    """Top-k right singular vectors via randomized SVD — avoids the huge full-U
    SVD on the tall pixel matrix (that was making _load look frozen)."""
    G = X - X.mean(0)
    rng = np.random.default_rng(seed)
    Y = G @ rng.standard_normal((G.shape[1], k + 12)).astype(np.float32)
    Q, _ = np.linalg.qr(Y)
    Vt = np.linalg.svd(Q.T @ G, full_matrices=False)[2]
    return Vt[:k]


def _load():
    if _DATA:
        return _DATA
    d = np.load(_NPZ)
    Xtr = d["Xtr"].reshape(len(d["Xtr"]), -1).astype(np.float32) / 255.
    Xte = d["Xte"].reshape(len(d["Xte"]), -1).astype(np.float32) / 255.
    Vt = _rand_pca(Xtr, BASE_DIM)
    btr, mu, sd = _zscore(Xtr @ Vt[:BASE_DIM].T)
    bte = (Xte @ Vt[:BASE_DIM].T - mu) / sd
    _DATA.update({"base_tr": btr, "base_te": bte, "ytr": d["ytr"], "yte": d["yte"],
                  "names": [str(n) for n in d["names"]], "Xtr_u8": d["Xtr_u8"] if "Xtr_u8" in d else d["Xtr"]})
    return _DATA


def _fknn(Ftr, ytr, Fte, yte, k=5):
    Ftr, m, s = _zscore(Ftr); Fte = (Fte - m) / s
    D2 = (Fte ** 2).sum(1)[:, None] + (Ftr ** 2).sum(1)[None, :] - 2 * Fte @ Ftr.T
    nn = np.argpartition(D2, k, axis=1)[:, :k]
    pred = np.array([np.bincount(ytr[nn[r]], minlength=10).argmax() for r in range(len(Fte))])
    return float((pred == yte).mean())


def _rotate_frame(Z, deg):
    """Rotate the whole coordinate frame by `deg` (Givens rotations on consecutive
    axis pairs). Lets a stacked layer view the SAME features from a rotated angle
    instead of the identical frame — the 'step the radial axis' option."""
    if not deg:
        return Z
    th = np.radians(deg); c, s = np.cos(th), np.sin(th)
    Z = Z.copy()
    for i in range(0, Z.shape[1] - 1, 2):
        a, b = Z[:, i].copy(), Z[:, i + 1].copy()
        Z[:, i] = c * a - s * b; Z[:, i + 1] = s * a + c * b
    return Z


def _struct_score(v):
    z = (v - v.mean()) / (v.std() + 1e-9)
    return float(abs(np.mean(z ** 4) - 3.0))


def _sep_score(v, y, k=10):
    m = v.mean(); bet = wit = 0.0
    for c in range(k):
        vc = v[y == c]
        if len(vc):
            bet += len(vc) * (vc.mean() - m) ** 2; wit += ((vc - vc.mean()) ** 2).sum()
    return float(bet / (wit + 1e-9))


# ---- the infinite lens generator ------------------------------------------

def _program(idx, n_axes):
    """Deterministic lens program for address `idx`. Richness (axis order,
    composition depth, product terms) grows as the index deepens — so exploration
    keeps reaching new, more complex regions of the infinite space."""
    rng = np.random.default_rng(idx * 2654435761 % (2 ** 32))
    order = 2 + int(rng.integers(0, 1 + min(3, idx // 900)))        # 2..5 axes
    ax = rng.choice(n_axes, min(order, n_axes), replace=False)
    co = rng.standard_normal(len(ax)); co /= np.linalg.norm(co) + 1e-9
    depth = 1 + int(rng.integers(0, 1 + min(2, idx // 2500)))       # 1..3 composed acts
    acts = [ACT_NAMES[int(rng.integers(len(ACT_NAMES)))] for _ in range(depth)]
    prog = {"ax": ax.tolist(), "co": co.tolist(), "acts": acts}
    if idx > 1200 and rng.random() < 0.3:                          # product of two sub-lenses
        ax2 = rng.choice(n_axes, min(2, n_axes), replace=False)
        co2 = rng.standard_normal(len(ax2)); co2 /= np.linalg.norm(co2) + 1e-9
        prog["mul"] = {"ax": ax2.tolist(), "co": co2.tolist(),
                       "acts": [ACT_NAMES[int(rng.integers(len(ACT_NAMES)))]]}
    return prog


def _apply_prog(prog, Z):
    t = Z[:, prog["ax"]] @ np.array(prog["co"])
    for a in prog["acts"]:
        t = ACTS[a](t)
    if "mul" in prog:
        m = prog["mul"]; t2 = Z[:, m["ax"]] @ np.array(m["co"])
        for a in m["acts"]:
            t2 = ACTS[a](t2)
        t = t * t2
    return t


def _complexity(prog):
    c = len(prog["ax"]) + len(prog["acts"])
    if "mul" in prog:
        c += len(prog["mul"]["ax"]) + len(prog["mul"]["acts"])
    return c


def _pos(idx, prog):
    r = 0.16 + 0.11 * min(_complexity(prog), 9)
    th = idx * GOLD
    return round(float(r * np.cos(th)), 4), round(float(r * np.sin(th)), 4)


# ---- residual stacking ----------------------------------------------------

def _layer_apply(layer, Xin):
    Z = (Xin - layer["in_mu"]) / layer["in_sd"]
    Z = _rotate_frame(Z, layer.get("rot", 0))
    cols = [Z]
    for prog, lm, ls in zip(layer["progs"], layer["lens_mu"], layer["lens_sd"]):
        cols.append(((_apply_prog(prog, Z) - lm) / ls)[:, None])
    B = np.concatenate(cols, 1)
    out = (B - layer["bank_mean"]) @ layer["out_c"].T
    return (out - layer["out_mu"]) / layer["out_sd"]


def _apply_stack(tr, te):
    for layer in STACK:
        tr, te = _layer_apply(layer, tr), _layer_apply(layer, te)
    return tr, te


def _explore(n_axes, chk_every, throttle, rot_deg):
    d = _load()
    ytr, yte = d["ytr"], d["yte"]
    in_tr_raw, in_te_raw = _apply_stack(d["base_tr"], d["base_te"])
    Ztr, in_mu, in_sd = _zscore(in_tr_raw)
    Zte = (in_te_raw - in_mu) / in_sd
    Ztr = _rotate_frame(Ztr, rot_deg); Zte = _rotate_frame(Zte, rot_deg)   # step the frame
    n_axes = min(n_axes, Ztr.shape[1])
    ev = min(1500, len(Ztr)); evte = min(1000, len(Zte))
    depth = len(STACK) + 1

    with _LOCK:
        STATE.update({"depth": depth, "acts": ACT_NAMES, "names": d["names"],
                      "points": [], "checkpoints": [], "best": None, "baseline": None,
                      "history": list(HISTORY), "done": False, "stop": False})

    base = _fknn(Ztr[:ev, :K], ytr[:ev], Zte[:evte, :K], yte[:evte])
    with _LOCK:
        STATE["baseline"] = round(base, 4)

    # reservoir of lens responses for the encoder (unbiased sample of the stream)
    res_tr, res_te, res_prog, res_norm, res_struct = [], [], [], [], []
    rng = np.random.default_rng(depth)
    u8 = d["Xtr_u8"]; nu8 = len(u8)
    idx = 0
    while idx < SOFT_CAP:
        with _LOCK:
            if STATE.get("stop"):
                break
        prog = _program(idx, n_axes)
        vtr = _apply_prog(prog, Ztr); vte = _apply_prog(prog, Zte)
        s = vtr[:ev].std()
        if not np.isfinite(s) or s < 1e-6:
            idx += 1; continue
        struct = _struct_score(vtr); sep = _sep_score(vtr, ytr)
        px, py = _pos(idx, prog)
        rec = {"idx": idx, "x": px, "y": py, "struct": round(struct, 3),
               "sep": round(sep, 4), "cx": _complexity(prog)}
        mm = vtr[:ev].mean()
        col_tr = (vtr[:ev] - mm) / s; col_te = (vte[:evte] - mm) / s
        # reservoir sampling -> bounded encoder bank
        if len(res_tr) < BANK_CAP:
            res_tr.append(col_tr); res_te.append(col_te); res_prog.append(prog)
            res_norm.append((mm, s)); res_struct.append(struct)
        else:
            j = int(rng.integers(idx + 1))
            if j < BANK_CAP:
                res_tr[j] = col_tr; res_te[j] = col_te; res_prog[j] = prog
                res_norm[j] = (mm, s); res_struct[j] = struct
        with _LOCK:
            STATE["points"].append(rec)
            b = STATE["best"]
            if b is None or struct > b["struct"]:
                order = np.argsort(vtr[:nu8])
                strip = [u8[k].tolist() for k in list(order[-8:][::-1]) + list(order[:8])]
                STATE["best"] = {**rec, "strip": strip, "prog": _prog_str(prog)}
        idx += 1
        if idx % chk_every == 0 and len(res_tr) >= K:
            Btr = np.concatenate([Ztr[:ev], np.stack(res_tr, 1)], 1)
            Bte = np.concatenate([Zte[:evte], np.stack(res_te, 1)], 1)
            c = np.linalg.svd(Btr - Btr.mean(0), full_matrices=False)[2][:K]
            acc = _fknn(Btr @ c.T, ytr[:ev], Bte @ c.T, yte[:evte])
            with _LOCK:
                STATE["checkpoints"].append({"n": idx, "acc": round(acc, 4)})
        if throttle:
            time.sleep(throttle)

    # freeze the layer from the reservoir (residual)
    Btr = np.concatenate([Ztr[:ev], np.stack(res_tr, 1)], 1); bank_mean = Btr.mean(0)
    c = np.linalg.svd(Btr - bank_mean, full_matrices=False)[2][:K]
    _, omu, osd = _zscore((Btr - bank_mean) @ c.T)
    layer = {"in_mu": in_mu, "in_sd": in_sd, "rot": rot_deg, "progs": res_prog,
             "lens_mu": np.array([n[0] for n in res_norm]), "lens_sd": np.array([n[1] for n in res_norm]),
             "bank_mean": bank_mean, "out_c": c, "out_mu": omu, "out_sd": osd}
    best_acc = max((cp["acc"] for cp in STATE["checkpoints"]), default=0.0)
    with _LOCK:
        STACK.append(layer)
        HISTORY.append({"depth": depth, "best": round(best_acc, 4), "baseline": STATE["baseline"]})
        STATE["history"] = list(HISTORY); STATE["done"] = True


def _prog_str(prog):
    inner = f"[{'+'.join(str(a) for a in prog['ax'])}]"
    s = "∘".join(prog["acts"]) + inner
    if "mul" in prog:
        s += " × " + prog["mul"]["acts"][0] + f"[{'+'.join(str(a) for a in prog['mul']['ax'])}]"
    return s


def start(n_axes=16, extend=False, rot_deg=0.0):
    with _LOCK:
        if STATE.get("running") and not STATE.get("done", True):
            return {"error": "exploration already running"}
    _load()
    if not extend:
        STACK.clear(); HISTORY.clear()
    with _LOCK:
        STATE.clear(); STATE["running"] = True
    n_axes = max(4, min(BASE_DIM, int(n_axes)))
    t = threading.Thread(target=_explore, args=(n_axes, 300, 0.003, float(rot_deg)), daemon=True)
    t.start()
    for _ in range(140):
        with _LOCK:
            if STATE.get("depth") is not None:
                return {k: STATE.get(k) for k in ("depth", "acts", "names")}
        time.sleep(0.05)
    return {"error": "explorer did not start"}


def stop():
    with _LOCK:
        STATE["stop"] = True
    return {"stopping": True}


def poll(since=0):
    with _LOCK:
        P = STATE.get("points", [])
        return {"new": P[since:], "count": len(P), "cap": SOFT_CAP,
                "depth": STATE.get("depth", 1), "history": STATE.get("history", []),
                "checkpoints": STATE.get("checkpoints", []), "best": STATE.get("best"),
                "baseline": STATE.get("baseline"), "done": STATE.get("done", False)}


if __name__ == "__main__":
    start(n_axes=14)
    time.sleep(8); stop()
    while not poll(0)["done"]:
        time.sleep(0.2)
    p = poll(0)
    print("explored", p["count"], "checkpoints", [(c["n"], c["acc"]) for c in p["checkpoints"]])
    print("baseline", p["baseline"], "best lens", p["best"]["prog"], "struct", p["best"]["struct"])
