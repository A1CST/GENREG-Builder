"""topic_steer.py - TOPIC-STEERED generation: the persistence topic state
drives word choice in the live word-level generator.

The goal (user's directive): a model that responds and HOLDS THE TOPIC.
Composition of two frozen models, no retraining:
  - the live autocomplete checkpoint (lm_model_word.json via lm_word_infer:
    W=16 context, V=2000 targets, replayed frozen, closed-form head), and
  - the persistence topic model (kid_plang_model.json: 631 accumulated-
    response detectors + ridge head over 8 wiki topics, module 32).

Steering: because the topic head is LINEAR over accumulated detector
responses, each target word w has a fixed per-topic contribution
s_w[t] = head(responses(w)). At decode time the prompt's accumulated state
picks the topic t*, and lambda * z(s[:, t*]) is added to the generator's
logits BEFORE top-5 selection - so topical words can enter the candidate
pool, which pure local statistics would never surface.

Honest evaluation (the steering model must NOT be the judge):
  - topic-hold judge = add-alpha content-word log-odds computed from the
    HELD-OUT test articles (corpus counts, no learned model): does the
    continuation classify to the prompt's topic?
  - next-word sanity: top-1 on held-out lm_word.npz windows with steering
    conditioned on each window's own state - coherence must not crater.
  - samples verbatim at every lambda.

  python topic_steer.py [--smoke]
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(_HERE, "radial_data")

LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_GEN = 24
TEMP = 0.9
PROMPTS = {
    "astronomy": ["the stars in the night sky", "the planets around the sun"],
    "chemistry": ["the acid and the gas in the water",
                  "the metal and the chemical elements"],
    "food":      ["i cooked the bread and the cheese",
                  "the wine and the meat for dinner"],
    "football":  ["the team won the game last night",
                  "he scored a goal in the match"],
    "geography": ["the river flows down the mountain",
                  "the ocean near the island coast"],
    "law":       ["the judge in the court room",
                  "the law and the police officer"],
    "medicine":  ["the doctor treated the sick patient",
                  "the disease spread through the blood"],
    "music":     ["she played the piano and sang",
                  "the band played a song on stage"],
}


def load_topic_model(torch, dev):
    from radial_evo import _tprims
    import radial_stack as rk
    tp = _tprims(torch)
    with open(os.path.join(RD, "kid_plang_model.json")) as f:
        tm = json.load(f)
    ze = np.load(os.path.join(RD, "embed_rs.npz"), allow_pickle=True)
    vocab = {str(w): i for i, w in enumerate(ze["vocab"])}
    E = torch.tensor(ze["feat"].astype(np.float32), device=dev)
    mu = torch.tensor(tm["embed_mu"], device=dev)
    sd = torch.tensor(tm["embed_sd"], device=dev)
    hm = torch.tensor(tm["head_mu"], device=dev)
    hs = torch.tensor(tm["head_sd"], device=dev)
    Wm = torch.tensor(tm["head_W"], device=dev)          # (n_gen+1, 8)

    def responses(words):
        """word list -> (n_in_vocab, n_genomes) detector responses."""
        ids = [vocab[w] for w in words if w in vocab]
        if not ids:
            return None
        z = ((E[torch.tensor(ids, device=dev)] - mu) / sd).clamp(-8, 8)
        cols = [torch.nan_to_num(rk.feature_vec(torch, tp, z, g),
                                 nan=0.0, posinf=0.0, neginf=0.0)
                .clamp(-1e6, 1e6) for g in tm["genomes"]]
        return torch.stack(cols, 1)

    def topic_probs(words):
        """accumulated state -> topic distribution (softmax of head logits)."""
        r = responses(words)
        if r is None:
            return None
        acc = r.mean(0, keepdim=True)
        lg = torch.hstack([(acc - hm) / hs,
                           torch.ones(1, 1, device=dev)]) @ Wm
        return torch.softmax(lg[0], 0)

    return tm, responses, topic_probs, hm, hs, Wm


def target_topic_scores(torch, dev, responses, hm, hs, Wm, targets):
    """(V, 8) per-target-word topic contributions, z-scored per topic so
    lambda is in comparable units. Out-of-embed-vocab targets score 0."""
    S = torch.zeros((len(targets), Wm.shape[1]), device=dev)
    r = responses(targets)                    # in-vocab rows only, in order
    ze = np.load(os.path.join(RD, "embed_rs.npz"), allow_pickle=True)
    vocab = {str(w): i for i, w in enumerate(ze["vocab"])}
    rows = [k for k, w in enumerate(targets) if w in vocab]
    S[torch.tensor(rows, device=dev)] = ((r - hm) / hs) @ Wm[:-1]
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    return S


def build_judge(topics):
    """Held-out corpus judge: add-alpha content-word log-probs per topic
    from each topic's TWO TEST articles (never seen by generator, steering
    model, or topic stream training). Counts are cached to json so the judge
    runs on machines without the local wiki dump (pods)."""
    from collections import Counter
    cache = os.path.join(RD, "topic_judge_counts.json")
    if os.path.exists(cache):
        with open(cache) as f:
            cnt = {t: Counter(c) for t, c in json.load(f).items()}
    else:
        import zetifile
        from radial_lm import _clean
        from build_topic_stream import TOPICS
        cnt = {}
        for t in topics:
            c = Counter()
            for title in TOPICS[t][6:]:
                _, text = zetifile.page_text(title)
                if text:
                    c.update(_clean(text).split())
            cnt[t] = c
        with open(cache, "w") as f:
            json.dump({t: dict(c) for t, c in cnt.items()}, f)
    glob = Counter()
    for c in cnt.values():
        glob.update(c)
    stop = {w for w, _ in glob.most_common(150)}
    V = len(glob)

    def judge(text):
        """-> (per-topic mean content-word log-odds, argmax topic)."""
        words = [w for w in text.split() if w not in stop and len(w) > 2]
        if not words:
            return None, None
        out = {}
        for t in topics:
            n = sum(cnt[t].values())
            lp = [np.log((cnt[t][w] + 0.5) / (n + 0.5 * V)) -
                  np.log((glob[w] + 0.5) / (sum(glob.values()) + 0.5 * V))
                  for w in words]
            out[t] = float(np.mean(lp))
        return out, max(out, key=out.get)

    return judge


def main(smoke=False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log_lines = []

    def log(m):
        log_lines.append(m); print(m, flush=True)

    log("[steer] building the word generator (frozen replay)...")
    import lm_word_infer as lwi
    lwi._build()                              # blocking build
    M = lwi._M
    step, w2i, targets = M["step"], M["w2i"], M["targets"]
    W, V, s_cal = M["W"], M["V"], M["s_cal"]
    log(f"[steer] generator ready: W={W} V={V} val {M['val_acc']:.4f} "
        f"({M['build_seconds']}s build)")

    tm, responses, topic_probs, hm, hs, Wm = load_topic_model(torch, dev)
    topics = tm["topics"]
    S = target_topic_scores(torch, dev, responses, hm, hs, Wm, targets)
    log(f"[steer] topic model: {len(tm['genomes'])} detectors, "
        f"{len(topics)} topics; S matrix {tuple(S.shape)}")
    judge = build_judge(topics)

    def generate(prompt, lam, seed=0):
        rng = np.random.default_rng(seed)
        words = [w for w in prompt.lower().split() if w]
        win = [w2i.get(w, -1) for w in words][-W:]
        while len(win) < W:
            win.insert(0, -1)
        ctx = list(words)                     # topic state reads ALL words
        p = topic_probs(ctx)
        t_star = int(p.argmax()) if p is not None else None
        out = []
        for _ in range(N_GEN):
            lg = step(win).detach()
            lg = lg * s_cal / TEMP
            if lam > 0 and t_star is not None:
                # steer toward NEW topical words: an already-emitted word
                # gets no bonus, so exhausted topic vocabulary cannot loop
                bonus = (lam * S[:, t_star]).clone()
                for wd in out:
                    kr = M["tgt_i"].get(wd)
                    if kr is not None:
                        bonus[kr] = 0.0
                lg = lg + bonus
            lg = lg.cpu().numpy().astype(np.float64)
            # repetition penalty scales with steering so it can't be outbid
            for wd in out[-16:]:
                kr = M["tgt_i"].get(wd)
                if kr is not None:
                    lg[kr] -= 2.0 + lam
            top = np.argsort(lg)[-5:]
            z = lg[top] - lg[top].max()
            pr = np.exp(z); pr /= pr.sum()
            k = int(rng.choice(top, p=pr))
            wd = targets[k]
            out.append(wd)
            ctx.append(wd)
            win = win[1:] + [w2i.get(wd, -1)]
        return " ".join(out), (topics[t_star] if t_star is not None else "?")

    # ---- generation sweep ----
    res = {"lambdas": LAMBDAS, "n_gen": N_GEN, "topics": list(topics),
           "samples": [], "hold": {}, "sanity": {}}
    lams = LAMBDAS[:3] if smoke else LAMBDAS
    for lam in lams:
        held = 0; total = 0
        for topic, prompts in PROMPTS.items():
            for pi, prompt in enumerate(prompts if not smoke else prompts[:1]):
                comp, t_star = generate(prompt, lam, seed=17 + pi)
                _, jt = judge(comp)
                ok = jt == topic
                held += int(ok); total += 1
                res["samples"].append({"lam": lam, "topic": topic,
                                       "state_topic": t_star,
                                       "prompt": prompt, "completion": comp,
                                       "judge": jt, "held": ok})
                log(f"  lam={lam:<4} [{topic:<10}] {prompt!r} -> {comp!r} "
                    f"(state {t_star}, judge {jt}{' OK' if ok else ''})")
        res["hold"][str(lam)] = round(held / total, 4)
        log(f"[steer] lam={lam}: topic-hold {held}/{total} = {held / total:.3f}")

    # ---- held-out next-word sanity ----
    z = np.load(os.path.join(RD, "lm_word.npz"), allow_pickle=True)
    ctx_te, yte = z["ctx_te"], z["yte"]
    n_chk = 150 if smoke else 400
    idx = np.random.default_rng(3).choice(len(yte), n_chk, replace=False)
    inv_vocab = {i: w for w, i in w2i.items()}
    for lam in lams:
        hit = 0
        for i in idx:
            wids = ctx_te[i]
            words = [inv_vocab.get(int(j)) for j in wids if int(j) >= 0]
            words = [w for w in words if w]
            p = topic_probs(words) if words else None
            t_star = int(p.argmax()) if p is not None else None
            lg = step(list(wids)).detach()
            lg = lg * s_cal
            if lam > 0 and t_star is not None:
                lg = lg + lam * S[:, t_star]
            hit += int(int(lg.argmax()) == int(yte[i]))
        res["sanity"][str(lam)] = round(hit / n_chk, 4)
        log(f"[steer] lam={lam}: held-out top-1 {hit}/{n_chk} = "
            f"{hit / n_chk:.4f}")

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "topic_steer_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("STEER RESULT: hold " + json.dumps(res["hold"]) +
        " sanity " + json.dumps(res["sanity"]))
    print("STEER DONE", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
