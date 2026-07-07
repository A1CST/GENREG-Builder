"""WordPipe visual demo — watch the language build up genome by genome.

A pygame window that (1) trains each specialist genome live, showing its fitness
curve emerge, and (2) lets you toggle each layer of the stack on/off to SEE how
the generated text transforms as capabilities are added:

    (nothing)      -> random letters (no vocabulary)
    + Vocabulary   -> real words, random order
    + Order        -> words follow a grammatical class skeleton
    + Selection    -> context-fit word choice (prev word), or Bidirectional (both neighbours)
    + Boundary     -> real sentences (periods, capitals)

Run:  python demo/demo.py       (first run trains ~4 min, then caches to demo/genomes.pkl)
"""
import os
import pickle
import re
import sys
import threading

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genreg_train import wordpipe as wp   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "genomes.pkl")
NCL, C, D = 32, 4, 24
GENS = {"order": 1000, "sel": 900, "bisel": 900, "bound": 600, "comma": 600}   # first-run training

# ---- theme ---------------------------------------------------------------
BG, PANEL, LINE = (14, 17, 22), (22, 27, 34), (40, 48, 58)
FG, MUTED, ACC = (222, 230, 238), (128, 140, 152), (90, 170, 255)
GOOD, WARN = (90, 200, 130), (230, 180, 90)


class Engine:
    """Backend: builds caches, trains the genomes (with live progress), generates."""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "starting"
        self.layers = {k: {"status": "queued", "curve": [], "metric": None}
                       for k in ("order", "sel", "bisel", "bound", "comma")}
        self.champs = {}
        self.ready = False
        self.busy = False
        self.err = None

    # -- generation data (built once) --
    def _build(self):
        self.phase = "loading corpus + classes (one-time)…"
        wp.build_word_corpus(4000); wp.induce_word_classes(NCL)
        self.table, self.w2c, self.vocab, self.nc, self.cids = wp.build_class_words(NCL)
        self.feat, _ = wp.word_features(4000, D)
        self.cents = wp.class_centroids(NCL, D)
        ids, _, _ = wp.build_word_corpus(4000)
        self.ids = ids
        self.logfreq = np.log1p(np.bincount(ids, minlength=len(self.vocab)).astype(np.float32))
        self.cfreq = np.bincount(self.cids, minlength=self.nc).astype(np.float64)
        self.cprob = self.cfreq / self.cfreq.sum()

    def _logger(self, key, metric_re):
        def logf(*args):
            line = " ".join(str(a) for a in args)
            g = re.search(r"gen (\d+)", line)
            m = re.search(metric_re, line)
            if g and m:
                with self.lock:
                    self.layers[key]["curve"].append((int(g.group(1)), float(m.group(1))))
                    self.layers[key]["metric"] = float(m.group(1))
        return logf

    def run(self):
        try:
            self._build()
            if os.path.exists(CACHE):
                self.phase = "loading cached genomes…"
                with open(CACHE, "rb") as f:
                    self.champs = pickle.load(f)
                for k in self.layers:
                    self.layers[k]["status"] = "ready"
            else:
                self._train()
            self.phase = "ready"
            self.ready = True
        except Exception as exc:                       # pragma: no cover
            import traceback; traceback.print_exc()
            self.err = f"{type(exc).__name__}: {exc}"
            self.phase = "error"

    def _train(self):
        self.phase = "training ORDER genome…"
        self.layers["order"]["status"] = "training"
        r = wp.run_class_lm(NCL, gens=GENS["order"], pop=200, C=C, E=10, H=64,
                            seed=7, log=self._logger("order", r"val_ppl=(-?\d+\.?\d*)"))
        self.champs["order"] = r["champ"]; self.layers["order"]["status"] = "ready"

        self.phase = "training SELECTION genome…"
        self.layers["sel"]["status"] = "training"
        r = wp.run_selection(NCL, gens=GENS["sel"], pop=200, D=D, K=7, seed=7,
                             log=self._logger("sel", r"val_logprob=(-?\d+\.?\d*)"))
        self.champs["sel"] = r["champ"]; self.layers["sel"]["status"] = "ready"

        self.phase = "training BIDIRECTIONAL selection…"
        self.layers["bisel"]["status"] = "training"
        r = wp.run_biselection(NCL, gens=GENS["bisel"], pop=200, D=D, K=7, seed=7,
                               log=self._logger("bisel", r"val_logprob=(-?\d+\.?\d*)"))
        self.champs["bisel"] = r["champ"]; self.layers["bisel"]["status"] = "ready"

        self.phase = "training BOUNDARY genome…"
        self.layers["bound"]["status"] = "training"
        r = wp.run_boundary(NCL, gens=GENS["bound"], pop=200, seed=7,
                            log=self._logger("bound", r"val_logprob=(-?\d+\.?\d*)"))
        self.champs["bound"] = r["champ"]; self.layers["bound"]["status"] = "ready"

        self.phase = "training COMMA genome…"
        self.layers["comma"]["status"] = "training"
        r = wp.run_comma(NCL, gens=GENS["comma"], pop=200, seed=7,
                         log=self._logger("comma", r"val_logprob=(-?\d+\.?\d*)"))
        self.champs["comma"] = r["champ"]; self.layers["comma"]["status"] = "ready"

        with open(CACHE, "wb") as f:
            pickle.dump(self.champs, f)

    def retrain(self):
        """Re-run training live (watch the fitness curves emerge). Runs in a
        thread; old champions keep working until new ones replace them."""
        if self.busy:
            return
        def _go():
            self.busy = True
            with self.lock:
                for k in self.layers:
                    self.layers[k].update(status="queued", curve=[], metric=None)
            try:
                self._train()
                self.phase = "ready"
            except Exception as exc:                    # pragma: no cover
                self.err = str(exc)
            self.busy = False
        threading.Thread(target=_go, daemon=True).start()

    # -- compose the enabled stack into text --
    def generate(self, en, n=280, seed=0):
        if not self.ready:
            return "…"
        rng = np.random.default_rng(seed)
        if en["order"] and "order" in self.champs:
            cls_seq = wp.gen_class_seq(self.champs["order"], C, n, self.cids[500:500 + C], rng, 0.8)
        else:
            cls_seq = list(rng.choice(self.nc, size=n, p=self.cprob))
        parts, prev, cur, clause = [], int(self.ids[499]), 0, 0
        for j, cl in enumerate(cls_seq):
            if cl not in self.table:
                continue
            mem = self.table[cl][0]
            if not en["vocab"]:
                # no vocabulary genome -> random non-word letters of a plausible length
                L = int(rng.integers(2, 9))
                parts.append("".join(chr(rng.integers(97, 123)) for _ in range(L)))
            else:
                if en["sel"] == "bi" and "bisel" in self.champs:
                    nxt = next((cls_seq[k] for k in range(j + 1, len(cls_seq))
                                if cls_seq[k] in self.table), cl)
                    w = wp._fill_bisel(prev, cl, nxt, self.table, self.feat, self.logfreq,
                                       self.cents, self.champs["bisel"], rng)
                elif en["sel"] == "uni" and "sel" in self.champs:
                    w = wp._fill_selected(prev, cl, self.table, self.feat, self.logfreq,
                                          self.champs["sel"], rng)
                else:
                    w = int(rng.choice(mem, p=self.table[cl][1]))
                parts.append(self.vocab[w]); prev = w
            cur += 1; clause += 1
            if en["vocab"]:
                if en["bound"] and "bound" in self.champs and (cur >= 45
                        or rng.random() < wp.boundary_prob(self.champs["bound"], cl, cur)):
                    parts.append("."); cur = 0; clause = 0
                elif en.get("commas") and "comma" in self.champs and clause >= 3 \
                        and rng.random() < wp.boundary_prob(self.champs["comma"], cl, clause):
                    parts.append(","); clause = 0
        text = " ".join(parts).replace(" .", ".").replace(" ,", ",")
        return re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


# ==========================================================================
# UI
# ==========================================================================
def wrap(text, font, width):
    lines, cur = [], ""
    for word in text.split(" "):
        t = (cur + " " + word).strip()
        if font.size(t)[0] <= width:
            cur = t
        else:
            lines.append(cur); cur = word
    if cur:
        lines.append(cur)
    return lines


def sparkline(surf, rect, pts, color, invert=False):
    x, y, w, h = rect
    pygame.draw.rect(surf, (10, 13, 17), rect)
    pygame.draw.rect(surf, LINE, rect, 1)
    if len(pts) < 2:
        return
    ys = [p[1] for p in pts]
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        hi = lo + 1
    n = len(pts)
    poly = []
    for i, (_, v) in enumerate(pts):
        px = x + 2 + (w - 4) * i / (n - 1)
        norm = (v - lo) / (hi - lo)
        if invert:
            norm = 1 - norm
        py = y + 2 + (h - 4) * (1 - norm)
        poly.append((px, py))
    pygame.draw.lines(surf, color, False, poly, 2)


LAYERS = [
    ("vocab", "Vocabulary", "emit real words instead of letter noise", "toggle"),
    ("order", "Order", "words follow a grammatical class skeleton", "toggle"),
    ("sel", "Selection", "context-fit word choice", "tri"),
    ("bound", "Boundary", "cut the stream into sentences", "toggle"),
    ("commas", "Commas", "internal punctuation (P(comma) per class + clause pos)", "toggle"),
]


def main():
    pygame.init()
    W, H = 1180, 760
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("WordPipe — watch the language evolve, genome by genome")
    clock = pygame.time.Clock()
    f_big = pygame.font.SysFont("consolas", 26, bold=True)
    f_h = pygame.font.SysFont("consolas", 18, bold=True)
    f = pygame.font.SysFont("consolas", 16)
    f_sm = pygame.font.SysFont("consolas", 13)
    f_txt = pygame.font.SysFont("georgia", 19)

    eng = Engine()
    threading.Thread(target=eng.run, daemon=True).start()

    en = {"vocab": True, "order": True, "sel": "bi", "bound": True, "commas": True}
    seed = [3]
    rects = {}

    def regen():
        return eng.generate(en, seed=seed[0])
    cached_text = [""]
    last_key = [None]

    running = True
    while running:
        # regenerate text when the stack or seed changes (and engine ready)
        key = (tuple(sorted((k, str(v)) for k, v in en.items())), seed[0], eng.ready)
        if key != last_key[0]:
            cached_text[0] = regen()
            last_key[0] = key

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                seed[0] += 1
            elif ev.type == pygame.MOUSEBUTTONDOWN and eng.ready:
                if "retrain" in rects and rects["retrain"].collidepoint(ev.pos):
                    eng.retrain()
                for key_, r in rects.items():
                    if key_ == "retrain" or not r.collidepoint(ev.pos):
                        continue
                    if key_ == "sel":
                        order = ["off", "uni", "bi"]
                        en["sel"] = order[(order.index(en["sel"]) + 1) % 3]
                    else:
                        en[key_] = not en[key_]

        screen.fill(BG)
        # header
        screen.blit(f_big.render("WordPipe", True, FG), (28, 22))
        screen.blit(f.render("watch the language build up, one evolved genome at a time",
                             True, MUTED), (30, 56))
        with eng.lock:
            phase = eng.phase
        pc = WARN if not eng.ready else GOOD
        screen.blit(f_sm.render(("● " + phase) if not eng.err else ("error: " + eng.err),
                                True, (230, 110, 110) if eng.err else pc), (W - 430, 30))
        # retrain button (watch the fitness curves emerge live)
        rects.clear()
        if eng.ready and not eng.busy:
            btn = pygame.Rect(W - 250, 52, 220, 26)
            pygame.draw.rect(screen, (28, 34, 42), btn, border_radius=6)
            pygame.draw.rect(screen, ACC, btn, 1, border_radius=6)
            screen.blit(f_sm.render("RETRAIN — watch it evolve", True, ACC), (btn.x + 16, btn.y + 6))
            rects["retrain"] = btn

        # ---- left: the genome stack ----
        px, pw = 28, 380
        pygame.draw.rect(screen, PANEL, (px, 92, pw, H - 120), border_radius=8)
        screen.blit(f_h.render("GENOME STACK", True, FG), (px + 16, 106))
        y = 140
        stmap = {"order": "order", "sel": "sel", "bound": "bound", "commas": "comma"}
        for lkey, name, desc, kind in LAYERS:
            box = pygame.Rect(px + 16, y, pw - 32, 118)
            on = (en[lkey] not in (False, "off"))
            pygame.draw.rect(screen, (28, 34, 42) if on else (18, 22, 28), box, border_radius=6)
            pygame.draw.rect(screen, ACC if on else LINE, box, 1, border_radius=6)
            # toggle chip
            chip = pygame.Rect(box.x + 12, box.y + 12, 54, 24)
            val = en[lkey]
            label = {True: "ON", False: "OFF"}.get(val, str(val).upper())
            col = GOOD if on else MUTED
            pygame.draw.rect(screen, col, chip, 1, border_radius=12)
            screen.blit(f_sm.render(label, True, col), (chip.x + 8, chip.y + 5))
            rects[lkey] = box
            screen.blit(f_h.render(name, True, FG), (box.x + 78, box.y + 10))
            for i, ln in enumerate(wrap(desc, f_sm, pw - 100)):
                screen.blit(f_sm.render(ln, True, MUTED), (box.x + 78, box.y + 34 + i * 15))
            # training curve + metric for trainable layers
            if lkey in stmap:
                st = eng.layers[stmap[lkey]]
                spark = pygame.Rect(box.x + 12, box.y + 70, pw - 56, 34)
                inv = (stmap[lkey] == "order")   # ppl: lower is better -> invert
                with eng.lock:
                    sparkline(screen, spark, list(st["curve"]), ACC, invert=inv)
                    stat = st["status"]; met = st["metric"]
                mtxt = f"{stat}" + (f"  ·  {met:.3f}" if met is not None else "")
                screen.blit(f_sm.render(mtxt, True, GOOD if stat == "ready" else WARN),
                            (box.x + 12, box.y + 70 - 16))
            y += 130

        # ---- right: the generated text ----
        tx, tw = 428, W - 428 - 28
        pygame.draw.rect(screen, PANEL, (tx, 92, tw, H - 120), border_radius=8)
        stack = []
        if en["vocab"]:
            stack.append("Vocabulary")
        if en["order"]:
            stack.append("Order")
        if en["sel"] == "uni":
            stack.append("Selection(prev)")
        elif en["sel"] == "bi":
            stack.append("Selection(both)")
        if en["bound"] and en["vocab"]:
            stack.append("Boundary")
        screen.blit(f_h.render("OUTPUT  ·  " + (" + ".join(stack) if stack else "nothing"),
                               True, ACC), (tx + 16, 106))
        area = pygame.Rect(tx + 16, 140, tw - 32, H - 210)
        for i, ln in enumerate(wrap(cached_text[0], f_txt, area.w)):
            if 140 + i * 27 > H - 90:
                break
            screen.blit(f_txt.render(ln, True, FG), (area.x, 140 + i * 27))
        screen.blit(f_sm.render("click a layer to toggle it  ·  SPACE = regenerate  ·  "
                                "first run trains ~4 min then caches", True, MUTED),
                    (tx + 16, H - 44))

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
