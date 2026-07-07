"""Pre-train the demo genomes -> demo/genomes.pkl (so the pygame demo starts
instantly), then print a sample at each stack level to verify the layers differ."""
import os, sys, time
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec = importlib.util.spec_from_file_location("demo", os.path.join(os.path.dirname(__file__), "demo.py"))
demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)

LOG = os.path.join(os.path.dirname(__file__), "build_cache.log")
open(LOG, "w").close()
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

eng = demo.Engine()
t = time.time(); eng.run()   # builds caches, trains (no cache), saves genomes.pkl
log(f"train+cache done in {time.time()-t:.0f}s  err={eng.err}")
eng.ready = True
levels = [
    ("nothing",        {"vocab": False, "order": False, "sel": "off", "bound": False}),
    ("+Vocabulary",    {"vocab": True,  "order": False, "sel": "off", "bound": False}),
    ("+Order",         {"vocab": True,  "order": True,  "sel": "off", "bound": False}),
    ("+Selection",     {"vocab": True,  "order": True,  "sel": "uni", "bound": False}),
    ("+Bidirectional", {"vocab": True,  "order": True,  "sel": "bi",  "bound": False}),
    ("+Boundary",      {"vocab": True,  "order": True,  "sel": "bi",  "bound": True}),
]
for name, en in levels:
    log(f"{name:>15}: {eng.generate(en, n=70, seed=3)[:180]}")
log("DONE.")
