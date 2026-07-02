"""Grab real dialogue from parquet-native HF datasets (no loading scripts). Dump all utterance text."""
import os
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__))
def log(s): print(s, flush=True)

OUT = os.path.join(HERE, "dialog_extra.txt")
total = 0
f = open(OUT, "w", encoding="utf-8")

def emit_dialogue(turns):                                    # one dialogue per line, turns kept in order
    global total
    ts = [" ".join(t.split()) for t in turns if isinstance(t, str) and len(t.split()) >= 1]
    if len(ts) >= 2:
        line = "  ".join(ts); f.write(line + "\n"); total += line.count(" ") + 1


def best_turnlist(r):                                        # find the dialogue (longest list of strings)
    best = []
    for v in r.values():
        if isinstance(v, list):
            ss = [x for x in v if isinstance(x, str)] or [x.get("text", "") for x in v if isinstance(x, dict)]
            ss = [x for x in ss if x and x.strip()]
            if len(ss) > len(best): best = ss
    return best

CANDS = [
    ("Estwld/empathetic_dialogues_llm", None),
    ("li2017dailydialog/daily_dialog", None),
    ("google/Synthetic-Persona-Chat", None),
    ("facebook/empathetic_dialogues", None),
    ("blended_skill_talk", None),
    ("Cynaptics/persona-chat", None),
]
for name, cfg in CANDS:
    try:
        ds = load_dataset(name, cfg, split="train") if cfg else load_dataset(name, split="train")
        c0 = total
        for r in ds:
            emit_dialogue(best_turnlist(r))
        log(f"OK {name}: +{total-c0:,} words")
    except Exception as e:
        log(f"skip {name}: {str(e)[:90]}")
f.close()
log(f"TOTAL dialogue words: {total:,}")
