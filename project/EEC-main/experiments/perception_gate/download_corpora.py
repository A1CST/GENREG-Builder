"""Pull MASSIVE real corpora: WikiText-103 (perception substrate) + DailyDialog (real dialogue)."""
import os
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__))
def log(s): print(s, flush=True)

log("downloading WikiText-103 (this is the big one)...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
n = 0
with open(os.path.join(HERE, "wiki_corpus.txt"), "w", encoding="utf-8") as f:
    for r in ds:
        t = r["text"].strip()
        if t:
            f.write(t + "\n"); n += t.count(" ") + 1
log(f"WikiText-103 written: ~{n:,} words")

for name, cfg in [("daily_dialog", None), ("li2017dailydialog/DailyDialog", None)]:
    try:
        dd = load_dataset(name, split="train", trust_remote_code=True)
        with open(os.path.join(HERE, "dialog_extra.txt"), "w", encoding="utf-8") as f:
            for r in dd:
                turns = r.get("dialog") or r.get("utterances") or []
                if turns: f.write("  ".join(u.strip() for u in turns) + "\n")
        log(f"dialogue corpus from {name}: ok"); break
    except Exception as e:
        log(f"{name} failed: {str(e)[:120]}")
log("DONE")
