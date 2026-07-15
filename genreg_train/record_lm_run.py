"""One-off: turn the fetched genreg_train/run_lm_intent.py log + artifact into
runs/lm/<run_id>/ folders — ONE per binary punctuation genome — so each shows
up on the /runs dashboard, grouped under "punctuation" via meta.json (the
dashboard's label/favorite/group/tags side-channel; see runstore.py). Not a
job driver — run this locally after fetching corpora/combined/lm_intent.pkl
and .log back from the I2 primary.
"""
import datetime
import json
import os
import pickle
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "corpora", "combined", "lm_intent.log")
ARTIFACT = os.path.join(ROOT, "corpora", "combined", "lm_intent.pkl")
RUNS_DIR = os.path.join(ROOT, "runs")

SPLIT_HEADER_RE = re.compile(r"=== training (\S+) \(")
# fitness field is soft-fit (mean log-prob, can be negative) since the
# GENREG_RULES §IV.1 soft-fitness switch; starved is the §III energy band.
GEN_RE_BALANCED = re.compile(
    r"gen\s+(\d+)\s+soft-fit=(-?[\d.]+)\s+starved=\d+\s+"
    r"holdout-balanced-acc=([\d.]+)\s+holdout-raw-acc=([\d.]+)")
GEN_RE_PLAIN = re.compile(
    r"gen\s+(\d+)\s+soft-fit=(-?[\d.]+)\s+starved=\d+\s+holdout-acc=([\d.]+)")


def per_split_logs(log_text):
    """Split the shared log into per-genome sections keyed by split name."""
    sections = {}
    parts = SPLIT_HEADER_RE.split(log_text)
    # parts = [preamble, key1, body1, key2, body2, ...]
    for i in range(1, len(parts), 2):
        sections[parts[i]] = parts[i + 1]
    return sections


def main():
    with open(LOG, "r", encoding="utf-8") as fh:
        log_text = fh.read()
    with open(ARTIFACT, "rb") as fh:
        art = pickle.load(fh)

    sections = per_split_logs(log_text)
    ts = datetime.datetime.now()
    stamp = ts.strftime("%Y%m%d-%H%M%S")

    written = []
    for key, split_result in art["splits"].items():
        is_contrastive = "holdout_balanced_acc" not in split_result
        history = []
        section = sections.get(key, "")
        if is_contrastive:
            for m in GEN_RE_PLAIN.finditer(section):
                gen, train_acc, acc = int(m.group(1)), float(m.group(2)), float(m.group(3))
                history.append({"gen": gen, "fitness": train_acc, "best": {"score": acc}})
        else:
            for m in GEN_RE_BALANCED.finditer(section):
                gen, train_acc, bal_acc, raw_acc = (int(m.group(1)), float(m.group(2)),
                                                    float(m.group(3)), float(m.group(4)))
                history.append({"gen": gen, "fitness": train_acc,
                                "best": {"score": bal_acc, "base": raw_acc}})

        rid = f"{stamp}-lm-{key}"
        d = os.path.join(RUNS_DIR, "lm", rid)
        os.makedirs(d, exist_ok=True)

        headline = split_result.get("holdout_balanced_acc", split_result.get("holdout_acc"))
        cfg = {
            "id": rid, "environment": "lm", "created": ts.isoformat(timespec="seconds"),
            "config": {
                "environment": "lm", "genome": key, "group": split_result["group"],
                "desc": split_result["desc"], "population": 120, "generations": 250,
                "device": "cpu (I2 primary)",
                "constraints": ["ctx_k=6"] + (
                    ["D=24, contrastive (true word vs 5 negatives)"] if is_contrastive
                    else ["D=16", "H=24", "binary, class-balanced",
                         "champion selection uses BALANCED holdout accuracy"]),
                "corpus": "corpora/combined/combined_corpus.txt (417MB, Wikipedia + Cornell Movie Dialogs)",
                "vocab_size": len(art["vocab"]),
            },
            "started": {"notes": split_result["desc"]},
            "status": "finished",
        }
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        with open(os.path.join(d, "history.jsonl"), "w", encoding="utf-8") as fh:
            for rec in history:
                fh.write(json.dumps(rec) + "\n")

        summary = {
            "id": rid, "environment": "lm", "status": "finished",
            "finished": ts.isoformat(timespec="seconds"),
            "gen": history[-1]["gen"] if history else None,
            "best": {"score": headline,
                    "base": split_result.get("holdout_raw_acc")},
            "n_examples": split_result["n_examples"],
            "checkpoint": None,   # artifact lives at corpora/combined/lm_intent.pkl, not
                                   # an engine_api-format checkpoint — see /api/lm/status
        }
        if not is_contrastive:
            summary["recall"] = split_result["recall"]
            summary["confusion"] = split_result["confusion"]
            summary["n_positive"] = split_result["n_positive"]
        with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        # group/tags side-channel the dashboard reads separately from config.json,
        # so each run is visibly grouped under its OWN genome group, not hardcoded.
        meta = {"label": key, "favorite": False, "group": split_result["group"], "tags": [key]}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        written.append((rid, split_result["group"], headline))
        print(f"wrote {d}")
        print(f"  group={split_result['group']}  headline-acc={headline:.4f}"
             + (f"  recall={split_result['recall']}" if not is_contrastive else ""))

    groups_seen = sorted({g for _, g, _ in written})
    print(f"\n{len(written)} genomes recorded across groups: {groups_seen}")


if __name__ == "__main__":
    main()
