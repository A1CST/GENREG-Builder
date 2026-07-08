"""Train sem_coh_local + sem_coh_theme (coherence.py) and run the decisive
probe for each: hold out real (centroid, next-word) pairs vs teleported-word
pairs the genome never trained on, and report accuracy on that held-out set
directly (this discriminator's natural probe IS its val_acc, unlike the
lexical genomes — there's no small hand-checkable word list for "does this
fit the local context", the context differs every time)."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "coherence.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import coherence as co

mined = co.mine_centroids()
log(f"mined {mined['n']} content-word occurrences, {len(mined['targets'])} training pairs "
   f"(local window={co.LOCAL_WIN}, theme window={co.THEME_WIN})")

res_local, _ = co.train_sem_coh_local(log=log, mined=mined)
log(f"\nsem_coh_local val_acc={res_local['val_acc']}")

res_theme, _ = co.train_sem_coh_theme(log=log, mined=mined)
log(f"\nsem_coh_theme val_acc={res_theme['val_acc']}")

log("\nPROBE verdicts (val_acc on held-out real-vs-teleported pairs, 0.5=chance):")
log(f"   sem_coh_local: {res_local['val_acc']}  -> "
   f"{'OK — beats chance' if res_local['val_acc'] > 0.6 else 'WEAK/BAD — near chance'}")
log(f"   sem_coh_theme: {res_theme['val_acc']}  -> "
   f"{'OK — beats chance' if res_theme['val_acc'] > 0.6 else 'WEAK/BAD — near chance'}")

with open(os.path.join(HERE, "sem_coh_local_champ.pkl"), "wb") as f:
    pickle.dump({"champ": res_local["champ"], "val_acc": res_local["val_acc"]}, f)
with open(os.path.join(HERE, "sem_coh_theme_champ.pkl"), "wb") as f:
    pickle.dump({"champ": res_theme["champ"], "val_acc": res_theme["val_acc"]}, f)
log("saved sem_coh_local_champ.pkl, sem_coh_theme_champ.pkl")
log("DONE")
