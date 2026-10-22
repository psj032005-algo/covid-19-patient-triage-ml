"""Live demo.   python src/demo.py [--task icu|death] [--n 10] [--mode balanced|random] [--seed S] [--all]
Everything is computed when you run it: the saved model scores the held-out test patients, a random sample is drawn,
and accuracy etc. are recomputed for that sample AND for the whole held-out test set (patients never seen in training)."""
import argparse
from pathlib import Path
import numpy as np, joblib
from sklearn.metrics import roc_auc_score, average_precision_score
import data
from pipeline import score, add_ratios, MultiImpute   # noqa: F401  (needed to unpickle)

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="icu", choices=["icu", "death"])
ap.add_argument("--n", type=int, default=10, help="patients in the random sample")
ap.add_argument("--mode", default="balanced", choices=["balanced", "random"],
                help="balanced: half positives / half negatives; random: real-world mix (mostly negatives)")
ap.add_argument("--seed", type=int, default=None, help="fix the random draw (omit for a new sample each run)")
ap.add_argument("--all", action="store_true", help="list every held-out test patient")
a = ap.parse_args()

root = Path(__file__).resolve().parent.parent
final = root / "models" / f"{a.task}_final.joblib"
art = joblib.load(final if final.exists() else root / "models" / f"{a.task}_best.joblib")
X, y, _ = data.load_task(a.task)
if art.get("ratios"):
    X = add_ratios(X)
thr, te = art["threshold"], np.array(art["test_idx"])
pos_name = "ICU" if a.task == "icu" else "died"

# ---- score ALL held-out patients once (so a patient's score is identical in every sample)
s = art["model"].predict_proba(X.iloc[te])[:, 1]
yt = y[te]; flag = (s >= thr).astype(int)


def metrics(yy, ff):
    tp = int(((ff == 1) & (yy == 1)).sum()); tn = int(((ff == 0) & (yy == 0)).sum())
    fp = int(((ff == 1) & (yy == 0)).sum()); fn = int(((ff == 0) & (yy == 1)).sum())
    return dict(n=len(yy), acc=(tp + tn) / max(len(yy), 1), sens=tp / max(tp + fn, 1) if (tp + fn) else float("nan"),
                spec=tn / max(tn + fp, 1) if (tn + fp) else float("nan"), prec=tp / max(tp + fp, 1) if (tp + fp) else float("nan"),
                tp=tp, tn=tn, fp=fp, fn=fn)


M = metrics(yt, flag)
model_desc = art["config"][1] if art["config"][0] == "improved" else f"{art['config'][2]} (selector={art['config'][0]}, resampling={art['config'][1]})"
print(f"Task: {a.task.upper()} | model: {model_desc} | decision threshold {thr:.3f}")
print(f"\nLIVE results on all {M['n']} held-out test patients ({int(yt.sum())} {pos_name}, prevalence {yt.mean():.1%}):")
print(f"  accuracy {M['acc']:.1%}  (always answering 'no' would score {1 - yt.mean():.1%})   AUROC {roc_auc_score(yt, s):.3f}   AUPRC {average_precision_score(yt, s):.3f}")
print(f"  sensitivity {M['sens']:.1%}   specificity {M['spec']:.1%}   precision {M['prec']:.1%}   (TP {M['tp']}  FP {M['fp']}  FN {M['fn']}  TN {M['tn']})")

# ---- draw the sample
rng = np.random.default_rng(a.seed)
if a.all:
    pick = np.arange(len(te))
elif a.mode == "balanced":
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    k = min(a.n // 2, len(pos))
    pick = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, min(a.n - k, len(neg)), replace=False)])
    rng.shuffle(pick)
else:
    pick = rng.choice(len(te), min(a.n, len(te)), replace=False)

print(f"\nSAMPLE of {len(pick)} patients ({a.mode}{', seed ' + str(a.seed) if a.seed is not None else ', new each run'}):")
print(f"{'patient':>8} {'true outcome':>13} {'risk score':>11} {'model says':>11} {'result':>7}")
for i in pick:
    ok = flag[i] == yt[i]
    print(f"{te[i]:>8} {pos_name if yt[i] else 'no':>13} {s[i]:>11.3f} {'HIGH RISK' if flag[i] else 'low':>11} {'OK' if ok else 'WRONG':>7}")
S = metrics(yt[pick], flag[pick])
print(f"\nSample accuracy: {S['tp'] + S['tn']}/{S['n']} = {S['acc']:.1%}"
      + (f" | caught {S['tp']}/{S['tp'] + S['fn']} {pos_name}" if (S['tp'] + S['fn']) else "")
      + (f" | correctly cleared {S['tn']}/{S['tn'] + S['fp']} others" if (S['tn'] + S['fp']) else ""))
print("(Small samples fluctuate; the full-test-set numbers above are the reliable ones.)")
