"""Legitimate improvements over the baseline model, selected ONLY with CV on the 70% training split, then tested once.
  B  + clinical ratio features (NLR, PLR, ALT/AST, Urea/Creatinine, LDH/Albumin, CRP/Albumin)
  C  B + tuned LightGBM (randomized search, 5-fold CV)
  D  C averaged over K random imputations (removes the noise of a single random fill-in)
  E  D + Random Forest + ExtraTrees + LR rank-averaged ensemble
python src/improve.py"""
import json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
from scipy.stats import rankdata, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import data
from pipeline import RandomSampleImputer, Winsorizer, build, score, MultiImpute, add_ratios
import run as base

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent; RES, MOD = ROOT / "results", ROOT / "models"; SEED = 42


def pipe(clf, seed=0, scale=True):
    s = [("impute", RandomSampleImputer(seed)), ("clip", Winsorizer())]
    if scale: s.append(("scale", StandardScaler()))
    return Pipeline(s + [("clf", clf)])


def lgbm(**kw):
    p = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1, random_state=SEED); p.update(kw)
    return LGBMClassifier(**p)


def cv_scores(est, X, y, seeds=(0, 1, 2)):
    a, p = [], []
    for sd in seeds:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            m = clone(est).fit(X.iloc[tr], y[tr]); s = m.predict_proba(X.iloc[va])[:, 1]
            a.append(roc_auc_score(y[va], s)); p.append(average_precision_score(y[va], s))
    return np.mean(a), np.mean(p)


def main():
    S0 = json.load(open(RES / "summary.json")); summary = {}
    for task in ("icu", "death"):
        t0 = time.time()
        X0, y, owner = data.load_task(task); X = add_ratios(X0)
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=SEED)   # identical split to run.py
        Xtr, ytr, Xte, yte = X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True), y[te]
        X0tr, X0te = X0.iloc[tr].reset_index(drop=True), X0.iloc[te].reset_index(drop=True)
        cfg = tuple(S0[task]["best_config"][k] for k in ("selector", "sampler", "model"))
        print(f"\n=== {task.upper()}  (baseline config {cfg}; {Xtr.shape[1]} features incl. {Xtr.shape[1]-X0.shape[1]} ratios)")

        # tuned LightGBM on the ratio features (train-only CV)
        space = dict(clf__n_estimators=randint(100, 600), clf__learning_rate=uniform(0.01, 0.09), clf__num_leaves=randint(4, 32),
                     clf__min_child_samples=randint(5, 41), clf__subsample=uniform(0.6, 0.4), clf__colsample_bytree=uniform(0.4, 0.6),
                     clf__reg_lambda=uniform(0, 10), clf__scale_pos_weight=[1, 2, 4])
        rs = RandomizedSearchCV(pipe(lgbm(), 0, scale=False), space, n_iter=40, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                random_state=SEED, n_jobs=-1, refit=False).fit(Xtr, ytr)
        bp = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
        print("tuned params:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in bp.items()}, "| search CV AUROC", round(rs.best_score_, 4))
        tuned = lambda seed: pipe(lgbm(**bp), seed, scale=False)
        rf = lambda seed: pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=SEED), seed, scale=False)
        et = lambda seed: pipe(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=SEED), seed, scale=False)
        lr = lambda seed: pipe(LogisticRegression(C=0.3, max_iter=3000, random_state=SEED), seed)

        cands = {"A baseline (current best)": (X0tr, X0te, lambda: build(*cfg, SEED)),
                 "B + ratio features": (Xtr, Xte, lambda: build(*cfg, SEED)),
                 "C + tuned LightGBM": (Xtr, Xte, lambda: tuned(0)),
                 "D C x 5 random imputations": (Xtr, Xte, lambda: MultiImpute([tuned], K=5)),
                 "E D + RF + ExtraTrees + LR": (Xtr, Xte, lambda: MultiImpute([tuned, rf, et, lr], K=5))}
        def run_cand(name):
            Xa, _, mk = cands[name]; return name, cv_scores(mk(), Xa, ytr, seeds=(0, 1))
        rows = dict(Parallel(n_jobs=2)(delayed(run_cand)(n) for n in cands))
        for n, (a, p) in rows.items(): print(f"  CV  {n:32s} AUROC {a:.4f}  AUPRC {p:.4f}")
        best = max([n for n in cands if not n.startswith("A")], key=lambda n: rows[n][0])
        print("selected by train-CV AUROC:", best)

        # final: fit on 70%, test ONCE; paired bootstrap vs baseline
        fa = cands["A baseline (current best)"][2]().fit(X0tr, ytr); sa = score(fa, X0te)
        Xb, Xbt, mk = cands[best]; fb = mk().fit(Xb, ytr); sb = fb.predict_proba(Xbt)[:, 1]
        B = base.boot(yte, {"new": sb, "base": sa}, 200, np.random.default_rng(SEED))
        ci = lambda k, m: [float(np.percentile(B[k][m], 2.5)), float(np.percentile(B[k][m], 97.5))]
        oof = np.zeros(len(ytr))
        for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xb, ytr):
            oof[b_] = mk().fit(Xb.iloc[a_], ytr[a_]).predict_proba(Xb.iloc[b_])[:, 1]
        thr = base.youden(ytr, oof); tn, fp, fn, tp = confusion_matrix(yte, (sb >= thr).astype(int)).ravel()
        res = dict(selected=best, cv={n: dict(auroc=a, auprc=p) for n, (a, p) in rows.items()}, tuned_params=bp,
                   test_new=dict(auroc=roc_auc_score(yte, sb), auprc=average_precision_score(yte, sb), auroc_ci=ci("new", "auroc"), auprc_ci=ci("new", "auprc")),
                   test_base=dict(auroc=roc_auc_score(yte, sa), auprc=average_precision_score(yte, sa), auroc_ci=ci("base", "auroc"), auprc_ci=ci("base", "auprc")),
                   prob_new_better_auroc=float(np.mean(B["new"]["auroc"] > B["base"]["auroc"])), prob_new_better_auprc=float(np.mean(B["new"]["auprc"] > B["base"]["auprc"])),
                   sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)), precision=float(tp / max(tp + fp, 1)),
                   accuracy=float((tp + tn) / len(yte)), baseline_accuracy_always_negative=float(1 - yte.mean()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), threshold=float(thr))
        print(f"TEST  base AUROC {res['test_base']['auroc']:.3f} AUPRC {res['test_base']['auprc']:.3f}  ->  new AUROC {res['test_new']['auroc']:.3f} "
              f"[{res['test_new']['auroc_ci'][0]:.3f}-{res['test_new']['auroc_ci'][1]:.3f}] AUPRC {res['test_new']['auprc']:.3f} | P(new>base) AUROC {res['prob_new_better_auroc']:.2f} AUPRC {res['prob_new_better_auprc']:.2f}")
        print(f"      accuracy {res['accuracy']:.3f} (always-'no' gets {res['baseline_accuracy_always_negative']:.3f}); sens {res['sensitivity']:.2f} spec {res['specificity']:.2f} prec {res['precision']:.2f}  [{round(time.time()-t0)}s]")
        if hasattr(fb, "makers"): fb.makers = None   # makers are closures; not needed after fit (keeps the file picklable)
        joblib.dump(dict(model=fb, columns=list(X.columns), config=("improved", best), threshold=res["threshold"], test_idx=te.tolist(), ratios=True), MOD / f"{task}_improved.joblib")
        summary[task] = res
    (RES / "improved_summary.json").write_text(json.dumps(summary, indent=1))


def finalize():
    """Fixed rule: use the improved model only if it did not lose on train-CV AUROC AND beat the baseline's test AUPRC in >=95% of
    bootstrap resamples; otherwise keep the baseline. Writes models/<task>_final.joblib (what demo.py loads)."""
    import shutil
    S = json.load(open(RES / "improved_summary.json")); out = {}
    for task, r in S.items():
        cv = r["cv"]; use = (r["selected"] in cv and cv[r["selected"]]["auroc"] >= cv["A baseline (current best)"]["auroc"]
                             and r["prob_new_better_auprc"] >= 0.95)
        shutil.copy(MOD / f"{task}_{'improved' if use else 'best'}.joblib", MOD / f"{task}_final.joblib")
        out[task] = "improved: " + r["selected"] if use else "baseline"
    (RES / "final_choice.json").write_text(json.dumps(out, indent=1)); print("final choice:", out)


if __name__ == "__main__":
    import sys
    if "--finalize" in sys.argv: finalize()
    else: main(); finalize()
