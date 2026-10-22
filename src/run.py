"""End-to-end experiment: python src/run.py [--quick]"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.inspection import permutation_importance
import data
from pipeline import build, score, SELECTORS, SAMPLERS, MODELS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RES, MOD = ROOT / "results", ROOT / "models"
RES.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)
SEED = 42


def cv_one(sel, samp, model, X, y, folds):
    aucs, aps = [], []
    for tr, va in folds:
        p = build(sel, samp, model, SEED).fit(X.iloc[tr], y[tr])
        s = score(p, X.iloc[va])
        aucs.append(roc_auc_score(y[va], s)); aps.append(average_precision_score(y[va], s))
    return dict(selector=sel, sampler=samp, model=model, cv_auroc=np.mean(aucs), cv_auroc_sd=np.std(aucs),
                cv_auprc=np.mean(aps), cv_auprc_sd=np.std(aps))


def boot(y, s_dict, n_boot, rng):
    """Paired bootstrap of the test set; same resamples for every score vector."""
    out = {k: {"auroc": [], "auprc": []} for k in s_dict}
    idx_all = np.arange(len(y))
    for _ in range(n_boot):
        while True:
            i = rng.choice(idx_all, len(y), replace=True)
            if 0 < y[i].sum() < len(i):
                break
        for k, s in s_dict.items():
            out[k]["auroc"].append(roc_auc_score(y[i], s[i])); out[k]["auprc"].append(average_precision_score(y[i], s[i]))
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in out.items()}


def youden(y, s):
    from sklearn.metrics import roc_curve
    f, t, th = roc_curve(y, s)
    return th[np.argmax(t - f)]


def main(quick, n_boot, n_jobs):
    summary = {}
    for task in ("icu", "death"):
        t0 = time.time()
        X, y, owner = data.load_task(task)
        print(f"\n=== {task.upper()}: n={len(X)} pos={y.sum()} ({y.mean():.1%}) feats={X.shape[1]}")
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=SEED)
        Xtr, ytr, Xte, yte = X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True), y[te]
        print(f"train {len(tr)} (pos {ytr.sum()}) | test {len(te)} (pos {yte.sum()})")
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
        sels, samps, models = (["none", "lasso"], ["none", "smoteenn"], ["LR", "LightGBM"]) if quick else (SELECTORS, SAMPLERS, MODELS)
        grid = [(a, b, c) for a in sels for b in samps for c in models]
        rows = Parallel(n_jobs=n_jobs)(delayed(cv_one)(a, b, c, Xtr, ytr, folds) for a, b, c in grid)
        cv = pd.DataFrame(rows).sort_values("cv_auroc", ascending=False).reset_index(drop=True)
        cv.to_csv(RES / f"cv_grid_{task}.csv", index=False)
        print(cv.head(8).round(3).to_string())
        best = cv.iloc[0]; cfg = (best.selector, best.sampler, best.model)
        print("BEST (by CV AUROC):", cfg)

        # final fit on the 70% training set, evaluated once on the 30% held-out set
        final = build(*cfg, SEED).fit(Xtr, ytr)
        s_te = score(final, Xte)
        from sklearn.model_selection import cross_val_predict
        oof = np.zeros(len(ytr))                  # out-of-fold scores -> operating threshold from training data only
        for a_, b_ in folds:
            oof[b_] = score(build(*cfg, SEED).fit(Xtr.iloc[a_], ytr[a_]), Xtr.iloc[b_])
        thr = youden(ytr, oof)
        tn, fp, fn, tp = confusion_matrix(yte, (s_te >= thr).astype(int)).ravel()
        joblib.dump(dict(model=final, columns=list(X.columns), config=cfg, threshold=float(thr), test_idx=te.tolist()),
                    MOD / f"{task}_best.joblib")

        # diagnostics: how much does the *pattern of missing values* alone predict the label?
        from lightgbm import LGBMClassifier
        mk = lambda: LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=1, random_state=SEED)
        Mtr, Mte = Xtr.isna().astype(int), Xte.isna().astype(int)
        diag = dict(missingness_mask_only_auroc=roc_auc_score(yte, mk().fit(Mtr, ytr).predict_proba(Mte)[:, 1]),
                    nan_aware_lightgbm_auroc=roc_auc_score(yte, mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
        print("diagnostics (leaky, NOT used for results):", diag)
        # all other (selector, sampler, model) combos' held-out numbers for the reference table
        # data-type comparison with the winning configuration
        subsets = {"All": list(X.columns)}
        for m in ("biochemical", "hematological", "immunological"):
            subsets[m.capitalize()] = [c for c in X.columns if owner[c] == m]
        scores = {"All": s_te}
        point = {}
        for name, cols in subsets.items():
            if name != "All":
                scores[name] = score(build(*cfg, SEED).fit(Xtr[cols], ytr), Xte[cols])
            point[name] = dict(auroc=roc_auc_score(yte, scores[name]), auprc=average_precision_score(yte, scores[name]))
        # baseline that ignores sampling/selection: plain LR
        base = build("none", "none", "LR", SEED).fit(Xtr, ytr)
        scores["Baseline LR"] = score(base, Xte)
        point["Baseline LR"] = dict(auroc=roc_auc_score(yte, scores["Baseline LR"]), auprc=average_precision_score(yte, scores["Baseline LR"]))

        B = boot(yte, scores, n_boot, np.random.default_rng(SEED))
        ci = {k: {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in d.items()} for k, d in B.items()}
        paired = {k: {m: float(np.mean(B["All"][m] - B[k][m] > 0)) for m in ("auroc", "auprc")} for k in B if k != "All"}

        imp = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=n_jobs)
        fi = pd.DataFrame(dict(feature=X.columns, importance=imp.importances_mean, sd=imp.importances_std,
                               group=[owner[c] for c in X.columns])).sort_values("importance", ascending=False)
        fi.to_csv(RES / f"importance_{task}.csv", index=False)

        np.savez(RES / f"scores_{task}.npz", y=yte, **{k.replace(" ", "_"): v for k, v in scores.items()},
                 **{"boot_auroc_" + k.replace(" ", "_"): v["auroc"] for k, v in B.items()},
                 **{"boot_auprc_" + k.replace(" ", "_"): v["auprc"] for k, v in B.items()})
        summary[task] = dict(n=int(len(X)), positives=int(y.sum()), n_train=int(len(tr)), n_test=int(len(te)),
                             test_positives=int(yte.sum()), best_config=dict(selector=cfg[0], sampler=cfg[1], model=cfg[2]),
                             cv_auroc=float(best.cv_auroc), cv_auprc=float(best.cv_auprc),
                             test=point, test_ci=ci, paired_prob_all_better=paired, threshold=float(thr), diagnostics={k: float(v) for k, v in diag.items()},
                             sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                             precision=float(tp / max(tp + fp, 1)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                             top_features=fi.head(10)[["feature", "group", "importance"]].to_dict("records"),
                             seconds=round(time.time() - t0))
        print(json.dumps({k: summary[task][k] for k in ("test", "sensitivity", "specificity")}, indent=1, default=float))
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.quick:   # never overwrite the real results with the smoke test
        RES, MOD = ROOT / "results_quick", ROOT / "models_quick"; RES.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)
    main(a.quick, 20 if a.quick else a.boot, a.jobs)
