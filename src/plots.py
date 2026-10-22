"""Figures for README / report / slides.  python src/plots.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

R = Path(__file__).resolve().parent.parent / "results"; F = R / "figures"; F.mkdir(exist_ok=True)
S = json.load(open(R / "summary.json"))
COL = {"All": "#1f5fa8", "Biochemical": "#d9792b", "Hematological": "#2a9d8f", "Immunological": "#7a5ea8", "Baseline_LR": "#6b7280"}
LS = {"All": "-", "Biochemical": "--", "Hematological": "-.", "Immunological": ":", "Baseline_LR": (0, (1, 1))}
LAB = {"All": "All lab features", "Baseline_LR": "Baseline LR"}
TITLE = {"icu": "ICU admission", "death": "In-hospital death"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e7eb", "grid.linewidth": .6, "axes.edgecolor": "#9ca3af", "figure.dpi": 150})

for task in ("icu", "death"):
    d = np.load(R / f"scores_{task}.npz"); y = d["y"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for k in COL:
        lab = LAB.get(k, k)
        f, t, _ = roc_curve(y, d[k]); ax[0].plot(f, t, color=COL[k], ls=LS[k], lw=1.6 if k == "All" else 1.2, label=lab)
        p, r, _ = precision_recall_curve(y, d[k]); ax[1].plot(r, p, color=COL[k], ls=LS[k], lw=1.6 if k == "All" else 1.2, label=lab)
    ax[0].plot([0, 1], [0, 1], color="#d1d5db", lw=.8); ax[1].axhline(y.mean(), color="#d1d5db", lw=.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC"); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    ax[0].legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.suptitle(f"{TITLE[task]} - held-out 30% test set", fontsize=10, x=.02, ha="left"); fig.tight_layout()
    fig.savefig(F / f"roc_pr_{task}.png"); plt.close(fig)

# bootstrap box plots
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey="row")
for j, task in enumerate(("icu", "death")):
    d = np.load(R / f"scores_{task}.npz")
    for i, m in enumerate(("auroc", "auprc")):
        keys = list(COL)
        bp = ax[i, j].boxplot([d[f"boot_{m}_{k}"] for k in keys], patch_artist=True, widths=.55, showfliers=False,
                              medianprops=dict(color="white", lw=1.4))
        for b, k in zip(bp["boxes"], keys): b.set(facecolor=COL[k], edgecolor="none")
        ax[i, j].set_xticks(range(1, 6)); ax[i, j].set_xticklabels(["All", "Biochem", "Haemato", "Immuno", "Base LR"], fontsize=8)
        ax[i, j].set_title(f"{TITLE[task]} - {m.upper()}", fontsize=9, loc="left")
fig.suptitle("Bootstrap distribution on the test set (200 resamples)", fontsize=10, x=.02, ha="left"); fig.tight_layout()
fig.savefig(F / "bootstrap.png"); plt.close(fig)

# permutation importance
GC = {"biochemical": "#d9792b", "hematological": "#2a9d8f", "immunological": "#7a5ea8", "demographics": "#6b7280"}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
for a, task in zip(ax, ("icu", "death")):
    fi = pd.read_csv(R / f"importance_{task}.csv").head(10)[::-1]
    a.barh(fi.feature, fi.importance, xerr=fi.sd, color=[GC[g] for g in fi.group], error_kw=dict(lw=.7, ecolor="#6b7280"))
    a.set_title(TITLE[task], fontsize=9, loc="left"); a.set_xlabel("AUROC drop when shuffled")
    a.grid(axis="y", visible=False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=c, label=k.capitalize()) for k, c in GC.items()], ncol=4, frameon=False, loc="lower center", fontsize=7.5)
fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(F / "importance.png"); plt.close(fig)

# data-quality diagnostic
fig, ax = plt.subplots(figsize=(5.2, 2.8)); x = np.arange(2); w = .27
vals = {"Missing-value pattern only\n(no lab values)": [S[t]["diagnostics"]["missingness_mask_only_auroc"] for t in ("icu", "death")],
        "Our pipeline\n(random fill-in)": [S[t]["test"]["All"]["auroc"] for t in ("icu", "death")],
        "Hematological block only\n(least affected)": [S[t]["test"]["Hematological"]["auroc"] for t in ("icu", "death")]}
for i, (k, v) in enumerate(vals.items()):
    b = ax.bar(x + (i - 1) * w, v, w * .92, color=["#b91c1c", "#1f5fa8", "#2a9d8f"][i], label=k)
    for xi, vi in zip(x + (i - 1) * w, v): ax.text(xi, vi + .01, f"{vi:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(["ICU admission", "In-hospital death"]); ax.set_ylim(.4, 1.05); ax.set_ylabel("Test AUROC")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(F / "data_quality.png"); plt.close(fig)
print("ok", sorted(p.name for p in F.iterdir()))
