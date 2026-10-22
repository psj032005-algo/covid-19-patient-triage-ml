import json; from pathlib import Path
root = Path(__file__).resolve().parent.parent
S = json.load(open(root / "results/summary.json")); I, D = S["icu"], S["death"]
X2 = json.load(open(root / "results/improved_summary.json"))
r = lambda d, k: f"{d['test'][k]['auroc']:.3f} / {d['test'][k]['auprc']:.3f}"
txt = f"""# COVID-19 Patient Triage from Routine Lab Data (Problem 12)

UE24CS352A Machine Learning mini-project - Group 23 - Prajwal S Jyanopanthar (PES1UG24CS594), Prakash N Heroor (PES1UG24CS595)

Adapts *CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death*
(Zhan & Li, CS229 2020) to a **public** dataset. The paper's CT/clinical data are private and its repo is no longer available, so we reproduce
its methodology (SMOTEENN, LASSO/PCA/FPR selection, LR/RF/SVM/MLP/LightGBM, data-type comparison, bootstrap) on routine blood values
for two outcomes: **ICU admission** and **in-hospital death** (no radiomics, no ventilation label).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Download the data (not committed, licence is non-commercial/educational) and copy the `.sav` files into `data/` **with their original names** - see [`data/README.md`](data/README.md).

## Run
```bash
python src/run.py            # full experiment (~2 min on 2 cores): CV grid, test set, bootstrap, saves models + results/
python src/improve.py        # improvement experiment (~5 min); picks the final model per task
python src/plots.py          # figures -> results/figures/
python src/build_report.py   # two-page PDF -> report/Writeup.pdf
python src/demo.py --task icu --n 10     # live demo: new random patients each run + live accuracy (use --task death for mortality)
python src/demo.py --task death --mode random --seed 3   # real-world mix instead of 50/50; --all lists every held-out patient
python src/run.py --quick    # tiny smoke test (writes to results_quick/, never touches results/)
```

## Results (stratified 70/30 split, 30% held out; test AUROC / AUPRC)
| Data used | ICU | Death |
|---|---|---|
| All lab features (best model) | {r(I,'All')} | {r(D,'All')} |
| Biochemical only | {r(I,'Biochemical')} | {r(D,'Biochemical')} |
| Hematological only | {r(I,'Hematological')} | {r(D,'Hematological')} |
| Immunological only | {r(I,'Immunological')} | {r(D,'Immunological')} |
| Baseline logistic regression | {r(I,'Baseline LR')} | {r(D,'Baseline LR')} |

Best models (by 5-fold CV AUROC): ICU - LightGBM + FPR selection; death - LightGBM, no selection, no resampling.
Full CV grids: `results/cv_grid_*.csv`; metrics with bootstrap CIs: `results/summary.json`.

## Improvement experiment (`python src/improve.py`)
Ratio features, tuned LightGBM, averaging over 5 random imputations and an ensemble were compared by CV on the training split only. ICU: baseline kept.
Death: variant D (ratios + tuned LightGBM x 5 imputations) is final - test AUROC {X2['death']['test_new']['auroc']:.3f}, AUPRC {X2['death']['test_new']['auprc']:.3f} (baseline {X2['death']['test_base']['auprc']:.3f}).
`demo.py` loads `models/<task>_final.joblib`, written by that script. Details: `results/improved_summary.json`.

## Data-quality caveats (read before trusting the numbers)
* No patient ID: files for one task are joined **by row position** (label order is identical across files).
* Missing values were filled with constants, and for the death task the constant differs by class (label leakage). We blank such
  placeholders (`src/data.py`), and impute with random draws from observed values so tree models cannot read the missingness pattern.
* Even so, the missing-value pattern alone gives AUROC {I['diagnostics']['missingness_mask_only_auroc']:.2f} (ICU) / {D['diagnostics']['missingness_mask_only_auroc']:.2f} (death), so absolute performance is probably optimistic.
  The hematological block (no repeated rows) is the least-affected estimate.

## Layout
```
src/data.py          load/align/clean the .sav files
src/pipeline.py      imputer, winsorizer, scaler, SMOTEENN, selectors, classifiers
src/run.py           CV grid -> held-out test -> bootstrap -> importance -> saved models
src/plots.py         figures        src/demo.py   live demo        src/build_report.py  PDF
results/             CV grids, summary.json, figures      report/Writeup.pdf   two-page write-up
```
Seed fixed to 42. Data: Huyut, *Routine Blood Values Datasets*, Mendeley Data v1, doi:10.17632/w9mdp9k472.1.
"""
(root / "README.md").write_text(txt)
