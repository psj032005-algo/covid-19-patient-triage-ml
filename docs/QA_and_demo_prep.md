# Review prep - Problem 12 (COVID-19 triage)

## 90-second pitch
The paper predicts ICU admission, ventilation and death for COVID-19 patients from CT radiomics, clinical and lab data. Its data is
private and its repo is gone, so we reproduced the *method* on a public lab-only dataset (Huyut, Mendeley) for ICU admission and death.
We compared SMOTEENN vs none, four feature-selection options and five classifiers (40 combinations per task, 5-fold CV), tested the winner
once on a 30% hold-out and bootstrapped it. Test AUROC is about 0.90 (ICU) and 0.94 (death). We also found that the dataset's imputation leaks
the label, handled it, and report results with that caveat.

## Live demo script (about 2 minutes)
1. `python src/demo.py --task icu --n 10` - explain: model type, threshold; the live block scores all held-out patients (accuracy, sensitivity, specificity, AUROC); each sample row is a held-out patient (new random draw each run) with risk score, flag and OK/WRONG; the last line is the sample's own accuracy. Note accuracy vs the 'always no' baseline.
2. `python src/demo.py --task death --n 6 --seed 1` - uses the improved death model; note that some deaths still get low scores (sensitivity is about 0.8, not 100%).
3. Open `results/figures/roc_pr_icu.png` and `data_quality.png`.
4. If asked to show retraining: `python src/run.py --quick` (seconds, writes to `results_quick/`) or the full `python src/run.py` (about 2 min).

## Likely questions and honest answers
**Why not the paper's dataset?** Private (Jinling Hospital); the authors' repo returns 404. No public dataset has radiomics + clinical + lab + these outcomes.

**Which parts of the paper did you reproduce?** SMOTEENN, LASSO/PCA/FPR selection, LR/RF/SVM/MLP/LightGBM, model comparison by data type, bootstrapping, feature importance. Not reproduced: CT radiomics, clinical symptoms, ventilation outcome, the date-based external cohort.

**Why is there no radiomics?** The dataset only has blood values. Our "data types" are the three lab blocks instead of Radiom/Clin/Lab.

**How did you join the files?** By row position - there is no patient ID. Justification: label order is identical across the three death files, and ICU labels agree wherever two files both carry one. It is an assumption and is stated in the report.

**What is the data-quality problem?** Missing values were replaced by constants. In the death files the constants differ by class (Ferritin 395 for 92% of deaths, D-dimer 1277 for 80%), so a model reads the label from the fill-in value (raw AUROC 1.00). We blank such placeholder values and impute with random draws from observed values.

**Is the problem fully fixed?** No. A model given only the missing-value *pattern* still gets AUROC 0.87 (ICU) / 0.97 (death), so the pattern encodes the outcome (partly real - sicker patients get more tests - partly artifact). Our numbers are probably optimistic. The hematological-only death result (0.83) is the least-affected estimate.

**Why random-draw imputation instead of the median?** A median leaves an identical constant that tree models can detect, which re-exposes the missingness pattern. Random draws remove that marker, at the cost of noise.

**How did you avoid leakage in model selection?** Resampling, selection, scaling and imputation are inside the CV pipeline (imbalanced-learn Pipeline), refit per fold. The 30% test set was used once, for the winning configuration. The decision threshold comes from out-of-fold training scores.

**Why AUROC and AUPRC, not accuracy?** Positives are 14% (ICU) and 9% (death); a model that always says "no" is 86% / 91% accurate. AUPRC is more informative under imbalance.

**Why did SMOTEENN not help?** Best CV AUPRC is lower with it on both tasks. With only ~160-195 positives, synthetic samples plus ENN cleaning can add noise; gradient boosting copes with imbalance on its own. The paper found the opposite on its data.

**What do the confidence intervals mean?** 200 bootstrap resamples of the test set (paired across models); 95% percentile interval. They reflect test-set sampling variability only, not the uncertainty from the single split or the data artifacts.

**Why LightGBM?** Best CV AUROC for both tasks; handles skewed lab values and interactions without much tuning.

**Did you tune hyperparameters?** Only the choice among 40 configurations with fixed, reasonable hyperparameters. No nested CV or search - listed as future work.

**Why is sensitivity only ~0.8 at the chosen threshold?** The threshold maximises Youden's J (sensitivity + specificity - 1) on out-of-fold scores. A hospital could choose a lower threshold to catch more cases at the cost of more false alarms (see the PR curve).

**Is this clinically usable?** No. Single dataset, no external validation, data artifacts, no calibration analysis.

**What would you do with more time?** Add radiomics, nested CV with hyperparameter search, calibration curves, external validation, patient-level IDs / proper time-based split.

**Individual contribution?** Fill in who did what (e.g. data cleaning and diagnostics / modelling and evaluation / write-up and slides) - both members must be able to explain every part.

## Numbers to remember
ICU: n=2000, 279 positive (14%); test AUROC 0.90 (0.87-0.93), AUPRC 0.64; best LightGBM + FPR selection.
Death: n=2597, 233 positive (9%); final (improved) model test AUROC 0.94, AUPRC 0.78 (baseline LightGBM 0.71).
Baseline LR: 0.82 (ICU), 0.86 (death). Paper: 0.91 / 0.92 / 0.83 (ICU / ventilation / death) - different data, not comparable.

## Added: improvement experiment (src/improve.py)
**Did you try to improve the model?** Yes, on the training split only: six clinical ratio features, a tuned LightGBM (40-draw random search), averaging over 5 random imputations, and a RF/ExtraTrees/LR ensemble. ICU: none beat the baseline (CV AUROC 0.909), so it is unchanged. Death: ratios + tuned LightGBM x 5 imputations won in CV and on the test set - AUPRC 0.71 to 0.78, AUROC unchanged at 0.94 - so it is the final death model.

**Why not report accuracy?** Positives are 14% / 9%, so answering "no" for everyone is 86% / 91% accurate. We report AUROC, AUPRC, sensitivity and specificity instead.

**Could you get higher scores?** Yes, by letting the model read which values are missing (NaN-aware LightGBM: AUROC about 0.95 ICU / 0.99 death). We deliberately do not, because the missing-value pattern reflects how the dataset was filled in, not patient physiology; it would be a leak, not a better model.

**Test-set hygiene?** All selection used CV on the 70% training split. The held-out 30% was scored for the baseline and once for the single improved candidate per task (disclosed in the report).
