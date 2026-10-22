# Preliminary Model Evaluation

## 1. Objective and scope

This document records the existing preliminary experiments for two
classification tasks: ICU admission and death prediction.

**Important scope limitation:** the current implementation uses routine
laboratory measurements and demographic features, not CT-derived
radiomics. It also does not currently implement mechanical ventilation
prediction. Therefore, these experiments do not yet fully implement the
assigned CT-based triage problem.

## 2. Experimental setup

| Task | Total samples | Positive cases | Training samples | Test samples | Test positives |
|---|---:|---:|---:|---:|---:|
| ICU admission | 2,000 | 279 | 1,400 | 600 | 84 |
| Death | 2,597 | 233 | 1,817 | 780 | 70 |

The saved experiment summary records LightGBM as the selected model
for both tasks. The selected ICU configuration uses FPR feature selection
without resampling. The selected death configuration uses no feature
selection and no resampling.

The existing summary records cross-validation AUROC and AUPRC, as well
as test-set metrics, bootstrap confidence intervals, and threshold-based
classification metrics.

## 3. Preliminary test results

| Task | Model | Test AUROC | Test AUPRC |
|---|---|---:|---:|
| ICU admission | LightGBM, all features | 0.900 | 0.639 |
| ICU admission | Biochemical features | 0.881 | 0.626 |
| ICU admission | Hematological features | 0.506 | 0.140 |
| ICU admission | Immunological features | 0.551 | 0.158 |
| ICU admission | Baseline Logistic Regression | 0.819 | 0.504 |
| Death | LightGBM, all features | 0.939 | 0.708 |
| Death | Biochemical features | 0.834 | 0.460 |
| Death | Hematological features | 0.826 | 0.358 |
| Death | Immunological features | 0.729 | 0.370 |
| Death | Baseline Logistic Regression | 0.856 | 0.570 |

### Cross-validation results for selected configurations

| Task | CV AUROC | CV AUPRC |
|---|---:|---:|
| ICU admission | 0.909 | 0.673 |
| Death | 0.955 | 0.792 |

These values are transcribed from the existing `results/summary.json`.
They are preliminary, not validated final performance estimates.

## 4. Threshold-based test metrics

| Task | Sensitivity | Specificity | Precision |
|---|---:|---:|---:|
| ICU admission | 0.786 | 0.831 | 0.431 |
| Death | 0.800 | 0.901 | 0.444 |

The saved summary reports the following confusion-matrix counts:

| Task | True positives | False positives | False negatives | True negatives |
|---|---:|---:|---:|---:|
| ICU admission | 66 | 87 | 18 | 429 |
| Death | 56 | 70 | 14 | 640 |

The selected thresholds are approximately 0.0783 for ICU admission and
0.0192 for death. These thresholds should be reviewed and justified
before any clinical interpretation.

## 5. Selected feature-importance results

### ICU admission

The highest-ranked features in the saved summary are ALT, Calcium, Age,
AST, and GGT.

### Death

The highest-ranked features in the saved summary are BASO, eGFR, RDW,
Troponin, and Procalcitonin.

Feature importance describes model behavior in this experiment. It does
not establish causation or clinical usefulness.

## 6. Critical validation concern: missingness and preprocessing

The saved diagnostic AUROC for a missingness-mask-only model is:

| Task | Missingness-mask-only AUROC |
|---|---:|
| ICU admission | 0.872 |
| Death | 0.975 |

These values are unusually informative for a diagnostic based only on
whether measurements are missing. They indicate that missingness
patterns are strongly associated with the labels in these data.

The current data loader also performs placeholder cleaning using
class-specific information from the full dataset. This may expose
outcome information during preprocessing, including information from
the future test set. The results must therefore not be treated as
leakage-safe until the preprocessing is investigated and corrected.

Next validation steps:

1. Identify placeholder values using source documentation or a
   defensible label-independent rule.
2. Fit any learned preprocessing using training data only.
3. Ensure feature selection and threshold selection do not use the test
   labels.
4. Rerun cross-validation and final test evaluation after the fix.
5. Report the corrected results separately from these preliminary
   results.

Simply removing the existing placeholder cleaning is not sufficient
without checking whether the raw data contains outcome-encoded values.

## 7. Reproducibility and artifacts

Relevant existing files:

- `src/data.py` — dataset loading and preprocessing
- `src/pipeline.py` — model pipeline
- `src/run.py` — experiment runner
- `src/improve.py` — additional experiments
- `results/summary.json` — saved aggregate metrics
- `results/cv_grid_icu.csv` and `results/cv_grid_death.csv` — model
  comparison results
- `results/importance_icu.csv` and `results/importance_death.csv` —
  feature-importance results
- `results/figures/` — saved evaluation figures

The project dependencies are listed in `requirements.txt`. Exact
reproduction should be verified by rerunning the relevant scripts with
the same data, configuration, and random seeds.

## 8. Limitations and next steps

- The current data are laboratory measurements, not CT radiomics.
- Mechanical ventilation prediction is not implemented.
- The dataset alignment checks do not by themselves establish
  patient-level alignment when explicit patient identifiers are absent.
- Missingness patterns and label-dependent preprocessing require
  further investigation.
- The experiments are not external validation and are not suitable for
  clinical decision-making.

The next milestone is a defensible, leakage-aware evaluation followed
by alignment with the assigned CT-radiomics and three-outcome problem.
