"""Builds report/Writeup.pdf (two pages) from results/summary.json.  python src/build_report.py"""
import json
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

root = Path(__file__).resolve().parent.parent
G = {t: pd.read_csv(root / "results" / f"cv_grid_{t}.csv") for t in ("icu", "death")}
def cvbase(t, m="LightGBM"):
    g = G[t]; r = g[(g.selector == "none") & (g.sampler == "none") & (g.model == m)]
    return float(r.cv_auroc.iloc[0])
Gm = {t: G[t].groupby("sampler")[["cv_auroc", "cv_auprc"]].max().to_dict("index") for t in G}
S = json.load(open(root / "results" / "summary.json")); I, D = S["icu"], S["death"]
X2 = json.load(open(root / "results" / "improved_summary.json")); FC = json.load(open(root / "results" / "final_choice.json"))
out = root / "report"; out.mkdir(exist_ok=True)
B = ParagraphStyle("b", fontName="Helvetica", fontSize=8.4, leading=10.6, spaceAfter=2.5)
H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9.6, leading=12, spaceBefore=4, spaceAfter=1.5, textColor=colors.HexColor("#1f3b6f"), keepWithNext=1)
T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceAfter=1)
SM = ParagraphStyle("sm", parent=B, fontSize=7.4, leading=9, textColor=colors.HexColor("#444444"))
f = lambda x: f"{x:.2f}"
ci = lambda d, k, m: f"{d['test'][k][m]:.2f} ({d['test_ci'][k][m][0]:.2f}-{d['test_ci'][k][m][1]:.2f})"
cfg = lambda d: f"{d['best_config']['classifier'] if 'classifier' in d['best_config'] else d['best_config']['model']}"
sel = lambda d: {"none": "no selection", "fpr": "FPR test (F-classif)", "lasso": "LASSO", "pca": "PCA"}[d["best_config"]["selector"]]

els = [
 Paragraph("CT-based Patient Triage of COVID-19 - Prediction of ICU Admission and Death (lab-data adaptation)", T),
 Paragraph("UE24CS352A Machine Learning - Mini-Project | Group 23, Problem Statement 12 | Prajwal S Jyanopanthar (PES1UG24CS594), Prakash N Heroor (PES1UG24CS595)", SM),
 Paragraph("1. Problem statement", H),
 Paragraph("Problem 12 reproduces Zhan &amp; Li (CS229, 2020): predict, at hospital admission, which COVID-19 inpatients will be admitted to the ICU, "
           "need mechanical ventilation, or die, so scarce resources can go to high-risk patients. The original used CT radiomics + clinical + lab data from a "
           "private cohort (1662 + 700 patients, Jinling Hospital); neither the data nor the authors' code repository is public. We therefore adapt the study to a "
           "<b>public lab-only dataset</b>: the same data-fusion/imbalance methodology, two of the three outcomes (ICU admission, death), no radiomics.", B),
 Paragraph("2. Dataset", H),
 Paragraph("<i>Routine Blood Values Datasets</i> (Huyut, Mendeley Data, v1, DOI 10.17632/w9mdp9k472.1; non-commercial/educational use). Each task is split into biochemical, "
           "hematological and immunological files. <b>ICU task:</b> 2000 adults (279 ICU, 14.0%), 54 features (25 biochemical, 17 hematological, 10 immunological, age, sex). "
           "<b>Death task:</b> 2597 adults (233 died, 9.0%), 37 features (12 + 15 + 10), no age/sex. There is no patient ID, so the three files are joined by row position; this is "
           "justified because label order is identical across files (death) and labels agree wherever two files both carry one (ICU), but it remains an assumption. "
           "The hematology/immunology ICU files leave 1035 labels blank, so labels come from the biochemical file.", B),
 Paragraph("<b>Data-quality findings (important).</b> (i) Missing lab values were filled with constants; in the death files the constant differs by class "
           "(e.g. Ferritin = 395 for 92% of deaths, D-dimer = 1277 for 80%), which leaks the label (raw single-block AUROC = 1.00). We blank any value covering &gt;25% of one class in a column "
           "(or whose most common value differs between classes) and treat impossible zeros as missing: 17.0k (ICU) and 32.8k (death) values. "
           "(ii) Even then, <i>which</i> values are missing predicts the outcome: a model given only the missingness mask scores AUROC "
           f"{I['diagnostics']['missingness_mask_only_auroc']:.2f} (ICU) / {D['diagnostics']['missingness_mask_only_auroc']:.2f} (death). Median imputation would let tree models read this mask, so we impute with random draws "
           "from each feature's observed training values. Residual leakage cannot be excluded; the hematological block (no repeated rows) is our least-affected estimate.", B),
 Paragraph("3. Approach", H),
 Paragraph("Following the paper: <b>(a) imbalance:</b> SMOTEENN vs. none; <b>(b) feature selection:</b> LASSO (L1 logistic), PCA (95% variance), FPR test, or none; "
           "<b>(c) classifiers:</b> Logistic Regression, Random Forest, SVM (RBF), MLP, LightGBM. All 40 combinations were scored per task with 5-fold stratified CV on a stratified 70% training split "
           "(preprocessing, resampling and selection are refit inside each fold, so nothing leaks). The configuration with the best CV AUROC was refit on the 70% and evaluated <b>once</b> on the untouched 30% test set. "
           "As in the paper's Radiom/Clin/Lab comparison, the winning configuration is re-trained on each data type (biochemical, hematological, immunological, all) and compared with a plain logistic-regression baseline using a paired bootstrap of the test set (200 resamples; 95% percentile CIs). "
           "The decision threshold is Youden's J on out-of-fold training scores. Importance is permutation importance on the test set.", B),
 Paragraph("4. Implementation", H),
 Paragraph("Python 3 / scikit-learn, imbalanced-learn, LightGBM. <font face='Courier'>src/data.py</font> loads, aligns and cleans the .sav files; <font face='Courier'>pipeline.py</font> defines the imputer, "
           "winsorizer (1-99%), scaler, SMOTEENN, selectors and models; <font face='Courier'>run.py</font> runs CV, test, bootstrap, importance and saves models; "
           "<font face='Courier'>plots.py</font>, <font face='Courier'>demo.py</font> and <font face='Courier'>build_report.py</font> make figures, the live demo and this PDF. "
           "Run time is about two minutes on 2 CPU cores; the seed is fixed (42).", B),
 Paragraph("5. Results (held-out 30% test set)", H),
]
rows = [["Data used", "ICU AUROC (95% CI)", "ICU AUPRC", "Death AUROC (95% CI)", "Death AUPRC"]]
for k, name in [("All", "All lab features"), ("Biochemical", "Biochemical only"), ("Hematological", "Hematological only"), ("Immunological", "Immunological only"), ("Baseline LR", "Baseline LR (all)")]:
    rows.append([name, ci(I, k, "auroc"), f(I["test"][k]["auprc"]), ci(D, k, "auroc"), f(D["test"][k]["auprc"])])
tb = Table(rows, colWidths=[38*mm, 36*mm, 22*mm, 36*mm, 24*mm])
tb.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.6), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6), ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 7.6),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")), ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#c8cfdc")), ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]))
els += [tb, Spacer(1, 3)]
els.append(Paragraph(
    f"Best models: ICU - {cfg(I)} with {sel(I)} and no resampling (CV AUROC {I['cv_auroc']:.2f}); death - {cfg(D)} with {sel(D)} and no resampling (CV AUROC {D['cv_auroc']:.2f}). "
    f"At the chosen threshold the ICU model reaches sensitivity {I['sensitivity']:.2f} / specificity {I['specificity']:.2f} / precision {I['precision']:.2f} ({I['tp']} TP, {I['fp']} FP, {I['fn']} FN, {I['tn']} TN); "
    f"the death model {D['sensitivity']:.2f} / {D['specificity']:.2f} / {D['precision']:.2f} ({D['tp']} TP, {D['fp']} FP, {D['fn']} FN, {D['tn']} TN). "
    f"Using all lab blocks beat every alternative in {min(v['auroc'] for v in D['paired_prob_all_better'].values()):.0%} or more of the death bootstrap resamples (AUROC); for ICU the margin over the biochemical block alone is smaller "
    f"({I['paired_prob_all_better']['Biochemical']['auroc']:.0%} of resamples), because biochemistry carries nearly all the ICU signal (hematological {I['test']['Hematological']['auroc']:.2f}, immunological {I['test']['Immunological']['auroc']:.2f}, i.e. near chance). "
    f"SMOTEENN did <u>not</u> help here: best CV AUROC/AUPRC with it is {Gm['icu']['smoteenn']['cv_auroc']:.2f}/{Gm['icu']['smoteenn']['cv_auprc']:.2f} (ICU) and {Gm['death']['smoteenn']['cv_auroc']:.2f}/{Gm['death']['smoteenn']['cv_auprc']:.2f} (death), versus {Gm['icu']['none']['cv_auroc']:.2f}/{Gm['icu']['none']['cv_auprc']:.2f} and {Gm['death']['none']['cv_auroc']:.2f}/{Gm['death']['none']['cv_auprc']:.2f} without it, unlike in the paper (threshold-free metrics; we tune the decision threshold instead). "
    "Top features: ICU - " + ", ".join(x["feature"] for x in I["top_features"][:5]) + "; death - " + ", ".join(x["feature"] for x in D["top_features"][:5]) + ".", B))
els.append(Table([[Image(str(root / "results/figures/roc_pr_icu.png"), width=87*mm, height=87*mm*3.1/7.2),
                   Image(str(root / "results/figures/data_quality.png"), width=72*mm, height=72*mm*2.8/5.2)]], colWidths=[90*mm, 80*mm]))
els += [Paragraph("Figure: left, ROC and precision-recall for ICU admission; right, the missing-value-pattern diagnostic (test AUROC).", SM),
 Paragraph("6. Can we do better? (improvement experiment)", H),
 Paragraph("We tried legitimate improvements, chosen with CV on the training split only and tested once: <b>B</b> six clinical ratio features (NLR, PLR, AST/ALT, urea/creatinine, LDH/albumin, CRP/albumin); "
           "<b>C</b> randomized-search-tuned LightGBM (40 draws); <b>D</b> C averaged over 5 random imputations (cancels the noise of one random fill-in); <b>E</b> D plus RF, ExtraTrees and LR (rank average). "
           f"<b>ICU:</b> nothing beat the baseline in CV (CV AUROC: baseline {X2['icu']['cv']['A baseline (current best)']['auroc']:.3f}, B {X2['icu']['cv']['B + ratio features']['auroc']:.3f}, C {X2['icu']['cv']['C + tuned LightGBM']['auroc']:.3f}, "
           f"D {X2['icu']['cv']['D C x 5 random imputations']['auroc']:.3f}, E {X2['icu']['cv']['E D + RF + ExtraTrees + LR']['auroc']:.3f}), so the baseline stays. "
           f"<b>Death:</b> D won in CV (AUPRC {X2['death']['cv']['A baseline (current best)']['auprc']:.2f} to {X2['death']['cv']['D C x 5 random imputations']['auprc']:.2f}) and on the test set "
           f"(AUROC {X2['death']['test_base']['auroc']:.2f} to {X2['death']['test_new']['auroc']:.2f}, AUPRC {X2['death']['test_base']['auprc']:.2f} to {X2['death']['test_new']['auprc']:.2f}; better in {X2['death']['prob_new_better_auprc']:.0%} of bootstrap resamples on AUPRC, "
           f"{X2['death']['prob_new_better_auroc']:.0%} on AUROC), so it is the final death model (sensitivity {X2['death']['sensitivity']:.2f}, specificity {X2['death']['specificity']:.2f}, precision {X2['death']['precision']:.2f}). "
           f"Plain accuracy is a poor yardstick here: always answering \"no\" scores {X2['icu']['baseline_accuracy_always_negative']:.0%} (ICU) and {X2['death']['baseline_accuracy_always_negative']:.0%} (death). "
           "We deliberately did <u>not</u> let models read the missing-value pattern (a NaN-aware LightGBM reaches AUROC "
           f"{I['diagnostics']['nan_aware_lightgbm_auroc']:.2f} / {D['diagnostics']['nan_aware_lightgbm_auroc']:.2f}) because that mostly reflects how the dataset was filled in, not patient physiology. "
           "The tables and curves in section 5 show the baseline pipeline; the test set was used for it and for the one improved model per task.", B),
 Paragraph("7. Conclusions", H),
 Paragraph(f"A lab-only model separates ICU-bound and fatal COVID-19 cases well on held-out data (ICU AUROC {I['test']['All']['auroc']:.2f}, death {D['test']['All']['auroc']:.2f}), echoing the paper's finding that fusing more data types helps "
           f"(all blocks together beat any single block). Gradient boosting was best; FPR feature selection helped the ICU model (CV AUROC {I['cv_auroc']:.2f} vs {cvbase('icu'):.2f} without selection) while the death model's best configuration uses no selection (CV AUROC {D['cv_auroc']:.2f}; LASSO/FPR variants are within 0.01). "
           "<b>Caveats:</b> the dataset's imputation artifacts mean absolute numbers are probably optimistic - the least-affected hematological-only death model scores "
           f"{D['test']['Hematological']['auroc']:.2f} - and ICU performance is carried by biochemistry; no external validation cohort, CT radiomics or ventilation label was available, so the paper's AUROCs "
           "(0.91 ICU, 0.92 ventilation, 0.83 death) are not directly comparable. With data-quality-safe collection (patient IDs, no label-dependent imputation) the same pipeline could be rerun unchanged. "
           "Future work: add radiomics features, nested CV with hyperparameter search, calibration and external validation.", B),
 Table([[Image(str(root / "results/figures/roc_pr_death.png"), width=88*mm, height=88*mm*3.1/7.2), Image(str(root / "results/figures/importance.png"), width=88*mm, height=88*mm*3.1/7.2)]], colWidths=[90*mm, 90*mm]),
 Paragraph("Figure: left, ROC and precision-recall for death; right, permutation importance of the top 10 features for both tasks (colour = data block).", SM),
 Paragraph("<b>References.</b> Zhan &amp; Li, CS229 Spring 2020, Stanford (Problem 12); Huyut, M.T., Routine Blood Values Datasets, Mendeley Data v1, doi:10.17632/w9mdp9k472.1; "
           "Batista et al. 2003 (SMOTEENN); Tibshirani 1996 (LASSO); Ke et al. 2017 (LightGBM); Pedregosa et al. 2011 (scikit-learn).", SM)]
doc = SimpleDocTemplate(str(out / "Writeup.pdf"), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=12*mm, bottomMargin=11*mm,
                        title="Mini-Project Write-up - Problem 12 (COVID-19 triage)", author="Group 23")
doc.build(els)
print("built")
