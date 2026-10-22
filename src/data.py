"""Load and clean the Huyut routine-blood-values COVID-19 data (Mendeley w9mdp9k472, v1).

Each task is split into three SPSS files (biochemical / hematological / immunological) with
NO patient identifier. Rows are combined by position; this works because the label order is
identical across files (checked in `check_alignment`). It is an assumption and is reported.
"""
from pathlib import Path
import numpy as np, pandas as pd, pyreadstat

import os
DATA = Path(os.environ.get("COVID_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
MODALITIES = ["biochemical", "hematological", "immunological"]
# columns that can legitimately be 0 (cell-count differentials) -> keep zeros
PLACEHOLDER_SHARE = 0.25
ZERO_OK = {"BASO", "EOS", "ESO", "MONO", "Sex", "yas"}
RENAME = {"yaş": "Age", "Treatmen.Unıt_Service": "label", "Output_information": "label",
          "D.DİMER": "D.Dimer", "Porcalcitonin": "Procalcitonin", "Amilase": "Amylase", "Amilase ": "Amylase",
          "ESO": "EOS", "Direct_Bilirubin": "Direct_bilirubin", "Creatine.Kinase": "Creatinine_Kinase"}


_KEY = {"icu": "mildly", "death": "survivor"}   # words that identify each task in the original Mendeley names


def _find(task, mod):
    """Locate the .sav file for (task, modality) by keywords, so the ORIGINAL Mendeley file names work as they are.
    (Files already named like icu_biochemical.sav also work.)"""
    norm = lambda p: "".join(ch.lower() if ch.isalnum() else " " for ch in p.name)
    hits = [p for p in sorted(DATA.glob("*.sav"))
            if f"{task}_{mod}" == p.stem.lower()
            or (mod[:6] in norm(p) and _KEY[task] in norm(p) and "healty" not in norm(p) and "healthy" not in norm(p))]
    if not hits:
        raise FileNotFoundError(f"No .sav file for task='{task}', modality='{mod}' in {DATA}. "
                                f"Put the Mendeley .sav files there (see data/README.md).")
    return hits[0]


def _read(task, mod):
    df, _ = pyreadstat.read_sav(_find(task, mod))
    return df.rename(columns=RENAME)


def check_alignment(task):
    ys = [_read(task, m)["label"].values for m in MODALITIES]
    if task == "death":
        return all(np.array_equal(ys[0], y) for y in ys[1:])
    # icu: hema/immuno labels partly missing; compare where present
    ok = True
    for y in ys[1:]:
        m = ~np.isnan(y)
        ok &= bool((ys[0][m] == y[m]).all())
    return ok


def load_task(task):
    """Return X (DataFrame), y (0/1 array), groups (dict column -> modality) for task in {'icu','death'}."""
    parts = {m: _read(task, m) for m in MODALITIES}
    y = parts["biochemical"]["label"].values            # biochemical file has every label
    y = (y == 2).astype(int) if task == "icu" else y.astype(int)
    cols, owner = [], {}
    for m, df in parts.items():
        for c in df.columns:
            if c == "label" or c in owner:
                continue
            owner[c] = "demographics" if c in ("Sex", "Age") else m
            cols.append(df[c].rename(c))
    X = pd.concat(cols, axis=1)
    # an impossible zero in a lab value is a missing value
    for c in X.columns:
        if c not in ZERO_OK and owner[c] != "demographics":
            X.loc[X[c] <= 0, c] = np.nan
    # The files contain imputed values: a missing lab value was replaced by a constant. In the DEATH files
    # the constant differs between survivors and non-survivors (e.g. Ferritin = 395 for 92% of deaths), which
    # leaks the label. Any value that makes up >25% of one class in a continuous lab column is treated as an
    # imputation placeholder and blanked; the pipeline then imputes class-blind with the training median.
    log = {}
    for c in X.columns:
        if owner[c] == "demographics":
            continue
        modes = {}
        for lab in (0, 1):
            vc = X.loc[y == lab, c].dropna().value_counts(normalize=True)
            modes[lab] = (vc.index[0], vc.iloc[0]) if len(vc) else (np.nan, 0.0)
        class_specific = modes[0][0] != modes[1][0]          # different "most common value" per class
        continuous = X[c].nunique() > 100
        for lab in (0, 1):
            val, share = modes[lab]
            if share > PLACEHOLDER_SHARE and (continuous or class_specific):
                hit = (X[c] == val) & (y == lab)
                log[(c, lab)] = int(hit.sum())
                X.loc[hit, c] = np.nan
    X.attrs["placeholders_blanked"] = log
    # drop exact duplicate patient rows (avoids the same record in train and test)
    full = X.assign(_y=y)
    keep = ~full.duplicated().values
    log = X.attrs.get("placeholders_blanked", {})
    X, y = X[keep].reset_index(drop=True), y[keep]
    X.attrs["placeholders_blanked"] = log
    return X, y, owner


if __name__ == "__main__":
    for t in ("icu", "death"):
        raw_n = len(_read(t, "biochemical"))
        X, y, o = load_task(t)
        print(t, "aligned:", check_alignment(t), "| raw rows", raw_n, "-> after dedup", len(X),
              "| positives", int(y.sum()), f"({y.mean():.1%})", "| features", X.shape[1],
              "| missing %", round(100 * X.isna().mean().mean(), 1),
              "| placeholder values blanked", sum(X.attrs["placeholders_blanked"].values()))
        print("   missing % by class:", {int(k): round(100 * X[y == k].isna().mean().mean(), 1) for k in (0, 1)})
        print({m: sum(v == m for v in o.values()) for m in set(o.values())})
