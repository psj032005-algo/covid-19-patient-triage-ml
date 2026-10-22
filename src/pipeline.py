"""Model zoo and preprocessing, following Zhan & Li (CS229 2020): SMOTEENN, LASSO / PCA / FPR
feature selection, and LR / RF / SVM / MLP / LightGBM classifiers."""
import warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectFpr, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """Fill each missing value with a random draw from that feature's observed training values.

    Median/mean imputation leaves a constant that tree models can detect, which re-exposes WHICH values were
    missing. In this dataset the missingness pattern alone predicts the label (AUROC 0.88 ICU / 0.98 death), so
    a constant fill-in would let models cheat. Random draws remove that marker."""
    def __init__(self, random_state=0):
        self.random_state = random_state
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.obs_ = [X[~np.isnan(X[:, j]), j] for j in range(X.shape[1])]
        return self
    def transform(self, X):
        X = np.array(X, float)
        rng = np.random.RandomState(self.random_state)
        for j, o in enumerate(self.obs_):
            m = np.isnan(X[:, j])
            if m.any():
                X[m, j] = rng.choice(o, m.sum()) if len(o) else 0.0
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to its [lo, hi] training percentiles (labs contain impossible outliers)."""
    def __init__(self, lo=1, hi=99):
        self.lo, self.hi = lo, hi
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_ = np.nanpercentile(X, self.lo, axis=0)
        self.hi_ = np.nanpercentile(X, self.hi, axis=0)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


def selector(name, seed):
    if name == "none":
        return "passthrough"
    if name == "lasso":
        return SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=500,
                                                   random_state=seed))
    if name == "fpr":
        return SelectFpr(f_classif, alpha=0.05)
    if name == "pca":
        return PCA(n_components=0.95, random_state=seed)
    raise ValueError(name)


def classifier(name, seed):
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=seed)
    if name == "SVM":
        return SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400, early_stopping=True,
                             random_state=seed)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1, random_state=seed)
    raise ValueError(name)


SELECTORS = ["none", "lasso", "fpr", "pca"]
SAMPLERS = ["none", "smoteenn"]
MODELS = ["LR", "RF", "SVM", "MLP", "LightGBM"]


def build(sel, samp, model, seed=0):
    steps = [("impute", RandomSampleImputer(seed)), ("clip", Winsorizer()), ("scale", StandardScaler())]
    if samp == "smoteenn":
        steps.append(("resample", SMOTEENN(random_state=seed)))
    steps += [("select", selector(sel, seed)), ("clf", classifier(model, seed))]
    return Pipeline(steps)


def score(pipe, X):
    """Continuous risk score for any classifier."""
    return pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") and not isinstance(
        pipe.named_steps["clf"], SVC) else pipe.decision_function(X)


def add_ratios(X):
    X = X.copy(); d = lambda a, b: (X[a] / X[b].where(X[b] > 0)) if a in X and b in X else None
    for name, (a, b) in {"NLR": ("NEU", "LYM"), "PLR": ("PLT", "LYM"), "AST_ALT": ("AST", "ALT"), "Urea_Creat": ("Urea", "Creatinine"),
                         "LDH_Alb": ("LDH", "Albumin"), "CRP_Alb": ("CRP", "Albumin")}.items():
        r = d(a, b)
        if r is not None: X[name] = r
    return X


class MultiImpute(BaseEstimator, ClassifierMixin):
    """Average several models, each trained/applied with a different random imputation (and a rank-average over model kinds)."""
    def __init__(self, makers, K=5):
        self.makers, self.K = makers, K
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.models_ = [[m(seed).fit(X, y) for seed in range(self.K)] for m in self.makers]
        return self
    def decision_function(self, X):
        out = 0
        for group in self.models_:
            s = np.mean([p.predict_proba(X)[:, 1] for p in group], axis=0)
            out = out + rankdata(s) / len(s) if len(self.models_) > 1 else out + s
        return out / len(self.models_)
    def predict_proba(self, X):
        s = self.decision_function(X); return np.c_[1 - s, s]


