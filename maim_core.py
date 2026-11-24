"""
M-AIM core library: synthetic multi-modal cohort, modality-specific encoders,
fusion models, statistics (DeLong, McNemar, Hosmer-Lemeshow, corrected resampled
t-test), decision-curve analysis, fairness, survival and visualisation helpers.

IMPORTANT: the cohort generated here is SYNTHETIC. All numbers produced by the
pipeline demonstrate the methodology, not clinical performance. Replace
`make_cohort()` with a loader for real, ethically approved data to obtain real results.
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_predict)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- data
CMR = ["T1_ms", "T2_ms", "LGE_pct", "ECV_pct", "LVEF_pct", "wall_motion"] + [f"rad_{i}" for i in range(6)]
ECG = ["ST_dev_mm", "QRS_ms", "QTc_ms", "PR_ms", "HR_bpm", "Twave_mV"] + [f"ecgemb_{i}" for i in range(10)]
BIO = ["log_hsTnT", "log_CRP", "log_NTproBNP", "ESR"]
DEMO = ["age", "sex"]
GROUPS = {"CMR": CMR, "ECG": ECG, "Biomarkers": BIO}
ALL_FEATS = CMR + ECG + BIO + DEMO
SUBTYPES = ["viral", "autoimmune", "drug-induced", "idiopathic"]


def make_cohort(n=1500, prevalence=0.40, seed=42, n_sites=3, site_shift=1.0):
    """Synthetic cohort with site (scanner) shift, ACS-like mimics, partial disease signal and missingness."""
    r = np.random.default_rng(seed)
    y = r.binomial(1, prevalence, n)
    site = r.integers(0, n_sites, n)
    age = np.clip(r.normal(38, 14, n), 18, 85).round()
    sex = r.binomial(1, 0.5, n)
    eth = r.choice(["Group A", "Group B", "Group C", "Group D"], n, p=[.4, .3, .2, .1])
    # disease strength (patchy / mild disease -> weak signal)
    eff = y * (0.5 + r.beta(2, 3, n)) * np.where(r.random(n) < 0.25, 0.35, 1.0)
    mimic = (y == 0) & (r.random(n) < 0.15)          # ACS-like controls with raised troponin / ST change
    off = (site - 1) * site_shift
    d = pd.DataFrame({"y": y, "site": site, "age": age, "sex": sex, "ethnicity": eth, "severity": eff})
    d["T1_ms"] = 985 + 75 * eff + r.normal(0, 38, n) + 35 * off
    d["T2_ms"] = 48 + 6.5 * eff + r.normal(0, 3.4, n) + 1.0 * off
    d["LGE_pct"] = np.clip(0.8 + 9 * eff * r.random(n) + r.normal(0, 1.6, n), 0, None)
    d["ECV_pct"] = 26 + 5 * eff + r.normal(0, 2.6, n)
    d["LVEF_pct"] = 58 - 8 * eff * r.random(n) + r.normal(0, 6, n)
    d["wall_motion"] = np.clip(0.3 + 1.4 * eff * r.random(n) + r.normal(0, .5, n), 0, None)
    for i in range(6):
        d[f"rad_{i}"] = r.choice([-1, 1]) * (0.25 + 0.1 * i) * eff + r.normal(0, 1, n)
    d["ST_dev_mm"] = 0.2 + 0.8 * eff + 0.7 * mimic + r.normal(0, 0.55, n)
    d["QRS_ms"] = 92 + 8 * eff + r.normal(0, 10, n)
    d["QTc_ms"] = 410 + 12 * eff + r.normal(0, 25, n)
    d["PR_ms"] = 160 + r.normal(0, 20, n)
    d["HR_bpm"] = 72 + 10 * eff + r.normal(0, 12, n)
    d["Twave_mV"] = 0.5 - 0.3 * eff + r.normal(0, 0.3, n)
    for i in range(10):
        d[f"ecgemb_{i}"] = r.choice([-1, 1]) * 0.3 * eff + r.normal(0, 1, n)
    d["log_hsTnT"] = np.log(10) + 1.6 * eff + 1.4 * mimic + r.normal(0, 1.0, n)
    d["log_CRP"] = np.log(3) + 1.0 * eff + r.normal(0, 1.0, n)
    d["log_NTproBNP"] = np.log(120) + 0.9 * eff + r.normal(0, 1.0, n)
    d["ESR"] = np.clip(12 + 8 * eff + r.normal(0, 8, n), 1, None)
    # subtype (myocarditis only) and survival
    d["subtype"] = np.where(y == 1, r.choice(SUBTYPES, n, p=[.45, .2, .1, .25]), "none")
    sub = d["subtype"].values
    d.loc[sub == "viral", "log_CRP"] += 0.35
    d.loc[sub == "autoimmune", "ESR"] += 10
    d.loc[sub == "autoimmune", "LGE_pct"] += 1.5
    d.loc[sub == "drug-induced", "QRS_ms"] += 8
    d.loc[sub == "drug-induced", "log_hsTnT"] -= 0.3
    hazard = 0.015 * np.exp(0.9 * eff + 0.02 * (58 - d["LVEF_pct"]))
    t_ev = r.exponential(1 / hazard)
    d["time_months"] = np.minimum(t_ev, 24.0)
    d["event"] = (t_ev <= 24.0).astype(int)
    # missingness
    for c in BIO:
        d.loc[r.random(n) < .05, c] = np.nan
    for c in [f"ecgemb_{i}" for i in range(10)]:
        d.loc[r.random(n) < .03, c] = np.nan
    d["age_group"] = pd.cut(d["age"], [0, 40, 65, 120], labels=["18-40", "41-65", ">65"]).astype(str)
    return d


# --------------------------------------------------------------------------- models
def _pipe(clf, cols):
    pre = ColumnTransformer([("s", Pipeline([("i", SimpleImputer(strategy="median")),
                                             ("z", StandardScaler())]), list(cols))], remainder="drop")
    return Pipeline([("pre", pre), ("clf", clf)])


def base_classifiers(seed=0):
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=3000),
        "Random Forest": RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed),
        "SVM (RBF)": SVC(probability=True, C=1.0, random_state=seed),
        "MLP": MLPClassifier((64, 32), alpha=1e-3, early_stopping=True, max_iter=600, random_state=seed),
    }


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class LateFusion(BaseEstimator, ClassifierMixin):
    """Late fusion: unimodal classifiers, probabilities averaged (or stacked if learned=True)."""
    def __init__(self, base="lr", learned=False, hgb_params=None, seed=0):
        self.base, self.learned, self.hgb_params, self.seed = base, learned, hgb_params, seed

    def _mk(self):
        if self.base == "gb":
            return HistGradientBoostingClassifier(**{**dict(max_depth=3, learning_rate=0.05, max_iter=200, random_state=self.seed), **(self.hgb_params or {})})
        return LogisticRegression(C=1.0, max_iter=3000)

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.m_ = {g: _pipe(self._mk(), c).fit(X, y) for g, c in GROUPS.items()}
        if self.learned:
            Z = np.column_stack([_logit(cross_val_predict(_pipe(self._mk(), c), X, y, cv=5, method="predict_proba")[:, 1]) for c in GROUPS.values()])
            self.meta_ = LogisticRegression(C=1.0).fit(Z, y)
        return self

    def predict_proba(self, X):
        P = np.column_stack([self.m_[g].predict_proba(X)[:, 1] for g in GROUPS])
        p = self.meta_.predict_proba(np.column_stack([_logit(P[:, i]) for i in range(3)]))[:, 1] if self.learned else P.mean(1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


class GatedCrossModalFusion(BaseEstimator, ClassifierMixin):
    """
    Intermediate-fusion surrogate of M-AIM (no deep-learning dependency).
      encoders : per modality, LR + gradient-boosting logits -> modality embedding z_m
      gate     : g_m = softmax_m(tau*|z_m|)   (confidence-weighted, as in the gated-fusion equations)
      cross    : pairwise products z_i*z_j     (cross-modal interaction terms)
      head     : regularised logistic regression on [g*z, z, z_i z_j]
    Modalities that are entirely missing for a row get z_m = 0 and are masked from the gate.
    """
    def __init__(self, tau=0.8, hgb_params=None, C=0.5, seed=0):
        self.tau, self.hgb_params, self.C, self.seed = tau, hgb_params, C, seed

    def _encs(self, cols):
        gb = HistGradientBoostingClassifier(**{**dict(max_depth=3, learning_rate=0.05, max_iter=150, random_state=self.seed), **(self.hgb_params or {})})
        return [_pipe(LogisticRegression(C=1.0, max_iter=3000), cols), _pipe(gb, cols)]

    def _feat(self, Zs, avail):
        Z = np.column_stack(Zs) * avail
        s = self.tau * np.abs(Z) - 1e9 * (1 - avail)
        g = np.exp(s - s.max(1, keepdims=True)) * avail
        g = g / np.clip(g.sum(1, keepdims=True), 1e-9, None)
        cross = np.column_stack([Z[:, 0] * Z[:, 1], Z[:, 0] * Z[:, 2], Z[:, 1] * Z[:, 2]])
        return np.column_stack([g * Z, Z, cross])

    @staticmethod
    def _avail(X):
        return np.column_stack([(~X[c].isna().all(axis=1)).astype(float).values for c in GROUPS.values()])

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.enc_ = {g: [e.fit(X, y) for e in self._encs(c)] for g, c in GROUPS.items()}
        cv = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        Zs = []
        for g, c in GROUPS.items():
            oof = [_logit(cross_val_predict(e, X, y, cv=cv, method="predict_proba")[:, 1]) for e in self._encs(c)]
            Zs.append(np.mean(oof, axis=0))
        self.head_ = LogisticRegression(C=self.C, max_iter=3000).fit(self._feat(Zs, np.ones((len(X), 3))), y)
        return self

    def _z(self, X):
        return [np.mean([_logit(e.predict_proba(X)[:, 1]) for e in self.enc_[g]], axis=0) for g in GROUPS]

    def predict_proba(self, X):
        p = self.head_.predict_proba(self._feat(self._z(X), self._avail(X)))[:, 1]
        return np.column_stack([1 - p, p])

    def modality_weights(self, X):
        Z = np.column_stack(self._z(X)); av = self._avail(X)
        s = self.tau * np.abs(Z); g = np.exp(s - s.max(1, keepdims=True)); return g / g.sum(1, keepdims=True)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


class SigmoidCalibrated(BaseEstimator, ClassifierMixin):
    """Platt (sigmoid) recalibration of any probabilistic estimator using out-of-fold predictions."""
    def __init__(self, base=None, cv=3, seed=0):
        self.base, self.cv, self.seed = base, cv, seed

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        oof = cross_val_predict(clone(self.base), X, y, cv=StratifiedKFold(self.cv, shuffle=True, random_state=self.seed), method="predict_proba")[:, 1]
        self.cal_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(oof).reshape(-1, 1), y)
        self.base_ = clone(self.base).fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.cal_.predict_proba(_logit(self.base_.predict_proba(X)[:, 1]).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)

    def modality_weights(self, X):
        return self.base_.modality_weights(X)


def make_early(clf_name="Gradient Boosting", cols=None, seed=0):
    return _pipe(base_classifiers(seed)[clf_name], cols or ALL_FEATS)


def build_models(seed=0, hgb_params=None, include_torch=False):
    """Dictionary name -> estimator factory (lambda) for the complete comparison."""
    M = {}
    for g, c in GROUPS.items():
        M[f"{g}-only (GB)"] = (lambda c=c: _pipe(base_classifiers(seed)["Gradient Boosting"], c))
    M["Clinical LR (Bio+Demo)"] = lambda: _pipe(LogisticRegression(C=1, max_iter=3000), BIO + DEMO)
    for n in ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM (RBF)", "MLP"]:
        M[f"Early fusion: {n}"] = (lambda n=n: make_early(n, seed=seed))
    if hgb_params:
        M["Early fusion: GB (tuned)"] = lambda: _pipe(HistGradientBoostingClassifier(**{**dict(random_state=seed), **hgb_params}), ALL_FEATS)
    M["Late fusion (mean)"] = lambda: LateFusion("lr", False, seed=seed)
    M["Late fusion (stacked)"] = lambda: LateFusion("gb", True, hgb_params, seed)
    M["M-AIM (gated cross-modal)"] = lambda: GatedCrossModalFusion(hgb_params=None, seed=seed)
    M["M-AIM-Enhanced (tuned+calibrated)"] = lambda: SigmoidCalibrated(
        GatedCrossModalFusion(hgb_params=hgb_params, C=0.3, seed=seed), cv=3, seed=seed)
    if include_torch:
        try:
            from maim_torch import TorchMAIM
            M["M-AIM (PyTorch cross-attention)"] = lambda: TorchMAIM(GROUPS, epochs=60, seed=seed)
        except Exception:
            pass
    return M


def tune_hgb(X, y, n_iter=12, seed=0):
    """Randomised hyper-parameter search (AUC, 5-fold) for the boosting encoders -> used by M-AIM-Enhanced."""
    space = {"clf__learning_rate": [.02, .03, .05, .08, .12], "clf__max_depth": [2, 3, 4, 5],
             "clf__max_iter": [80, 120, 160], "clf__l2_regularization": [0, .1, 1, 5],
             "clf__min_samples_leaf": [10, 20, 40]}
    rs = RandomizedSearchCV(make_early("Gradient Boosting", seed=seed), space, n_iter=n_iter, scoring="roc_auc",
                            cv=StratifiedKFold(5, shuffle=True, random_state=seed), random_state=seed, n_jobs=-1)
    rs.fit(X, y)
    return {k.replace("clf__", ""): v for k, v in rs.best_params_.items()}, rs.best_score_


# --------------------------------------------------------------------------- metrics & statistics
def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn = ((yhat == 0) & (y == 0)).sum(); fp = ((yhat == 1) & (y == 0)).sum()
    fn = ((yhat == 0) & (y == 1)).sum(); tp = ((yhat == 1) & (y == 1)).sum()
    d = lambda a, b: a / b if b else np.nan
    return {"AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p), "Accuracy": accuracy_score(y, yhat),
            "Bal.Acc": balanced_accuracy_score(y, yhat), "Sensitivity": d(tp, tp + fn), "Specificity": d(tn, tn + fp),
            "PPV": d(tp, tp + fp), "NPV": d(tn, tn + fn), "F1": f1_score(y, yhat), "MCC": matthews_corrcoef(y, yhat),
            "Brier": brier_score_loss(y, p)}


def youden_threshold(y, p):
    ts = np.linspace(.05, .95, 91)
    j = [(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1)) + (((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1)) - 1 for t in ts]
    return ts[int(np.argmax(j))]


def cv_evaluate(factory, X, y, k=5, reps=2, seed=0):
    """Repeated stratified k-fold. Returns per-fold metrics (DataFrame) and out-of-fold mean probabilities."""
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=reps, random_state=seed)
    rows, oof, cnt = [], np.zeros(len(y)), np.zeros(len(y))
    ntr = nte = 0
    for f, (tr, te) in enumerate(rskf.split(X, y)):
        m = factory().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        rows.append({"fold": f, **metrics(y[te], p)})
        oof[te] += p; cnt[te] += 1; ntr, nte = len(tr), len(te)
    return pd.DataFrame(rows), oof / cnt, (ntr, nte)


def corrected_ttest(a, b, ntr, nte):
    """Nadeau & Bengio (2003) corrected resampled paired t-test on per-fold scores."""
    d = np.asarray(a) - np.asarray(b); k = len(d)
    v = d.var(ddof=1)
    if v == 0: return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + nte / ntr) * v)
    return t, 2 * stats.t.sf(abs(t), k - 1)


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out


def delong_test(y, p1, p2):
    """DeLong (1988) paired comparison of two correlated AUCs. Returns auc1, auc2, z, two-sided p."""
    y = np.asarray(y); pos = y == 1; m = pos.sum(); n = len(y) - m
    P = [np.asarray(p1), np.asarray(p2)]
    tx, ty, tz = [], [], []
    for p in P:
        tx.append(_midrank(p[pos])); ty.append(_midrank(p[~pos])); tz.append(_midrank(np.concatenate([p[pos], p[~pos]])))
    tx, ty, tz = map(np.array, (tx, ty, tz))
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0: return aucs[0], aucs[1], 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def mcnemar(y, p1, p2, thr=0.5):
    c1 = ((p1 >= thr) == y); c2 = ((p2 >= thr) == y)
    b = (c1 & ~c2).sum(); c = (~c1 & c2).sum()
    if b + c == 0: return 1.0
    return stats.binomtest(int(b), int(b + c), .5).pvalue


def bootstrap_ci(y, p, fn=roc_auc_score, B=500, seed=0):
    r = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = r.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        v.append(fn(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def hosmer_lemeshow(y, p, g=10):
    d = pd.DataFrame({"y": y, "p": p}); d["b"] = pd.qcut(d.p.rank(method="first"), g, labels=False)
    a = d.groupby("b").agg(o=("y", "sum"), e=("p", "sum"), n=("y", "size"))
    chi = (((a.o - a.e) ** 2) / (a.e * (1 - a.e / a.n)).clip(lower=1e-9)).sum()
    return chi, stats.chi2.sf(chi, g - 2), a


def calibration_slope_intercept(y, p):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def ece(y, p, bins=10):
    f, m = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return float(np.mean(np.abs(f - m)))


def decision_curve(y, p, ts=np.linspace(.01, .8, 80)):
    n = len(y); prev = y.mean()
    nb = [((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in ts]
    all_ = [prev - (1 - prev) * t / (1 - t) for t in ts]
    return ts, np.array(nb), np.array(all_)


def lake_louise_baseline(d):
    """Rule-based surrogate of the Updated Lake Louise Criteria: T2-based AND (T1-based OR LGE) positive."""
    t2 = d["T2_ms"] > 52; t1 = (d["T1_ms"] > 1030) | (d["ECV_pct"] > 30) | (d["LGE_pct"] > 4)
    return (t2 & t1).astype(float).values


def fairness_table(d, y, p, col, thr=0.5):
    rows = []
    for g, idx in d.groupby(col).groups.items():
        i = d.index.get_indexer(idx); yy, pp = y[i], p[i]
        if len(np.unique(yy)) < 2: continue
        m = metrics(yy, pp, thr)
        rows.append({"attribute": col, "group": str(g), "n": len(i), "AUC": m["AUC"], "TPR": m["Sensitivity"],
                     "FPR": 1 - m["Specificity"], "PosRate": (pp >= thr).mean()})
    t = pd.DataFrame(rows)
    return t


def group_permutation_importance(model, X, y, groups, reps=10, seed=0):
    r = np.random.default_rng(seed); base = roc_auc_score(y, model.predict_proba(X)[:, 1]); out = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(reps):
            Xp = X.copy(); perm = r.permutation(len(X)); Xp[cols] = X[cols].values[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[g] = (np.mean(drops), np.std(drops))
    return base, out


def linear_shap(X, y):
    """Exact SHAP values for a logistic model with independent-feature baseline: phi_j = coef_j * (z_j - mean z_j)."""
    pre = Pipeline([("i", SimpleImputer(strategy="median")), ("z", StandardScaler())])
    Z = pre.fit_transform(X); lr = LogisticRegression(C=1.0, max_iter=3000).fit(Z, y)
    return pd.DataFrame(lr.coef_[0] * (Z - Z.mean(0)), columns=X.columns), pd.DataFrame(Z, columns=X.columns)


def masked_modality_auc(model, X, y):
    res = {"all modalities": roc_auc_score(y, model.predict_proba(X)[:, 1])}
    for g, c in GROUPS.items():
        Xm = X.copy(); Xm[c] = np.nan
        res[f"without {g}"] = roc_auc_score(y, model.predict_proba(Xm)[:, 1])
    return res


# --------------------------------------------------------------------------- explainability demo (synthetic signals)
def demo_signals(n=500, seed=1):
    r = np.random.default_rng(seed)
    y = r.binomial(1, .5, n)
    yy, xx = np.mgrid[0:24, 0:24]
    imgs = np.zeros((n, 24, 24)); ecg = np.zeros((n, 200)); tt = np.linspace(0, 1, 200)
    for i in range(n):
        base = r.normal(0, .35, (24, 24))
        if y[i]:
            cy, cx = 12 + r.integers(-4, 5), 12 + r.integers(-4, 5)
            base += 1.2 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8)
        imgs[i] = base
        beat = np.exp(-((tt - .30) ** 2) / .0006) * 1.0 - .15 * np.exp(-((tt - .27) ** 2) / .0004) + .3 * np.exp(-((tt - .65) ** 2) / .01)
        if y[i]: beat += .35 * ((tt > .36) & (tt < .55))
        ecg[i] = beat + r.normal(0, .08, 200)
    return y, imgs, ecg


def occlusion_map(model, img, size=4):
    base = model.predict_proba(img.reshape(1, -1))[0, 1]; H, W = img.shape; m = np.zeros((H, W))
    for a in range(0, H, 2):
        for b in range(0, W, 2):
            t = img.copy(); t[a:a + size, b:b + size] = 0
            m[a:a + size, b:b + size] = np.maximum(m[a:a + size, b:b + size], base - model.predict_proba(t.reshape(1, -1))[0, 1])
    return m


# =========================================================================== shared utilities (M-AIM + TriCARD)
from itertools import combinations
from math import factorial


def _hgb(seed=0, **kw):
    p = dict(max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed); p.update(kw)
    return HistGradientBoostingClassifier(**p)


def _auc_any(y, P, classes):
    """AUC for binary (positive-class column) or macro one-vs-rest for multiclass."""
    if P.shape[1] == 2:
        return roc_auc_score(y, P[:, 1])
    return roc_auc_score(y, P, multi_class="ovr", labels=classes)


def mask_modalities(X, groups, tau, rng, keep_one=True):
    """Randomly blank whole modalities (NaN) with probability tau each; optionally guarantee >=1 modality survives."""
    Xm = X.copy(); n = len(X); names = list(groups)
    drop = rng.random((n, len(names))) < tau
    if keep_one:
        bad = drop.all(1); keep = rng.integers(0, len(names), n)
        drop[bad, keep[bad]] = False
    for j, g in enumerate(names):
        if drop[:, j].any():
            Xm.loc[drop[:, j], groups[g]] = np.nan
    return Xm


def tau_schedule(e, E, tau_max=0.5, warmup_frac=0.5):
    """Linear warm-up 0 -> tau_max, then cosine annealing back to 0 (curriculum modality dropout)."""
    w = max(1, int(E * warmup_frac))
    if e < w:
        return tau_max * (e + 1) / w
    return 0.5 * tau_max * (1 + np.cos(np.pi * (e - w) / max(1, E - w)))


class ModalityDropoutHGB(BaseEstimator, ClassifierMixin):
    """
    Boosted trees cannot be trained 'per epoch', so the curriculum is realised as stacked augmented copies:
    copy e is masked with dropout tau_e = tau_schedule(e). NaN-aware HistGradientBoosting then learns
    explicit missing-modality branches. Works for binary and multiclass targets.
    """
    def __init__(self, groups=None, tau_max=0.5, copies=8, seed=0):
        self.groups, self.tau_max, self.copies, self.seed = groups, tau_max, copies, seed

    def fit(self, X, y):
        self.cols_ = sum(self.groups.values(), []); rng = np.random.default_rng(self.seed)
        parts = [X[self.cols_]]
        for e in range(self.copies):
            parts.append(mask_modalities(X, self.groups, tau_schedule(e, self.copies, self.tau_max), rng)[self.cols_])
        self.m_ = _hgb(self.seed).fit(pd.concat(parts, ignore_index=True), np.tile(np.asarray(y), self.copies + 1))
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X[self.cols_])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def dropout_robustness(Xd, yd, Xt, yt, groups, taus=(0, .1, .2, .3, .4, .5), tau_max=0.5, copies=8, seed=0, reps=5):
    """Test-time modality-dropout curves: standard vs curriculum-trained fusion, plus unimodal models."""
    cols = sum(groups.values(), []); classes = np.unique(yd)
    std = _hgb(seed).fit(Xd[cols], yd)
    cur = ModalityDropoutHGB(groups, tau_max, copies, seed).fit(Xd, yd)
    uni = {g: _hgb(seed).fit(Xd[c], yd) for g, c in groups.items()}
    preds = [("Fusion, standard training", lambda X: std.predict_proba(X[cols])),
             ("Fusion, curriculum modality-dropout", cur.predict_proba)]
    preds += [(f"{g}-only", (lambda X, g=g: uni[g].predict_proba(X[groups[g]]))) for g in groups]
    rows = []
    for tau in taus:
        for name, fn in preds:
            a = []
            for r_ in range(reps if tau > 0 else 1):
                Xm = mask_modalities(Xt, groups, tau, np.random.default_rng(seed * 100 + r_)) if tau > 0 else Xt
                a.append(_auc_any(yt, fn(Xm), classes))
            rows.append({"tau": tau, "model": name, "AUC": float(np.mean(a)), "sd": float(np.std(a))})
    return pd.DataFrame(rows)


def coalition_shapley(v, players):
    """Exact Shapley values for a small player set. v(tuple_of_sorted_players) -> float. Efficiency: sum(phi)=v(all)-v(())."""
    cache, n = {}, len(players)
    val = lambda S: cache.setdefault(tuple(sorted(S)), v(tuple(sorted(S))))
    phi = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for r_ in range(n):
            for S in combinations(others, r_):
                phi[p] += factorial(r_) * factorial(n - r_ - 1) / factorial(n) * (val(S + (p,)) - val(S))
    return phi


def modality_shapley_table(Xd, yd, Xt, yt, groups, seed=0, class_names=None):
    """
    Data-derived modality attribution: value of a coalition S = one-vs-rest test AUC of a model trained on
    the modalities in S (v(empty)=0.5). Exact Shapley per class; alpha = max(phi,0)/sum(max(phi,0)).
    """
    classes = np.unique(yd); names = list(groups); models = {}
    def fit(S):
        if S not in models:
            cols = sum((groups[g] for g in S), []); models[S] = (cols, _hgb(seed).fit(Xd[cols], yd))
        return models[S]
    report = classes[1:] if len(classes) == 2 else classes
    rows = []
    for c in report:
        j = list(classes).index(c)
        def v(S, j=j, c=c):
            if not S: return 0.5
            cols, m = fit(S); return roc_auc_score((yt == c).astype(int), m.predict_proba(Xt[cols])[:, j])
        phi = coalition_shapley(v, names); pos = {k: max(x, 0) for k, x in phi.items()}; s = sum(pos.values()) or 1
        for g in names:
            rows.append({"class": class_names[c] if class_names else c, "modality": g, "phi": phi[g], "alpha": pos[g] / s, "AUC_all": v(tuple(sorted(names)))})
    return pd.DataFrame(rows)


def single_feature_audit(X, y, max_auc_flag=0.90):
    """Leakage screen: single-feature AUC and 'one-sided' binary flags (never present in one class)."""
    y = np.asarray(y); rows = []
    for c in X.columns:
        v = X[c]; ok = v.notna().values
        if ok.sum() < 20 or v[ok].nunique() < 2 or len(np.unique(y[ok])) < 2: continue
        a = roc_auc_score(y[ok], v[ok]); a = max(a, 1 - a)
        binary = set(pd.unique(v[ok])) <= {0, 1, 0.0, 1.0}
        row = {"feature": c, "AUC": a, "missing": 1 - ok.mean(), "binary": binary, "flag": ""}
        if binary:
            on = (v[ok] == 1).values; yy = y[ok]
            n0, n1 = int((on & (yy == 0)).sum()), int((on & (yy == 1)).sum())
            row.update(prevalence=on.mean(), n_in_class0=n0, n_in_class1=n1)
            if (n0 == 0 or n1 == 0) and on.mean() > 0.005:
                row["flag"] = "one-sided: never occurs in one class (possible label leakage)"
        if a >= max_auc_flag:
            row["flag"] = (row["flag"] + "; " if row["flag"] else "") + "single-feature AUC >= 0.90"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
