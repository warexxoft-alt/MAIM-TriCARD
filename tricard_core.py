"""
TriCARD experiments: PTB-XL-style benchmark with leakage audit & patient-level splitting, unsupervised structure,
tri-modal (ECG+PCG+Echo) fusion with data-derived Shapley attribution and curriculum modality-dropout.
Cohorts are SYNTHETIC unless real PTB-XL metadata is loaded with `load_ptbxl_metadata`.
"""
import ast
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, silhouette_score, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import maim_core as mc

# --------------------------------------------------------------------------- PTB-XL-style metadata
META = ["age", "sex", "height", "weight", "pacemaker"]
FLAGS = ["QRS_LVH", "LI_LVH", "S12_LVH", "QRS_CLBB", "SLI_LVH", "RS_LVH", "R56_LVH"]
ECGM = ["HR_bpm", "PR_ms", "QRS_ms", "QTc_ms", "ST_dev_mm", "R_amp_mV", "axis_deg"]
FEATSETS = {
    "A: metadata + diagnostic flags (chapter setup)": META + FLAGS,
    "B: metadata only": META,
    "C: metadata + measured ECG intervals": META + ECGM,
    "D: C + diagnostic flags (leaky upper bound)": META + ECGM + FLAGS,
}


def make_ptbxl_like(n=6000, seed=0, leak=1.0, age_outliers=True):
    """Synthetic PTB-XL-like table: repeated patients, official-style strat_fold (patient-level), partial weight, 'age=300' coding, diagnostic flags."""
    r = np.random.default_rng(seed); n_pat = int(n * 0.865); pid = r.integers(0, n_pat, n)
    u = r.normal(0, 1, n_pat); age_p = np.clip(r.normal(60, 17, n_pat), 18, 95); sex_p = r.binomial(1, .48, n_pat)
    qrs_off = r.normal(0, 6, n_pat); hr_off = r.normal(0, 7, n_pat); fold_p = r.integers(1, 11, n_pat)
    age = np.clip(age_p[pid] + r.integers(0, 3, n), 18, 95); sex = sex_p[pid]
    logit = 0.25 + 0.9 * u[pid] + 0.03 * (age - 60); ab = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    d = pd.DataFrame({"patient_id": pid, "strat_fold": fold_p[pid], "abnormal": ab, "age": age, "sex": sex})
    d["height"] = np.where(sex == 1, r.normal(176, 8, n), r.normal(163, 7, n)); d["weight"] = np.where(r.random(n) < 0.55, np.nan, r.normal(75, 15, n))
    d["pacemaker"] = (r.random(n) < 0.0134).astype(int)
    p = lambda a_, b_: np.where(ab == 1, a_ * leak + (1 - leak) * b_, b_)
    d["QRS_CLBB"] = ((ab == 1) & (r.random(n) < 0.18 * leak)).astype(int)            # never present in NORM (as in the chapter)
    d["QRS_LVH"] = (r.random(n) < p(.40, .25)).astype(int); d["LI_LVH"] = (r.random(n) < p(.22, .10)).astype(int)
    d["S12_LVH"] = (r.random(n) < p(.15, .08)).astype(int); d["SLI_LVH"] = (r.random(n) < p(.12, .05)).astype(int)
    d["RS_LVH"] = (r.random(n) < p(.03, .015)).astype(int); d["R56_LVH"] = (r.random(n) < p(.03, .015)).astype(int)
    d["HR_bpm"] = r.normal(72, 11, n) + hr_off[pid] + 4 * ab; d["PR_ms"] = r.normal(160, 22, n) + 8 * ab
    d["QRS_ms"] = r.normal(92, 9, n) + qrs_off[pid] + 8 * ab + 35 * d.QRS_CLBB; d["QTc_ms"] = r.normal(415, 24, n) + 10 * ab
    d["ST_dev_mm"] = r.normal(0.1, 0.25, n) + 0.25 * ab; d["R_amp_mV"] = r.normal(1.2, 0.4, n) + 0.2 * ab * (d.QRS_LVH == 1); d["axis_deg"] = r.normal(40, 35, n) + 12 * ab
    if age_outliers:
        old = (r.random(n) < 0.006); d.loc[old, "age"] = 300     # PTB-XL codes ages >89 as 300 (de-identification)
    return d


def clean_age(a):
    """PTB-XL de-identification: ages > 89 are stored as 300. Treat as missing (use an indicator) instead of letting them dominate mean/SD."""
    a = a.copy().astype(float); a[a > 120] = np.nan; return a


def load_ptbxl_metadata(csv_file):
    """Minimal real-data loader for ptbxl_database.csv (+ optional wfdb signals later). Label: NORM with likelihood>=50 -> normal.
    DOES NOT create diagnostic-flag columns (they would be label-derived)."""
    d = pd.read_csv(csv_file, index_col=0)
    codes = d["scp_codes"].apply(ast.literal_eval)
    d["abnormal"] = (~codes.apply(lambda c: c.get("NORM", 0) >= 50)).astype(int)
    d = d.rename(columns={"patient_id": "patient_id"})
    d["pacemaker"] = d.get("pacemaker", pd.Series(0, index=d.index)).notna().astype(int) if "pacemaker" in d else 0
    d["patient_id"] = d["patient_id"].astype(int)
    return d[[c for c in ["patient_id", "strat_fold", "abnormal", "age", "sex", "height", "weight", "pacemaker"] if c in d.columns]]


def _models(seed=0):
    m = {"Logistic Regression": LogisticRegression(max_iter=3000),
         "Random Forest": RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
         "GradBoost": GradientBoostingClassifier(random_state=seed),
         "SVM (RBF)": SVC(probability=True, random_state=seed),
         "MLP": MLPClassifier((64, 32), early_stopping=True, max_iter=500, random_state=seed),
         "KNN (k=7)": KNeighborsClassifier(7)}
    try:
        from xgboost import XGBClassifier
        m = {"XGBoost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9, eval_metric="logloss", n_jobs=2, random_state=seed), **m}
    except ImportError:
        pass
    return m


def _pipe(clf):
    return Pipeline([("i", SimpleImputer(strategy="median")), ("z", StandardScaler()), ("c", clf)])


def run_ptbxl_benchmark(df, seed=0, k=5, featsets=None):
    df = df.copy(); df["age"] = clean_age(df["age"]); y = df.abnormal.values
    tr, te = (df.strat_fold <= 8).values, (df.strat_fold == 10).values
    featsets = featsets or FEATSETS; rows, probs = [], {}
    for fs, cols in featsets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols: continue
        Xtr, ytr, Xte, yte = df.loc[tr, cols], y[tr], df.loc[te, cols], y[te]; grp = df.loc[tr, "patient_id"].values
        for name, clf in _models(seed).items():
            cv = []
            for a, b in GroupKFold(k).split(Xtr, ytr, grp):
                cv.append(roc_auc_score(ytr[b], _pipe(clone_est(clf)).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]))
            m = _pipe(clone_est(clf)).fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; mt = mc.metrics(yte, p)
            lo, hi = mc.bootstrap_ci(yte, p, B=200)
            rows.append({"Feature set": fs, "Model": name, "AUC (grouped CV)": np.mean(cv), "CV SD": np.std(cv, ddof=1), "AUC test": mt["AUC"], "AUC lo": lo, "AUC hi": hi,
                         "Accuracy": mt["Accuracy"], "F1": mt["F1"], "MCC": mt["MCC"], "AP": mt["PR-AUC"],
                         "Sens (abnormal recall)": mt["Sensitivity"], "Spec (normal recall)": mt["Specificity"]})
            probs[(fs, name)] = p
    res = pd.DataFrame(rows)
    out = {"table": res, "probs": probs, "y_te": y[te], "n_train": int(tr.sum()), "n_test": int(te.sum())}
    # leakage audit on training data (all candidate columns)
    allc = [c for c in META + FLAGS + ECGM if c in df.columns]
    out["audit"] = mc.single_feature_audit(df.loc[tr, allc], y[tr])
    # patient-overlap inflation: random record split vs patient-grouped split (same model/features)
    cols = [c for c in FEATSETS["C: metadata + measured ECG intervals"] if c in df.columns]; X = df[cols]; pids = df.patient_id.values; r_ = []
    for s in range(5):
        a, b = train_test_split(np.arange(len(df)), test_size=.2, stratify=y, random_state=s); overlap = len(set(pids[a]) & set(pids[b]))
        r_.append({"split": "random record-level", "AUC": roc_auc_score(y[b], _pipe(GradientBoostingClassifier(random_state=s)).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]), "patients in both": overlap})
        gk = list(GroupKFold(5).split(X, y, pids))[s]
        r_.append({"split": "patient-grouped", "AUC": roc_auc_score(y[gk[1]], _pipe(GradientBoostingClassifier(random_state=s)).fit(X.iloc[gk[0]], y[gk[0]]).predict_proba(X.iloc[gk[1]])[:, 1]), "patients in both": 0})
    out["split_compare"] = pd.DataFrame(r_)
    # learning curve (feature set C, GradBoost, grouped by stratified folds)
    Xl, yl = df.loc[tr, cols], y[tr]
    ts, trs, vas = learning_curve(_pipe(GradientBoostingClassifier(random_state=seed)), Xl, yl, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                  scoring="roc_auc", train_sizes=np.linspace(.1, 1, 6))
    out["learning"] = pd.DataFrame({"n": ts, "train": trs.mean(1), "train_sd": trs.std(1), "val": vas.mean(1), "val_sd": vas.std(1)})
    # permutation importance, best leak-free model
    clean = res[~res["Feature set"].str.startswith(("A", "D"))].sort_values("AUC test", ascending=False).iloc[0]
    ccols = [c for c in featsets[clean["Feature set"]] if c in df.columns]
    clf = _pipe(clone_est(_models(seed)[clean["Model"]])).fit(df.loc[tr, ccols], y[tr])
    pi = permutation_importance(clf, df.loc[te, ccols], y[te], scoring="roc_auc", n_repeats=10, random_state=seed)
    out["perm"] = pd.DataFrame({"feature": ccols, "AUC drop": pi.importances_mean, "sd": pi.importances_std}).sort_values("AUC drop", ascending=False)
    out["best_clean"] = (clean["Feature set"], clean["Model"], clean["AUC test"])
    out["unsup_A"] = unsupervised(df, FEATSETS["A: metadata + diagnostic flags (chapter setup)"], seed)
    out["unsup_C"] = unsupervised(df, FEATSETS["C: metadata + measured ECG intervals"], seed)
    return out


def clone_est(e):
    from sklearn.base import clone
    return clone(e)


def unsupervised(df, cols, seed=0, n=2500):
    cols = [c for c in cols if c in df.columns]; d = df.copy(); d["age"] = clean_age(d["age"])
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(d[cols])); y = d.abnormal.values
    pca = PCA(min(10, len(cols))).fit(Z); idx = np.random.default_rng(seed).choice(len(d), min(n, len(d)), replace=False)
    P = pca.transform(Z[idx])[:, :5]; rows = []
    for k in range(2, 9):
        lab = KMeans(k, n_init=10, random_state=seed).fit(P)
        rows.append({"k": k, "Inertia": lab.inertia_, "Silhouette": silhouette_score(P, lab.labels_), "ARI vs label": adjusted_rand_score(y[idx], lab.labels_), "NMI vs label": normalized_mutual_info_score(y[idx], lab.labels_)})
    return {"evr": pca.explained_variance_ratio_, "table": pd.DataFrame(rows), "pc": pca.transform(Z[idx])[:, :3], "y": y[idx], "n_unique_rows": int(pd.DataFrame(Z).round(6).drop_duplicates().shape[0])}


# --------------------------------------------------------------------------- tri-modal cohort
CLASSES = ["Normal sinus rhythm", "Atrial fibrillation", "Aortic stenosis", "Ventricular hypertrophy", "Heart failure"]
# per-class shifts in SD units: (NSR, AF, AS, VH, HF)
_SHIFTS = {
    "ECG": {"ECG_HR": (0, 1.1, 0, 0, .7), "ECG_irregularity": (0, 3, 0, 0, .3), "ECG_Pwave": (0, -3, 0, 0, 0), "ECG_QRS": (0, 0, .4, .8, .8),
            "ECG_Rvoltage": (0, 0, .6, 2.2, -.2), "ECG_STT": (0, 0, .4, 1.0, .8), "ECG_QTc": (0, 0, .2, .4, .5)},
    "PCG": {"PCG_sys_murmur": (0, .3, 3.0, .5, .8), "PCG_S2_amp": (0, 0, -2.0, 0, -.3), "PCG_S3": (0, 0, 0, .2, 2.0), "PCG_domfreq": (0, 0, .8, 0, 0)},
    "Echo": {"Echo_LVEF": (0, 0, -.3, -.2, -2.8), "Echo_wall": (0, 0, .8, 2.5, .3), "Echo_AVgrad": (0, 0, 3.2, .3, 0), "Echo_LAvol": (0, 1.0, .8, .8, 1.8), "Echo_Eee": (0, .2, .6, 1.0, 2.0)},
}
TRI_GROUPS = {g: list(v) + [f"{g}_emb{i}" for i in range(3)] + [f"{g}_nuis"] for g, v in _SHIFTS.items()}


def make_trimodal(n=2500, seed=0, sep=1.0, label_noise=0.03):
    """Synthetic standardised tri-modal features whose class shifts encode textbook physiology (assumption-driven, not data-driven)."""
    r = np.random.default_rng(seed); y = r.choice(5, n, p=[.30, .20, .20, .15, .15]); d = {}
    for g, feats in _SHIFTS.items():
        for f, sh in feats.items():
            d[f] = np.asarray(sh)[y] * sep + r.normal(0, 1, n)
        for i in range(3):
            d[f"{g}_emb{i}"] = r.normal(0, 1, (5,))[y] * 0.35 * sep + r.normal(0, 1, n)
        d[f"{g}_nuis"] = r.normal(0, 1, n)
    X = pd.DataFrame(d); flip = r.random(n) < label_noise; y = np.where(flip, r.integers(0, 5, n), y)
    return X, y


CHAPTER_T17 = pd.DataFrame({"ECG": [.40, .60, .20, .45, .35], "PCG": [.35, .25, .45, .15, .20], "Echo": [.25, .15, .35, .40, .45]}, index=CLASSES)


def run_trimodal(n=2500, seed=0, sep=1.0, copies=8, tau_max=0.5):
    X, y = make_trimodal(n, seed, sep); Xd, Xt, yd, yt = train_test_split(X, y, test_size=.3, stratify=y, random_state=seed)
    G = TRI_GROUPS; names = list(G); classes = np.unique(yd); rows = []; fitted = {}
    from itertools import combinations
    for r_ in range(1, 4):
        for S in combinations(names, r_):
            cols = sum((G[g] for g in S), []); m = mc._hgb(seed).fit(Xd[cols], yd); P = m.predict_proba(Xt[cols]); fitted[S] = P
            rows.append({"Modalities": " + ".join(S), "k": r_, "Macro AUC (OvR)": roc_auc_score(yt, P, multi_class="ovr"), "Accuracy": accuracy_score(yt, P.argmax(1)),
                         "Macro F1": f1_score(yt, P.argmax(1), average="macro"), **{f"AUC {CLASSES[j][:12]}": roc_auc_score((yt == j).astype(int), P[:, j]) for j in range(5)}})
    perf = pd.DataFrame(rows).sort_values("Macro AUC (OvR)", ascending=False)
    # paired bootstrap: tri-modal vs best unimodal
    best_uni = perf[perf.k == 1].iloc[0]["Modalities"]; Pf = fitted[tuple(names)]; Pu = fitted[(best_uni,)]; rb = np.random.default_rng(seed); diffs = []
    for _ in range(300):
        i = rb.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) < 5: continue
        diffs.append(roc_auc_score(yt[i], Pf[i], multi_class="ovr") - roc_auc_score(yt[i], Pu[i], multi_class="ovr"))
    diffs = np.array(diffs); boot = {"best_unimodal": best_uni, "delta": float(np.mean(diffs)), "ci": tuple(np.percentile(diffs, [2.5, 97.5])), "p": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}
    shap = mc.modality_shapley_table(Xd, yd, Xt, yt, G, seed, class_names=CLASSES)
    alpha = shap.pivot(index="class", columns="modality", values="alpha").loc[CLASSES, names]
    cmp = pd.DataFrame({"mean |Δα| vs chapter Table 17": (alpha - CHAPTER_T17[names]).abs().mean(1), "dominant (data-derived)": alpha.idxmax(1), "dominant (chapter)": CHAPTER_T17.idxmax(1)})
    drop = mc.dropout_robustness(Xd, yd, Xt, yt, G, tau_max=tau_max, copies=copies, seed=seed)
    return {"perf": perf, "boot": boot, "shap": shap, "alpha": alpha, "alpha_cmp": cmp, "drop": drop, "n": n, "sep": sep}


# --------------------------------------------------------------------------- benchmark table (corrected labelling)
def benchmark_table(pt=None, tri=None):
    rows = [
        ["PACFNet (Lin et al. 2025)", "ECG + PCG", "EPHNOGRAM", "CVD detection", .945, "chapter-reported", "No (different data/task)"],
        ["HCTG-Net (Xiong et al. 2025)", "ECG", "MIT-BIH", "Arrhythmia", .964, "chapter-reported", "No"],
        ["DDR-Net (Zhu & Li 2025)", "ECG + PCG", "CinC 2016", "CVD detection", .970, "chapter-reported", "No"],
        ["1D-CNN (Ahmed et al. 2023)", "ECG", "dataset to verify (chapter: PTB-XL)", "Arrhythmia", .993, "chapter-reported - verify dataset", "No"],
        ["EchoNet-Dynamic (Ouyang et al. 2020)", "Echo video", "EchoNet-Dynamic", "LVEF / HFrEF", .940, "chapter-reported - verify metric", "No"],
        ["TriCARD XGBoost (chapter)", "Metadata only (not tri-modal)", "PTB-XL subsample", "Normal/abnormal", .8135, "chapter-reported", "Reference for this work"],
    ]
    if pt is not None:
        t = pt["table"]
        for fs in FEATSETS:
            s = t[t["Feature set"] == fs]
            if len(s): b = s.sort_values("AUC test", ascending=False).iloc[0]; rows.append([f"This work: {b['Model']} [{fs[:1]}]", "Metadata/measured ECG (synthetic)", "synthetic PTB-XL-like", "Normal/abnormal", b["AUC test"], "computed here", "Synthetic - not comparable"])
    if tri is not None:
        p = tri["perf"].iloc[0]; rows.append([f"This work: tri-modal fusion ({p['Modalities']})", "ECG+PCG+Echo features (synthetic)", "synthetic", "5-class", p["Macro AUC (OvR)"], "computed here", "Synthetic - not comparable"])
    return pd.DataFrame(rows, columns=["System", "Modality", "Dataset", "Task", "AUC", "Source", "Like-for-like?"])
