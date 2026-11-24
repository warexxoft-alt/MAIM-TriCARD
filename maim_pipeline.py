"""End-to-end M-AIM experiment: tuning, repeated k-fold CV, hold-out test, statistics, generalisability, survival."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from maim_core import *  # noqa
from maim_core import _logit

KEY_MODEL = "M-AIM (gated cross-modal)"
ENH_MODEL = "M-AIM-Enhanced (tuned+calibrated)"


def run_pipeline(n=1200, prev=0.40, seed=42, k=5, reps=1, n_iter=8, shift=1.0, use_torch=False, progress=None):
    P = lambda f, t: progress(f, t) if progress else None
    R = {"params": dict(n=n, prev=prev, seed=seed, k=k, reps=reps, n_iter=n_iter, shift=shift)}
    d = make_cohort(n, prev, seed, 3, shift); X = d[ALL_FEATS]; y = d.y.values
    R["data"] = d
    dev, te = train_test_split(np.arange(n), test_size=.3, stratify=y, random_state=seed)
    Xd, yd, Xt, yt = X.iloc[dev], y[dev], X.iloc[te], y[te]
    R.update(idx_dev=dev, idx_te=te)
    P(.03, "Hyper-parameter search (RandomizedSearchCV)")
    best, best_cv = tune_hgb(Xd, yd, n_iter, seed); R["best_params"], R["best_cv_auc"] = best, best_cv
    models = build_models(seed, best, use_torch)
    folds, oof, tp, fitted = {}, {}, {}, {}
    for i, (name, fac) in enumerate(models.items()):
        P(.05 + .75 * i / len(models), f"Cross-validating: {name}")
        f, o, sz = cv_evaluate(fac, Xd, yd, k, reps, seed)
        folds[name], oof[name], R["split_sizes"] = f, o, sz
        m = fac().fit(Xd, yd); fitted[name] = m; tp[name] = m.predict_proba(Xt)[:, 1]
    R.update(folds=folds, oof=oof, test_p=tp, fitted=fitted, yt=yt, yd=yd, Xt=Xt, Xd=Xd)

    # ---- tables
    cv_rows = []
    for nme, f in folds.items():
        row = {"Model": nme}
        for c in ["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC", "Brier"]:
            row[c] = f"{f[c].mean():.3f} ± {f[c].std():.3f}"; row[c + "_m"] = f[c].mean()
        cv_rows.append(row)
    R["cv_table"] = pd.DataFrame(cv_rows).sort_values("AUC_m", ascending=False)

    ho = []
    for nme, p in tp.items():
        thr = youden_threshold(yd, oof[nme]); m = metrics(yt, p); my = metrics(yt, p, thr)
        lo, hi = bootstrap_ci(yt, p)
        ho.append({"Model": nme, **m, "AUC_lo": lo, "AUC_hi": hi, "Youden thr": thr,
                   "Sens@Youden": my["Sensitivity"], "Spec@Youden": my["Specificity"]})
    ll = lake_louise_baseline(d.iloc[te]); mm = metrics(yt, ll)
    ho.append({"Model": "Lake Louise (rule surrogate)", **mm, "AUC_lo": np.nan, "AUC_hi": np.nan, "Youden thr": .5,
               "Sens@Youden": mm["Sensitivity"], "Spec@Youden": mm["Specificity"]})
    tp["Lake Louise (rule surrogate)"] = ll
    R["holdout"] = pd.DataFrame(ho).sort_values("AUC", ascending=False)

    # ---- significance
    ref_candidates = [m for m in folds if m.startswith("Early fusion")]
    best_base = max(ref_candidates, key=lambda m: folds[m]["AUC"].mean()); R["best_baseline"] = best_base
    sig = []
    for nme in tp:
        if nme == ENH_MODEL: continue
        row = {"Comparator": nme}
        if nme in folds and len(folds[nme]) == len(folds[ENH_MODEL]):
            t, pv = corrected_ttest(folds[ENH_MODEL]["AUC"], folds[nme]["AUC"], *R["split_sizes"])
            row.update({"ΔAUC (CV)": folds[ENH_MODEL]["AUC"].mean() - folds[nme]["AUC"].mean(), "t (corrected)": t, "p (t-test)": pv})
            try: row["p (Wilcoxon)"] = stats.wilcoxon(folds[ENH_MODEL]["AUC"], folds[nme]["AUC"]).pvalue
            except Exception: row["p (Wilcoxon)"] = np.nan
        a1, a2, z, pz = delong_test(yt, tp[ENH_MODEL], tp[nme])
        row.update({"ΔAUC (hold-out)": a1 - a2, "z (DeLong)": z, "p (DeLong)": pz, "p (McNemar)": mcnemar(yt, tp[ENH_MODEL], tp[nme])})
        sig.append(row)
    sg = pd.DataFrame(sig)
    pcols = [c for c in sg.columns if c.startswith("p (")]
    R["sig"] = sg
    # Holm-Bonferroni on DeLong p-values
    pv = sg["p (DeLong)"].values; o = np.argsort(pv); adj = np.empty_like(pv); run = 0
    for r_, i in enumerate(o): run = max(run, (len(pv) - r_) * pv[i]); adj[i] = min(1, run)
    sg["p (DeLong, Holm)"] = adj
    names = list(folds)
    mat = np.array([[delong_test(yt, tp[a], tp[b])[3] if a != b else np.nan for b in names] for a in names])
    R["p_matrix"] = pd.DataFrame(mat, index=names, columns=names)
    mdl = [m for m in names if len(folds[m]) == len(folds[names[0]])]
    A = np.column_stack([folds[m]["AUC"].values for m in mdl])
    R["friedman"] = stats.friedmanchisquare(*A.T)
    R["anova"] = stats.f_oneway(*A.T)
    # ---- goodness of fit
    gof = []
    for nme in names:
        p = tp[nme]; chi, pv_, _ = hosmer_lemeshow(yt, p); s, i_ = calibration_slope_intercept(yt, p)
        gof.append({"Model": nme, "HL χ²": chi, "HL p": pv_, "Cal. slope": s, "Cal. intercept": i_, "ECE": ece(yt, p), "Brier": brier_score_loss(yt, p)})
    R["gof"] = pd.DataFrame(gof).sort_values("Brier")

    # ---- site-wise external validation (leave-one-site-out) & missing modality
    P(.82, "Leave-one-site-out external validation")
    sites = []
    for s in sorted(d.site.unique()):
        tr = np.where(d.site != s)[0]; ts = np.where(d.site == s)[0]
        for nm, fac in [("Early fusion: GB", lambda: make_early("Gradient Boosting", seed=seed)), (KEY_MODEL, lambda: GatedCrossModalFusion(seed=seed))]:
            m = fac().fit(X.iloc[tr], y[tr]); p = m.predict_proba(X.iloc[ts])[:, 1]
            sites.append({"Held-out site": f"Site {s}", "Model": nm, "n": len(ts), **metrics(y[ts], p)})
    R["loso"] = pd.DataFrame(sites)
    R["missing"] = masked_modality_auc(fitted[ENH_MODEL], Xt, yt)

    # ---- fairness (Enhanced model, hold-out)
    dt = d.iloc[te].reset_index(drop=True); pe = tp[ENH_MODEL]
    R["fair"] = pd.concat([fairness_table(dt.assign(sex=dt.sex.map({0: "F", 1: "M"})), yt, pe, c) for c in ["sex", "age_group", "ethnicity", "site"]], ignore_index=True)

    # ---- importance
    P(.88, "Explainability")
    R["perm"] = group_permutation_importance(fitted[ENH_MODEL], Xt, yt, GROUPS, 8, seed)
    R["shap"] = linear_shap(Xd, yd)
    R["gate_w"] = fitted[KEY_MODEL].modality_weights(Xt)

    # ---- multiclass subtype (myocarditis patients)
    dm = d[d.y == 1]; Xm = dm[ALL_FEATS]; ym = pd.Categorical(dm.subtype, SUBTYPES).codes
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.pipeline import Pipeline
    clf = Pipeline([("pre", make_early("Logistic Regression", seed=seed).named_steps["pre"]), ("c", LogisticRegression(max_iter=3000, C=0.5))])
    pr = cross_val_predict(clf, Xm, ym, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")
    R["subtype"] = dict(auc_ovr=roc_auc_score(ym, pr, multi_class="ovr", average="macro"), f1_macro=f1_score(ym, pr.argmax(1), average="macro"),
                        per_class={SUBTYPES[j]: roc_auc_score((ym == j).astype(int), pr[:, j]) for j in range(4)}, n=len(ym))

    # ---- survival / risk stratification
    P(.93, "Survival analysis")
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.utils import concordance_index
        lo_d = pd.DataFrame({"risk": _logit(oof[ENH_MODEL]), "T": d.time_months.values[dev], "E": d.event.values[dev]})
        cox = CoxPHFitter().fit(lo_d, "T", "E")
        rt = _logit(tp[ENH_MODEL]); dtst = pd.DataFrame({"risk": rt, "T": d.time_months.values[te], "E": d.event.values[te]})
        q60, q90 = np.quantile(_logit(oof[ENH_MODEL]), [.6, .9])
        dtst["grp"] = np.where(rt >= q90, "High", np.where(rt >= q60, "Intermediate", "Low"))
        cidx = concordance_index(dtst["T"], -dtst["risk"], dtst["E"])
        sub = dtst[dtst.grp.isin(["High", "Low"])].assign(high=lambda z: (z.grp == "High").astype(int))
        try:
            hr = CoxPHFitter().fit(sub[["T", "E", "high"]], "T", "E"); HR = float(np.exp(hr.params_["high"])); ci = np.exp(hr.confidence_intervals_.loc["high"].values)
            hrp = float(hr.summary.loc["high", "p"])
        except Exception:
            HR, ci, hrp = np.nan, (np.nan, np.nan), np.nan
        km = {}
        for g in ["Low", "Intermediate", "High"]:
            s = dtst[dtst.grp == g]
            if len(s) > 2: kmf = KaplanMeierFitter().fit(s["T"], s["E"], label=g); km[g] = kmf.survival_function_.reset_index()
        R["surv"] = dict(cindex=cidx, HR=HR, HR_ci=tuple(ci), HR_p=hrp, km=km, groups=dtst.grp.value_counts().to_dict(),
                         cox_summary=cox.summary[["coef", "exp(coef)", "p"]])
    except Exception as e:  # lifelines missing
        R["surv"] = dict(error=str(e))
    P(1.0, "Done")
    return R
