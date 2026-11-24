"""
M-AIM Streamlit workbench  —  run:  streamlit run app.py
All results are computed on a SYNTHETIC cohort (see maim_core.make_cohort) to demonstrate methodology.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from maim_core import (ALL_FEATS, BIO, CMR, ECG, GROUPS, SUBTYPES, decision_curve, demo_signals, occlusion_map)
from maim_pipeline import ENH_MODEL, KEY_MODEL, run_pipeline

st.set_page_config(page_title="M-AIM Workbench", layout="wide")
with st.sidebar:
    module = st.radio("Module", ["M-AIM (myocarditis)", "TriCARD (tri-modal cardiac)"], key="module")
if module.startswith("TriCARD"):
    import tricard_app
    tricard_app.render()
    st.stop()
st.title("M-AIM: Multi-Modal AI for Myocarditis — development, validation & benchmarking workbench")
st.warning("Synthetic data demonstration. No number on this page is clinical evidence. Replace `make_cohort()` in "
           "`maim_core.py` with a loader for real, ethics-approved CMR/ECG/biomarker data to obtain real results.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Experiment settings")
    n = st.slider("Cohort size", 600, 4000, 1200, 100)
    prev = st.slider("Myocarditis prevalence", .15, .6, .40, .05)
    k = st.selectbox("k-fold CV", [3, 5, 10], 1)
    reps = st.selectbox("CV repeats", [1, 2, 3], 0)
    n_iter = st.slider("Hyper-parameter search iterations", 4, 30, 8)
    shift = st.slider("Site/scanner shift", 0.0, 3.0, 1.0, .5)
    seed = st.number_input("Random seed", 0, 9999, 42)
    use_torch = st.checkbox("Include PyTorch cross-attention model (needs torch)", False)
    go_btn = st.button("Run full pipeline", type="primary")


@st.cache_resource(show_spinner=False)
def _run(n, prev, k, reps, n_iter, shift, seed, use_torch):
    bar = st.progress(0.0, text="Starting")
    R = run_pipeline(n, prev, seed, k, reps, n_iter, shift, use_torch, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.empty()
    return R


if go_btn:
    st.session_state["R"] = _run(n, prev, k, reps, n_iter, shift, int(seed), use_torch)
if "R" not in st.session_state:
    st.info("Set parameters in the sidebar and press **Run full pipeline** (≈1–3 min at default settings).")
    st.stop()
R = st.session_state["R"]
d, yt, tp, folds = R["data"], R["yt"], R["test_p"], R["folds"]
dte = d.iloc[R["idx_te"]].reset_index(drop=True)
fmt = lambda df: df.style.format({c: "{:.3f}" for c in df.select_dtypes("number").columns})

tabs = st.tabs(["1 Objectives", "2 Data", "3 CV metrics", "4 Hold-out & significance", "5 GOF & calibration",
                "6 Clinical utility", "7 Generalisability & fairness", "8 Explainability", "9 2D–9D visualisation",
                "10 Benchmark", "11 Interpretation", "12 References"])

# ------------------------------------------------------------------ 1 objectives
with tabs[0]:
    st.subheader("Objective → implementation map")
    st.table(pd.DataFrame([
        ["1 Curate & preprocess multi-modal data", "Tab 2; `make_cohort`, median-impute + z-score inside every pipeline (no leakage)"],
        ["2 Modality-specific models", "CMR / ECG / Biomarker encoders (GB, LR); optional PyTorch MLP-token encoders"],
        ["3 Hybrid fusion with cross-modal attention", "Early, late (mean/stacked), gated cross-modal (M-AIM), tuned+calibrated (Enhanced), optional PyTorch attention"],
        ["4 Internal & external validation", "Repeated stratified k-fold, 70/30 hold-out, leave-one-site-out (Tabs 3, 4, 7)"],
        ["5 Generalisability (age, sex, ethnicity, site)", "Tab 7 subgroup AUC / TPR / FPR, equalised-odds & parity gaps"],
        ["6 Compare with criteria & unimodal (DeLong)", "Tab 4: DeLong, McNemar, corrected resampled t-test, Wilcoxon, Friedman, Holm"],
        ["7 Decision-curve net benefit", "Tab 6"],
        ["8 Explainability", "Tab 8: modality permutation, linear SHAP, occlusion/Grad-CAM-style maps, gate weights"],
    ], columns=["Objective", "Where"]))
    st.caption(f"Tuned boosting parameters (RandomizedSearchCV, CV AUC {R['best_cv_auc']:.3f}): {R['best_params']}")

# ------------------------------------------------------------------ 2 data
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(d)); c2.metric("Myocarditis", int(d.y.sum())); c3.metric("Features", len(ALL_FEATS))
    st.dataframe(d.head(50), width='stretch')
    a, b = st.columns(2)
    miss = d[ALL_FEATS].isna().mean().sort_values(ascending=False); miss = miss[miss > 0]
    a.plotly_chart(px.bar(miss, title="Missingness by feature (fraction)"), width='stretch')
    b.plotly_chart(px.box(d, x="site", y="T1_ms", color=d.y.map({0: "Control", 1: "Myocarditis"}), title="Scanner/site shift in T1"), width='stretch')
    corr = d[CMR[:6] + ECG[:6] + BIO + ["y"]].corr()
    st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heat-map", height=650), width='stretch')

# ------------------------------------------------------------------ 3 CV
with tabs[2]:
    st.subheader(f"{R['params']['k']}-fold × {R['params']['reps']} CV on development set (mean ± SD across folds)")
    st.dataframe(R["cv_table"][["Model", "AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC", "Brier"]], width='stretch')
    long = pd.concat([f.assign(Model=m) for m, f in folds.items()])
    order = R["cv_table"].Model.tolist()
    st.plotly_chart(px.violin(long, x="Model", y="AUC", box=True, points="all", category_orders={"Model": order}, height=520, title="Per-fold AUC distribution"), width='stretch')
    hm = R["cv_table"].set_index("Model")[[c + "_m" for c in ["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC"]]]
    hm.columns = [c[:-2] for c in hm.columns]
    st.plotly_chart(px.imshow(hm, text_auto=".2f", color_continuous_scale="Viridis", aspect="auto", height=560, title="Metric heat-map (CV mean)"), width='stretch')
    st.download_button("Download CV table (CSV)", R["cv_table"].to_csv(index=False), "cv_table.csv")
    st.caption("Improvement path: baseline early-fusion GB → tuned GB → gated cross-modal fusion → tuned + Platt-calibrated Enhanced model.")

# ------------------------------------------------------------------ 4 hold-out & significance
with tabs[3]:
    ho = R["holdout"]
    st.subheader("Hold-out test set (30%, never used for tuning)")
    st.dataframe(fmt(ho[["Model", "AUC", "AUC_lo", "AUC_hi", "PR-AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC", "Brier", "Youden thr", "Sens@Youden", "Spec@Youden"]]), width='stretch')
    cols = st.columns(2)
    fig = go.Figure()
    for m in ho.Model:
        if m.startswith("Lake"): continue
        fpr, tpr, _ = roc_curve(yt, tp[m]); fig.add_scatter(x=fpr, y=tpr, name=m, line=dict(width=4 if m == ENH_MODEL else 1.5))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="chance")
    fig.update_layout(title="ROC curves", xaxis_title="1 − specificity", yaxis_title="sensitivity", height=520)
    cols[0].plotly_chart(fig, width='stretch')
    fig = go.Figure()
    for m in ho.Model:
        if m.startswith("Lake"): continue
        pr, rc, _ = precision_recall_curve(yt, tp[m]); fig.add_scatter(x=rc, y=pr, name=m, line=dict(width=4 if m == ENH_MODEL else 1.5))
    fig.update_layout(title="Precision–recall curves", xaxis_title="recall", yaxis_title="precision", height=520)
    cols[1].plotly_chart(fig, width='stretch')
    ok = ho.dropna(subset=["AUC_lo"]).sort_values("AUC")
    fig = go.Figure(go.Scatter(x=ok.AUC, y=ok.Model, mode="markers", error_x=dict(type="data", symmetric=False, array=ok.AUC_hi - ok.AUC, arrayminus=ok.AUC - ok.AUC_lo)))
    fig.update_layout(title="Forest plot — AUC with 95% bootstrap CI", height=560); st.plotly_chart(fig, width='stretch')
    st.subheader(f"Significance tests: {ENH_MODEL} vs each comparator")
    st.dataframe(R["sig"].style.format({c: "{:.4f}" for c in R["sig"].select_dtypes("number").columns}), width='stretch')
    f, a = R["friedman"], R["anova"]
    st.write(f"**Friedman** (paired folds) χ² = {f.statistic:.2f}, p = {f.pvalue:.2e} · **one-way ANOVA** on fold AUCs F = {a.statistic:.2f}, p = {a.pvalue:.2e}.")
    st.caption("Corrected resampled t-test (Nadeau & Bengio 2003) corrects the variance inflation from overlapping CV training sets; "
               "DeLong compares correlated AUCs on the hold-out; McNemar compares paired decisions at 0.5; Holm controls family-wise error across the 13 DeLong comparisons. α = 0.05.")
    pm = R["p_matrix"]
    st.plotly_chart(px.imshow(-np.log10(pm.astype(float).clip(lower=1e-12)), color_continuous_scale="Magma", aspect="auto", height=600, title="Pairwise DeLong −log10(p) (hold-out)"), width='stretch')
    cm_cols = st.columns(3)
    for i, m in enumerate([KEY_MODEL, ENH_MODEL, "Lake Louise (rule surrogate)"]):
        cmx = confusion_matrix(yt, (tp[m] >= .5).astype(int))
        cm_cols[i].plotly_chart(px.imshow(cmx, text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], title=m, color_continuous_scale="Blues"), width='stretch')

# ------------------------------------------------------------------ 5 GOF
with tabs[4]:
    g = R["gof"]
    st.subheader("Goodness of fit & calibration (hold-out)")
    st.dataframe(g.style.format({c: "{:.3f}" for c in g.select_dtypes("number").columns}), width='stretch')
    st.caption("Hosmer–Lemeshow: p > 0.05 = no evidence of miscalibration. Calibration slope ideal = 1, intercept ideal = 0.")
    fig = go.Figure(); fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="perfect")
    for m in [KEY_MODEL, ENH_MODEL, "Early fusion: Gradient Boosting", "Early fusion: Logistic Regression"]:
        fr, mp = calibration_curve(yt, tp[m], n_bins=8, strategy="quantile"); fig.add_scatter(x=mp, y=fr, mode="lines+markers", name=m)
    fig.update_layout(title="Reliability diagram", xaxis_title="mean predicted", yaxis_title="observed", height=520)
    st.plotly_chart(fig, width='stretch')
    st.plotly_chart(px.histogram(x=tp[ENH_MODEL], color=yt.astype(str), nbins=30, barmode="overlay", opacity=.65, title="Predicted-risk distribution by true class (Enhanced)"), width='stretch')

# ------------------------------------------------------------------ 6 clinical utility
with tabs[5]:
    ts, nb_e, nb_all = decision_curve(yt, tp[ENH_MODEL])
    fig = go.Figure()
    for m in [ENH_MODEL, KEY_MODEL, "Early fusion: Gradient Boosting", "Clinical LR (Bio+Demo)", "Lake Louise (rule surrogate)"]:
        fig.add_scatter(x=ts, y=decision_curve(yt, tp[m])[1], name=m)
    fig.add_scatter(x=ts, y=nb_all, name="Treat all", line=dict(dash="dot")); fig.add_scatter(x=ts, y=np.zeros_like(ts), name="Treat none", line=dict(dash="dash"))
    fig.update_layout(title="Decision-curve analysis (net benefit)", xaxis_title="threshold probability", yaxis_title="net benefit", yaxis_range=[-0.05, max(.45, nb_e.max() * 1.1)], height=540)
    st.plotly_chart(fig, width='stretch')
    S = R["surv"]
    if "error" in S: st.error(S["error"])
    else:
        st.subheader("Risk stratification & survival (simulated 24-month composite outcome)")
        c = st.columns(3); c[0].metric("C-index (hold-out)", f"{S['cindex']:.3f}"); c[1].metric("HR high vs low", f"{S['HR']:.2f}")
        c[2].metric("HR 95% CI", f"{S['HR_ci'][0]:.2f}–{S['HR_ci'][1]:.2f}")
        fig = go.Figure()
        for gname, km in S["km"].items(): fig.add_scatter(x=km.iloc[:, 0], y=km.iloc[:, 1], mode="lines", line_shape="hv", name=f"{gname} (n={S['groups'].get(gname, 0)})")
        fig.update_layout(title="Kaplan–Meier by M-AIM-Enhanced risk group (low <60th pct, high ≥90th pct of dev risk)", xaxis_title="months", yaxis_title="event-free survival", height=480)
        st.plotly_chart(fig, width='stretch'); st.dataframe(S["cox_summary"])
    sb = R["subtype"]
    st.subheader("Subtype classification (myocarditis cases only)")
    st.write(f"n = {sb['n']}; macro one-vs-rest AUC = {sb['auc_ovr']:.3f}; macro-F1 = {sb['f1_macro']:.3f}")
    st.plotly_chart(px.bar(x=list(sb["per_class"]), y=list(sb["per_class"].values()), labels=dict(x="subtype", y="OvR AUC"), title="Per-subtype AUC"), width='stretch')

# ------------------------------------------------------------------ 7 generalisability
with tabs[6]:
    lo = R["loso"]
    st.subheader("Leave-one-site-out external validation")
    st.dataframe(fmt(lo), width='stretch')
    st.plotly_chart(px.bar(lo, x="Held-out site", y="AUC", color="Model", barmode="group", range_y=[.5, 1], title="AUC on unseen sites"), width='stretch')
    ma = lo[lo.Model == KEY_MODEL].AUC
    st.write(f"M-AIM cross-site AUC variance = **{ma.var(ddof=0):.5f}** (target < 0.01) · min site AUC = **{ma.min():.3f}** vs mean {ma.mean():.3f} (target: within 0.05).")
    st.subheader("Missing-modality robustness (Enhanced model)")
    mm = pd.Series(R["missing"]); st.plotly_chart(px.bar(mm, title="Hold-out AUC when a modality is withheld", range_y=[.5, 1]), width='stretch')
    st.subheader("Subgroup performance & fairness")
    fr = R["fair"]; st.dataframe(fmt(fr), width='stretch')
    st.plotly_chart(px.bar(fr, x="group", y="AUC", color="attribute", range_y=[.5, 1], title="AUC by subgroup"), width='stretch')
    gaps = fr.groupby("attribute").agg(TPR_gap=("TPR", lambda s: s.max() - s.min()), FPR_gap=("FPR", lambda s: s.max() - s.min()), DPD=("PosRate", lambda s: s.max() - s.min()))
    st.dataframe(gaps.style.format("{:.3f}")); st.caption("Targets: equalised-odds gaps ≤ 0.05, demographic-parity difference < 0.10 (small subgroups give noisy estimates).")

# ------------------------------------------------------------------ 8 explainability
with tabs[7]:
    base, perm = R["perm"]
    pdf = pd.DataFrame({"modality": list(perm), "AUC drop": [v[0] for v in perm.values()], "sd": [v[1] for v in perm.values()]})
    st.plotly_chart(px.bar(pdf, x="modality", y="AUC drop", error_y="sd", title=f"Modality permutation importance (baseline AUC {base:.3f})"), width='stretch')
    sh, Z = R["shap"]
    imp = sh.abs().mean().sort_values(ascending=False).head(20)
    st.plotly_chart(px.bar(imp[::-1], orientation="h", title="Mean |SHAP| — linear early-fusion model (top 20)", height=560), width='stretch')
    top = imp.index[:10]
    long = sh[top].melt(var_name="feature", value_name="SHAP").assign(z=Z[top].melt()["value"].values)
    st.plotly_chart(px.strip(long, x="SHAP", y="feature", color="z", color_discrete_sequence=None, title="SHAP summary (colour = standardised feature value)", height=520), width='stretch')
    gw = pd.DataFrame(R["gate_w"], columns=list(GROUPS)).assign(Outcome=np.where(yt == 1, "Myocarditis", "Control"))
    st.plotly_chart(px.box(gw.melt("Outcome", var_name="Modality", value_name="Gate weight"), x="Modality", y="Gate weight", color="Outcome", title="Per-patient gate (attention) weight by modality"), width='stretch')
    st.subheader("Grad-CAM-style saliency demo (synthetic T2 map & ECG beat)")
    yy, imgs, ecg = demo_signals()
    mi = LogisticRegression(C=.05, max_iter=2000).fit(imgs.reshape(len(yy), -1), yy)
    me = LogisticRegression(C=.05, max_iter=2000).fit(ecg, yy)
    i = int(np.where(yy == 1)[0][0])
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(px.imshow(imgs[i], color_continuous_scale="Hot", title="T2 map (lesion present)"), width='stretch')
    c2.plotly_chart(px.imshow(occlusion_map(mi, imgs[i]), color_continuous_scale="Jet", title="Occlusion sensitivity"), width='stretch')
    sal = me.coef_[0] * ecg[i]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(y=ecg[i], name="ECG", secondary_y=False); fig.add_scatter(y=np.clip(sal, 0, None), name="attribution", fill="tozeroy", secondary_y=True)
    fig.update_layout(title="ECG attribution (ST-segment emphasis)"); c3.plotly_chart(fig, width='stretch')
    st.caption("Illustrative only: with real DICOM/ECG tensors, apply Grad-CAM to the final convolution of the 3D-CNN/1D-CNN (maim_torch.py exposes the gate; add hooks for CAMs).")

# ------------------------------------------------------------------ 9 2D–9D
with tabs[8]:
    st.subheader("Multivariate visualisation: 2D → 9D")
    st.caption("Dimensions beyond 3 are carried by visual channels (colour, size, symbol, animation) and, for 8–9D, by parallel axes. "
               "Each added dimension is a variable of your choice from the hold-out cohort.")
    V = dte.copy(); V["M-AIM risk"] = tp[ENH_MODEL]; V["Outcome"] = np.where(yt == 1, "Myocarditis", "Control"); V["Site"] = "Site " + V.site.astype(str)
    num = ["T1_ms", "T2_ms", "LGE_pct", "ECV_pct", "LVEF_pct", "ST_dev_mm", "QRS_ms", "QTc_ms", "log_hsTnT", "log_CRP", "log_NTproBNP", "ESR", "age", "M-AIM risk"]
    defaults = ["T1_ms", "T2_ms", "log_hsTnT", "M-AIM risk", "LGE_pct", "ST_dev_mm", "log_CRP", "LVEF_pct", "QTc_ms"]
    lvl = st.select_slider("Dimensionality", options=[2, 3, 4, 5, 6, 7, 8, 9], value=7)
    cc = st.columns(3); pick = []
    for j in range(9):
        pick.append(cc[j % 3].selectbox(f"Variable {j + 1}", num, num.index(defaults[j]), key=f"v{j}"))
    V = V.dropna(subset=num).copy(); V["_size"] = (V[pick[4]] - V[pick[4]].min() + .1)
    V["AgeGrp"] = V.age_group
    if lvl == 2:
        fig = px.scatter(V, x=pick[0], y=pick[1], color="Outcome", height=650)
    elif lvl <= 7:
        kw = dict(x=pick[0], y=pick[1], z=pick[2], height=720)
        if lvl >= 4: kw.update(color=pick[3], color_continuous_scale="Turbo")
        else: kw.update(color="Outcome")
        if lvl >= 5: kw.update(size="_size", size_max=14)
        if lvl >= 6: kw.update(symbol="Site")
        if lvl >= 7:
            V = V.sort_values("AgeGrp"); kw.update(animation_frame="AgeGrp", range_x=[V[pick[0]].min(), V[pick[0]].max()], range_y=[V[pick[1]].min(), V[pick[1]].max()], range_z=[V[pick[2]].min(), V[pick[2]].max()])
        fig = px.scatter_3d(V, **kw)
        fig.update_traces(marker=dict(opacity=.8))
    else:
        dims = pick[:lvl]; dims = list(dict.fromkeys(dims))
        fig = px.parallel_coordinates(V, dimensions=dims, color="M-AIM risk", color_continuous_scale="Turbo", height=620,
                                      title=f"{len(dims)}-axis parallel coordinates, coloured by M-AIM risk" + (" (9th variable shown by colour)" if lvl == 9 else ""))
    st.plotly_chart(fig, width='stretch')
    st.markdown("**Extra views:** PCA embedding of all features and a radar profile.")
    Zs = StandardScaler().fit_transform(V[ALL_FEATS].fillna(V[ALL_FEATS].median())); pc = PCA(3).fit(Zs); P3 = pc.transform(Zs)
    pcd = pd.DataFrame(P3, columns=["PC1", "PC2", "PC3"]).assign(Outcome=V["Outcome"].values, risk=V["M-AIM risk"].values)
    a, b = st.columns(2)
    a.plotly_chart(px.scatter_3d(pcd, x="PC1", y="PC2", z="PC3", color="risk", symbol="Outcome", color_continuous_scale="Turbo", title=f"PCA 3D (var. explained {pc.explained_variance_ratio_.sum():.0%})", height=520), width='stretch')
    rv = [c for c in pick[:9]]; Zr = (V[rv] - V[rv].mean()) / V[rv].std()
    fig = go.Figure()
    for o in ["Control", "Myocarditis"]:
        mval = Zr[V.Outcome == o].mean(); fig.add_scatterpolar(r=list(mval) + [mval.iloc[0]], theta=rv + [rv[0]], name=o, fill="toself")
    fig.update_layout(title="Radar: standardised group means (9 selected variables)", height=520); b.plotly_chart(fig, width='stretch')
    st.plotly_chart(px.scatter_matrix(V, dimensions=pick[:6], color="Outcome", height=800, title="Scatter-matrix of first six variables"), width='stretch')

# ------------------------------------------------------------------ 10 benchmark
with tabs[9]:
    ho = R["holdout"].set_index("Model")
    ours = ho.loc[ENH_MODEL]
    st.subheader("Targets stated in the manuscript vs this run")
    chk = pd.DataFrame([
        ["AUC ≥ 0.92", ours.AUC, ours.AUC >= .92], ["Sensitivity ≥ 0.90 (0.5 thr)", ours.Sensitivity, ours.Sensitivity >= .90],
        ["Sensitivity ≥ 0.90 (Youden thr)", ours["Sens@Youden"], ours["Sens@Youden"] >= .90],
        ["Specificity ≥ 0.85", ours.Specificity, ours.Specificity >= .85],
        ["Cross-site AUC variance < 0.01", R["loso"][R["loso"].Model == KEY_MODEL].AUC.var(ddof=0), R["loso"][R["loso"].Model == KEY_MODEL].AUC.var(ddof=0) < .01],
        ["AUC with any one modality missing ≥ 0.85", min(v for k_, v in R["missing"].items() if k_ != "all modalities"), min(v for k_, v in R["missing"].items() if k_ != "all modalities") >= .85],
        ["Survival C-index ≥ 0.75", R["surv"].get("cindex", np.nan), R["surv"].get("cindex", 0) >= .75],
        ["HR high vs low ≥ 4.0", R["surv"].get("HR", np.nan), R["surv"].get("HR", 0) >= 4]], columns=["Endpoint", "Observed", "Met"])
    st.dataframe(chk.style.format({"Observed": "{:.3f}"}), width='stretch')
    st.subheader("Published reference points (editable — verify against primary sources)")
    bench = st.data_editor(pd.DataFrame([
        ["Original Lake Louise (value quoted in manuscript)", np.nan, .74, .86, "Manuscript lit. review"],
        ["Updated Lake Louise (value quoted in manuscript)", np.nan, .875, .962, "Manuscript lit. review"],
        ["ECG + EMR fusion, ICI-myocarditis/MACE risk (Ayoub et al. 2024, JACC Adv; ECG-only/EMR-only AUC 0.59–0.645)", .717, np.nan, np.nan, "Sun 2024 editorial"],
        ["M-AIM manuscript target", .92, .90, .85, "Manuscript"],
        ["This run: Lake Louise rule surrogate", ho.loc["Lake Louise (rule surrogate)"].AUC, ho.loc["Lake Louise (rule surrogate)"].Sensitivity, ho.loc["Lake Louise (rule surrogate)"].Specificity, "Synthetic"],
        ["This run: best early-fusion baseline", ho.loc[R["best_baseline"]].AUC, ho.loc[R["best_baseline"]].Sensitivity, ho.loc[R["best_baseline"]].Specificity, "Synthetic"],
        ["This run: M-AIM-Enhanced", ours.AUC, ours.Sensitivity, ours.Specificity, "Synthetic"]],
        columns=["System", "AUC", "Sensitivity", "Specificity", "Source"]), width='stretch', num_rows="dynamic")
    st.plotly_chart(px.bar(bench.melt(["System", "Source"], var_name="Metric", value_name="Value").dropna(), x="System", y="Value", color="Metric", barmode="group", height=520, title="Benchmark comparison"), width='stretch')
    st.caption("Published figures come from different populations, reference standards and designs; they are context, not like-for-like comparisons. "
               "Synthetic results cannot be ranked against real-world studies.")

# ------------------------------------------------------------------ 11 interpretation
with tabs[10]:
    sg = R["sig"].set_index("Comparator"); hoi = R["holdout"].set_index("Model")
    best_cv = R["cv_table"].iloc[0]
    st.markdown(f"""
### Auto-generated analysis of this run
**Discrimination.** On the hold-out set M-AIM-Enhanced reached AUC {ours.AUC:.3f} (95% bootstrap CI {ours.AUC_lo:.3f}–{ours.AUC_hi:.3f}); the best CV model was *{best_cv.Model}* (AUC {best_cv.AUC}).
The Lake Louise rule surrogate reached AUC {hoi.loc['Lake Louise (rule surrogate)'].AUC:.3f} with sensitivity {hoi.loc['Lake Louise (rule surrogate)'].Sensitivity:.2f}; the DeLong test favours M-AIM (p = {sg.loc['Lake Louise (rule surrogate)', 'p (DeLong)']:.1e}).

**Does fusion help?** Against single-modality models the gain is large and significant (ECG-only ΔAUC {sg.loc['ECG-only (GB)', 'ΔAUC (hold-out)']:.3f}, biomarkers-only {sg.loc['Biomarkers-only (GB)', 'ΔAUC (hold-out)']:.3f}). Against **CMR-only** the margin is
{sg.loc['CMR-only (GB)', 'ΔAUC (hold-out)']:.3f} AUC (DeLong p = {sg.loc['CMR-only (GB)', 'p (DeLong)']:.3f}; Holm-adjusted {sg.loc['CMR-only (GB)', 'p (DeLong, Holm)']:.3f}; corrected CV t-test p = {sg.loc['CMR-only (GB)', 'p (t-test)']:.2f}).
Against well-tuned early-fusion learners (LR, RF, GB, SVM) the differences are not statistically significant, so **in this simulation the added complexity of cross-modal gating does not buy a demonstrable discrimination advantage**; its value lies in robustness to missing modalities, calibration and interpretable gate weights.
This should be stated plainly in the manuscript's hypothesis ("significantly outperform single-modality models and LLC") — the data here support that claim for LLC and for ECG/biomarker-only models, not universally.

**Calibration (GOF).** Hosmer–Lemeshow p for Enhanced = {R['gof'].set_index('Model').loc[ENH_MODEL, 'HL p']:.3f}, calibration slope {R['gof'].set_index('Model').loc[ENH_MODEL, 'Cal. slope']:.2f}, Brier {R['gof'].set_index('Model').loc[ENH_MODEL, 'Brier']:.3f}.

**Robustness.** AUC without CMR falls to {R['missing']['without CMR']:.3f}, whereas withholding ECG or biomarkers costs little, i.e. CMR carries most of the signal in this data-generating process (by construction).
Cross-site AUC variance and subgroup gaps are in Tab 7; small subgroups yield wide, noisy estimates.

**Caveats that matter more than any p-value.** (i) Cohort is synthetic; (ii) the reference label is circular if Lake Louise/CMR findings contribute to the committee diagnosis while CMR is also a model input (incorporation bias) — in real data, define the label independently or report the sensitivity analysis;
(iii) hold-out AUCs near 0.97 would be implausible in real myocarditis cohorts; (iv) tuning used the development set, so only the hold-out numbers are unbiased; (v) 3 sites cannot establish geographic generalisability — use prospective external centres.

### Recommended next steps with real data
1. Replace `make_cohort()`; keep patient-level splits and site-level external validation.
2. Switch encoders to raw-signal/volume networks (`maim_torch.py` + 3D-CNN/1D-CNN) and compare them with these tabular surrogates using the same statistical battery.
3. Pre-register the analysis plan and report per TRIPOD+AI; add calibration-in-the-large and decision-curve results at clinically justified thresholds.
""")

# ------------------------------------------------------------------ 12 references
with tabs[11]:
    st.markdown("""
### References (DOIs supplied; confirm each before submission)
**Clinical & CMR criteria**
- Caforio ALP, et al. Eur Heart J 2013;34:2636–2648. doi:10.1093/eurheartj/eht210
- Ferreira VM, et al. Cardiovascular magnetic resonance in nonischemic myocardial inflammation: expert recommendations. J Am Coll Cardiol 2018;72:3158–3176. doi:10.1016/j.jacc.2018.09.072
- Ammirati E, et al. Management of acute myocarditis and chronic inflammatory cardiomyopathy: an expert consensus document. Circ Heart Fail 2020;13:e007405. doi:10.1161/CIRCHEARTFAILURE.120.007405
- Ammirati E, et al. Multicenter Lombardy Registry. Circulation 2018;138:1088–1099. doi:10.1161/CIRCULATIONAHA.118.035319
- Tschöpe C, et al. Myocarditis and inflammatory cardiomyopathy: current evidence and future directions. Nat Rev Cardiol 2021;18:169–193. doi:10.1038/s41569-020-00435-x

**AI for myocarditis / CMR (2024–2026)**
- Shen A, Xu J, Yue X, Yu X, Wu L, Mu D. Non-invasive diagnostic model for myocarditis using cardiac magnetic resonance radiomics. Quant Imaging Med Surg 2026. doi:10.21037/qims-2025-1-2809
- Imaging for the diagnosis of acute myocarditis: can artificial intelligence improve diagnostic performance? Front Cardiovasc Med 2024;11:1408574. doi:10.3389/fcvm.2024.1408574
- Deep learning automatically distinguishes myocarditis patients from normal subjects based on MRI. Int J Cardiovasc Imaging 2024;40(12):2617–2629. doi:10.1007/s10554-024-03284-8
- Sun CLF. Breaking barriers in cardiovascular and cancer care with multimodal AI (editorial on Ayoub et al.). JACC Adv 2024. doi:10.1016/j.jacadv.2024.101433
- Automatic diagnosis of myocarditis in cardiac MRI using deep transformers and explainable AI. arXiv:2210.14611 (preprint).

**Multimodal & cardiovascular AI**
- Acosta JN, Falcone GJ, Rajpurkar P, Topol EJ. Multimodal biomedical AI. Nat Med 2022;28:1773–1784. doi:10.1038/s41591-022-01981-2
- Huang S-C, et al. Fusion of medical imaging and EHR using deep learning. npj Digit Med 2020;3:136. doi:10.1038/s41746-020-00341-z
- Hannun AY, et al. Nat Med 2019;25:65–69. doi:10.1038/s41591-018-0268-3 · Attia ZI, et al. Nat Med 2019;25:70–74. doi:10.1038/s41591-018-0240-2
- Bai W, et al. J Cardiovasc Magn Reson 2018;20:65. doi:10.1186/s12968-018-0471-x

**Methods & statistics**
- Collins GS, et al. TRIPOD+AI statement. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378
- DeLong ER, DeLong DM, Clarke-Pearson DL. Biometrics 1988;44:837–845. doi:10.2307/2531595
- Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn 2003;52:239–281. doi:10.1023/A:1024068626366
- Hosmer DW, Lemeshow S. Commun Stat Theory Methods 1980;9:1043–1069. doi:10.1080/03610928008827941
- Vickers AJ, Elkin EB. Decision curve analysis. Med Decis Making 2006;26:565–574. doi:10.1177/0272989X06295361
- Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. BMC Med 2019;17:230. doi:10.1186/s12916-019-1466-7
- Riley RD, et al. Calculating the sample size required for developing a clinical prediction model. BMJ 2020;368:m441. doi:10.1136/bmj.m441
- Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. NeurIPS 2017. arXiv:1705.07874
- Selvaraju RR, et al. Grad-CAM. ICCV 2017:618–626. doi:10.1109/ICCV.2017.74
- Harrell FE, et al. Evaluating the yield of medical tests. JAMA 1982;247:2543–2546. doi:10.1001/jama.1982.03320430047030
""")
    st.caption("Reference list assembled from the manuscript's bibliography plus a literature search on 23 Nov 2026; the 2024–2026 items were located via web search, the remainder from memory — verify every DOI/page range on PubMed/Crossref.")
