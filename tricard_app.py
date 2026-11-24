"""TriCARD Streamlit module — imported by app.py (sidebar: Module -> TriCARD)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve

import tricard_core as tc
import tricard_signals as ts

fmt = lambda df, p=3: df.style.format({c: f"{{:.{p}f}}" for c in df.select_dtypes("number").columns})
LEADC = px.colors.qualitative.Set2


def render():
    st.title("TriCARD: Tri-Modal Cardiac AI — signal lab, PTB-XL benchmark, fusion & robustness")
    st.warning("Everything here runs on SYNTHETIC data (or files you upload). The chapter's own numbers are shown as 'chapter-reported' and re-examined, "
               "not reproduced: its recordings, images and PTB-XL table are not available to this app.")
    with st.sidebar:
        st.header("TriCARD settings")
        n_pt = st.slider("PTB-XL-like records", 2000, 12000, 5000, 500)
        n_tri = st.slider("Tri-modal cohort size", 1200, 5000, 2000, 200)
        sep = st.slider("Class separability (tri-modal)", .5, 1.5, 1.0, .1)
        k = st.selectbox("Grouped CV folds", [3, 5], 0, key="tk")
        seed = int(st.number_input("Seed", 0, 9999, 7, key="tseed"))
        run = st.button("Run TriCARD experiments", type="primary")
    if run:
        with st.spinner("Benchmark, leakage audit, tri-modal fusion, Shapley attribution, dropout curves (≈1–2 min)…"):
            df = tc.make_ptbxl_like(n_pt, seed)
            st.session_state["T"] = dict(df=df, pt=tc.run_ptbxl_benchmark(df, seed, k), tri=tc.run_trimodal(n_tri, seed, sep))
    T = st.session_state.get("T")
    tabs = st.tabs(["A Objectives & audit of chapter", "B ECG signal lab", "C Strip & echo image lab", "D Breathing/PCG audio lab",
                    "E PTB-XL benchmark", "F Leakage audit & splits", "G Unsupervised structure", "H Tri-modal fusion & Shapley",
                    "I Modality-dropout robustness", "J Benchmark & interpretation", "K References"])

    # ------------------------------------------------------------------ A
    with tabs[0]:
        st.subheader("What the chapter claims, what this module implements, and what the analysis found")
        st.table(pd.DataFrame([
            ["TC.2 Butterworth 0.5–40 Hz", "`tricard_signals.bandpass` (zero-phase SOS)", "Implemented"],
            ["TC.3 R-peak/HRV", "`detect_r_peaks`, `hrv_metrics`, Poincaré, optional RR cleaning", "Implemented; adds a duration-reliability flag"],
            ["PSD / ACF / STFT / correlation", "Tab B", "Implemented"],
            ["TC.17 MFCC, breathing rate", "`mfcc` (20 coef., 25 ms), `breathing_analysis` (peaks + ACF)", "Implemented; validated on known rates"],
            ["Strip/echo image statistics, Sobel, quality", "Tab C; `echo_quality` replaces ill-posed SNR/contrast", "Implemented with corrections"],
            ["PTB-XL 7-classifier benchmark", "Tab E; patient-grouped CV on official-style folds", "Implemented"],
            ["Feature importance, PCA, K-means", "Tabs E, G with label-agreement (ARI/NMI)", "Implemented"],
            ["Tri-modal fusion (TC.42–51)", "Tab H (NaN-aware boosted fusion over modality blocks)", "Surrogate (no deep encoders)"],
            ["Curriculum modality dropout (TC.59–63)", "`ModalityDropoutHGB`, `tau_schedule`, measured curves", "Implemented and measured, not assumed"],
            ["Shapley modality attribution (TC.68–73)", "`modality_shapley_table` — exact coalition Shapley", "Implemented; chapter's Table 17 was literature-set"],
        ], columns=["Chapter element", "Implementation", "Status"]))
        st.markdown("""
#### Issues found in the chapter that this module tests or corrects
1. **Table 17 is not computed.** Its note says the weights are "literature-informed", yet the text calls them Shapley values satisfying TC.5. Tab H computes exact Shapley attribution from fitted models and compares it with Table 17.
2. **Figure 32 (dropout) is a simulation.** Tab I trains and tests real models under modality masking, including standard vs curriculum-trained fusion.
3. **Possible label leakage.** QRS-CLBB (and other diagnostic flags) are derived from the same SCP statements that define NORM. A classifier fed these flags partly re-reads the label. Tab F screens for one-sided flags and compares feature sets with and without them.
4. **Patient overlap.** PTB-XL has 21,801 records from 18,869 patients; random record splits put one patient in train and test. Tab F quantifies the inflation; Tab E uses grouped CV and official folds.
5. **Age 300.** Table 5 reports an age range of 2–300 and mean 62.8 ± 32.3 — the 300 is PTB-XL's de-identification code for ages >89. Treat as missing, not as data.
6. **Sensitivity/specificity swapped.** Section 4.7.3/Table 14 call the abnormal-class recall "specificity" and the normal-class recall "sensitivity". With abnormal as the positive class these are reversed; the app reports both explicitly.
7. **HRV on 5 s.** RMSSD of 482–521 ms from 6–7 beats is a detection/duration artefact; Tab B shows it vanishes on 60 s and with RR cleaning.
8. **Echo SNR/contrast.** Negative-dB "SNR" and contrast ≡ 1.000 for every view are not informative; Tab C uses CNR, tissue SNR, Laplacian variance and Michelson contrast.
9. **Heart-rate cross-check.** Lead-2 HR 95.2 bpm vs echo 83 bpm is described as "cross-validating"; a 12 bpm gap from different moments is a consistency check, not validation.
10. **Reference/benchmark hygiene.** Table 16 mixes tasks, datasets and metrics (e.g. LVEF regression vs classification AUC); "multi-modal consistently outperforms unimodal" needs matched comparisons. "Projected AUC ≥ 0.95" is a target, not a result. The 2017 ATS reference is for the six-minute-walk test, not respiratory rate norms; Devereux 2004 is cited for ECG-vs-echo LVH prevalence but is a prognostic LV-mass study.
""")
        st.caption("Chapter-internal inconsistencies worth fixing: the text lists seven classifiers but ranks LR (0.8127) above RF (0.8133); 'GradBoost best-balanced' is asserted though its sens/spec are not tabulated; "
                   "Table 14 (N = 600) uses a 3,000-record subsample, so a 95% CI on AUC 0.8135 is roughly ±0.03 and the 0.801–0.817 spread among classifiers is within noise.")

    # ------------------------------------------------------------------ B ECG
    with tabs[1]:
        st.subheader("ECG signal laboratory")
        src = st.radio("Source", ["Synthetic", "Upload CSV (rows = samples, columns = leads)"], horizontal=True)
        c = st.columns(4); fs = c[0].number_input("fs (Hz)", 100, 2000, 500); dur = c[1].slider("Duration (s)", 5, 60, 5); hr = c[2].slider("Heart rate", 45, 140, 80)
        irr = c[3].checkbox("Irregular rhythm (AF-like)", False)
        if src == "Synthetic":
            ecg = ts.synth_ecg(fs, dur, 5, hr, irregular=irr, seed=3)
        else:
            up = st.file_uploader("ECG CSV", type=["csv", "txt"])
            if up is None: st.info("Upload a file or switch to synthetic."); st.stop()
            ecg = pd.read_csv(up, header=None).select_dtypes("number").values.T.astype(float)
        L, N = ecg.shape; t = np.arange(N) / fs; filt = ts.bandpass(ecg, fs); nm = [f"Lead {i + 1}" for i in range(L)]
        fig = make_subplots(L, 1, shared_xaxes=True, subplot_titles=nm, vertical_spacing=.04)
        for i in range(L):
            fig.add_scatter(x=t[:int(10 * fs)], y=ecg[i, :int(10 * fs)], line=dict(color="lightgrey"), name="raw", showlegend=i == 0, row=i + 1, col=1)
            fig.add_scatter(x=t[:int(10 * fs)], y=filt[i, :int(10 * fs)], line=dict(color=LEADC[i % 8]), name="filtered", showlegend=i == 0, row=i + 1, col=1)
        fig.update_layout(height=170 * L + 60, title="Raw vs 0.5–40 Hz band-passed (first 10 s)"); st.plotly_chart(fig, width="stretch")
        st.dataframe(fmt(ts.descriptive_stats(ecg), 4), width="stretch")
        a, b = st.columns(2)
        fig = go.Figure()
        for i in range(L):
            f_, p_, dom = ts.psd(ecg[i], fs); m = f_ <= 80; fig.add_scatter(x=f_[m], y=10 * np.log10(p_[m] + 1e-12), name=f"{nm[i]} (dom {dom:.2f} Hz = {dom * 60:.0f} bpm)")
        fig.update_layout(title="Welch PSD", xaxis_title="Hz", yaxis_title="dB"); a.plotly_chart(fig, width="stretch")
        fig = go.Figure(); lag = np.arange(301) / fs
        for i in range(L): fig.add_scatter(x=lag, y=ts.acf(ecg[i]), name=nm[i])
        ci = 1.96 / np.sqrt(N); fig.add_hline(y=ci, line_dash="dash", line_color="red"); fig.add_hline(y=-ci, line_dash="dash", line_color="red")
        fig.update_layout(title="Autocorrelation (±1.96/√N)", xaxis_title="lag (s)"); b.plotly_chart(fig, width="stretch")
        st.markdown("**R-peaks and HRV — effect of window length and RR cleaning**")
        lead = st.selectbox("Lead", nm, 1); i = nm.index(lead); rp = ts.detect_r_peaks(filt[i], fs)
        fig = go.Figure(); m5 = t <= 10
        fig.add_scatter(x=t[m5], y=filt[i][m5], name="filtered"); r10 = rp[rp / fs <= 10]; fig.add_scatter(x=r10 / fs, y=filt[i][r10], mode="markers", marker=dict(color="red", size=9), name="R-peaks")
        fig.update_layout(title=f"{lead}: {len(rp)} beats detected in {N / fs:.0f} s", xaxis_title="s"); st.plotly_chart(fig, width="stretch")
        rows = []
        for w in [5, 10, 30, N / fs]:
            if w > N / fs + 1e-9: continue
            sel = rp[rp / fs <= w]
            for cl in (False, True):
                rows.append({"Window (s)": round(w), "RR cleaned": cl, **ts.hrv_metrics(sel, fs, w, cl)})
        st.dataframe(fmt(pd.DataFrame(rows), 1), width="stretch")
        rr = np.diff(rp) / fs * 1000
        if len(rr) > 3:
            fig = px.scatter(x=rr[:-1], y=rr[1:], labels=dict(x="RR(n) ms", y="RR(n+1) ms"), title="Poincaré plot"); fig.add_shape(type="line", x0=rr.min(), y0=rr.min(), x1=rr.max(), y1=rr.max(), line=dict(dash="dash"))
            st.plotly_chart(fig, width="stretch")
        f_, tt, S = ts.stft_db(filt[i], fs); mm = f_ <= 75
        st.plotly_chart(px.imshow(S[mm], x=tt, y=f_[mm], origin="lower", aspect="auto", color_continuous_scale="Plasma", title="STFT spectrogram (dB), 128/96", labels=dict(x="s", y="Hz")), width="stretch")
        corr = np.corrcoef(ecg); st.plotly_chart(px.imshow(corr, x=nm, y=nm, zmin=-1, zmax=1, color_continuous_scale="RdBu", text_auto=".2f", title="Cross-lead correlation"), width="stretch")
        st.caption("In real data HRV from <60 s is unreliable; the 'reliable' flag is a heuristic. Zero-phase filtering (sosfiltfilt) doubles the effective order and cannot be used causally in real-time deployment.")

    # ------------------------------------------------------------------ C images
    with tabs[2]:
        st.subheader("ECG-strip and echocardiogram image laboratory")
        ups = st.file_uploader("Upload strip/echo images (PNG/JPG); leave empty for synthetic demos", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if ups: imgs = {u.name: ts.load_image(u) for u in ups}
        else: imgs = {"Synthetic strip A": ts.synth_strip(0), "Synthetic strip B": ts.synth_strip(1), "Synthetic strip A (duplicate)": ts.synth_strip(0), "Synthetic echo 1": ts.synth_echo(1), "Synthetic echo 2": ts.synth_echo(2)}
        rows = [{"Image": k_, **ts.image_stats(v)} for k_, v in imgs.items()]; stdf = pd.DataFrame(rows)
        dup = stdf.groupby("md5").Image.apply(list); dup = dup[dup.apply(len) > 1]
        st.dataframe(fmt(stdf, 2), width="stretch")
        if len(dup): st.warning("Duplicate images detected (identical pixel hash): " + "; ".join(", ".join(v) for v in dup) + ". Remove duplicates before splitting to avoid train/test leakage.")
        sel = st.selectbox("Inspect", list(imgs)); im = imgs[sel]; g = ts.to_gray(im); mag = ts.sobel_mag(g); mag = mag / (mag.max() + 1e-12)
        thr = st.slider("Edge threshold (fraction of max Sobel)", .02, .5, .12, .01)
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(px.imshow(g, color_continuous_scale="gray", title="Greyscale"), width="stretch"); c2.plotly_chart(px.imshow(mag, color_continuous_scale="magma", title="Sobel magnitude"), width="stretch")
        c3.plotly_chart(px.imshow(mag > thr, color_continuous_scale="gray", title=f"Edge map (>{thr:.2f})"), width="stretch")
        st.plotly_chart(px.histogram(g.ravel(), nbins=64, title=f"Intensity histogram (Otsu = {ts.otsu(g)})"), width="stretch")
        st.markdown("**Regional brightness and image quality**")
        a, b = st.columns(2); a.dataframe(pd.Series(ts.region_brightness(im), name="mean intensity").to_frame().style.format("{:.1f}"))
        b.dataframe(pd.Series(ts.echo_quality(im), name="value").to_frame().style.format("{:.3f}"))
        st.caption("Echo metrics are meaningful only for ultrasound sectors. Quadrant→chamber mapping depends on display convention; verify before clinical interpretation. "
                   "For strips, a trace-extraction step (threshold + skeleton) is the usual precursor to a 1D-ViT/CNN; Otsu threshold is shown above.")

    # ------------------------------------------------------------------ D audio
    with tabs[3]:
        st.subheader("Breathing / phonocardiogram audio laboratory")
        up = st.file_uploader("Upload WAV", type=["wav"])
        if up is not None: x, fsa = ts.load_wav(up)
        else:
            c = st.columns(2); br = c[0].slider("Synthetic breathing rate (br/min)", 6, 30, 14); hra = c[1].slider("Synthetic heart rate", 50, 120, 75); x, fsa = ts.synth_audio(hr=hra, breaths_per_min=br)
        st.dataframe(pd.Series(ts.audio_stats(x, fsa), name="value").to_frame().style.format("{:.4g}"))
        bres = ts.breathing_analysis(x, fsa); c = st.columns(3)
        c[0].metric("Cycles detected", bres["n_cycles"]); c[1].metric("Rate (peaks)", "–" if np.isnan(bres["rate_peaks"]) else f"{bres['rate_peaks']:.1f}"); c[2].metric("Rate (ACF)", "–" if np.isnan(bres["rate_acf"]) else f"{bres['rate_acf']:.1f}")
        te = np.arange(len(bres["env"])) / bres["fs_env"]; fig = go.Figure(); fig.add_scatter(x=te, y=bres["env"], name="envelope"); fig.add_scatter(x=te[bres["peaks"]], y=bres["env"][bres["peaks"]], mode="markers", marker=dict(color="red", size=9), name="breaths")
        fig.update_layout(title="Breathing envelope"); st.plotly_chart(fig, width="stretch")
        from scipy import signal as sg
        f_, tt, S = sg.spectrogram(x, fsa, nperseg=1024, noverlap=768); mm = f_ <= min(1500, fsa / 2); st.plotly_chart(px.imshow(10 * np.log10(S[mm] + 1e-12), x=tt, y=f_[mm], origin="lower", aspect="auto", color_continuous_scale="Viridis", title="STFT (dB)"), width="stretch")
        M = ts.mfcc(x, fsa, 20); st.plotly_chart(px.bar(x=np.arange(1, 21), y=M.mean(1), error_y=M.std(1), labels=dict(x="MFCC index", y="mean ± SD"), title="MFCC statistics (20 coefficients, 25 ms / 10 ms)"), width="stretch")
        st.caption("Short windows give unreliable breathing rates (the chapter's 1 cycle in 8 s). Here peak-based and ACF-based estimates are both shown; validated on synthetic signals of known rate. "
                   "A breathing recording cannot stand in for cardiac PCG: murmurs and S1/S2 need stethoscope data (e.g. PhysioNet/CinC 2016, EPHNOGRAM) and segmentation (Springer et al. 2016).")

    if T is None:
        for t_ in tabs[4:10]:
            with t_: st.info("Press **Run TriCARD experiments** in the sidebar.")
        with tabs[10]: _refs()
        return
    pt, tri, df = T["pt"], T["tri"], T["df"]; tab = pt["table"]

    # ------------------------------------------------------------------ E
    with tabs[4]:
        st.subheader(f"PTB-XL-style benchmark — official-style folds (train 1–8, test 10; grouped CV on patients); n_train={pt['n_train']}, n_test={pt['n_test']}")
        fsel = st.selectbox("Feature set", list(tc.FEATSETS), 0)
        t1 = tab[tab["Feature set"] == fsel].sort_values("AUC test", ascending=False)
        st.dataframe(fmt(t1.drop(columns=["Feature set"])), width="stretch")
        fig = go.Figure(go.Scatter(x=t1["AUC test"], y=t1.Model, mode="markers", error_x=dict(type="data", symmetric=False, array=t1["AUC hi"] - t1["AUC test"], arrayminus=t1["AUC test"] - t1["AUC lo"])))
        fig.update_layout(title="Test AUC with 95% bootstrap CI (overlapping intervals ⇒ models not distinguishable)"); st.plotly_chart(fig, width="stretch")
        piv = tab.pivot(index="Model", columns="Feature set", values="AUC test"); piv.columns = [c[:1] for c in piv.columns]
        st.plotly_chart(px.imshow(piv, text_auto=".3f", color_continuous_scale="Viridis", aspect="auto", title="Test AUC: model × feature set (A flags, B meta, C +ECG intervals, D leaky)"), width="stretch")
        c = st.columns(2); fs_, mdl, a_ = pt["best_clean"]
        c[0].plotly_chart(px.bar(pt["perm"].head(12)[::-1], x="AUC drop", y="feature", error_x="sd", orientation="h", title=f"Permutation importance — {mdl}, set {fs_[:1]}"), width="stretch")
        lc = pt["learning"]; fig = go.Figure()
        for col, nm_ in [("train", "Train"), ("val", "Validation")]:
            fig.add_scatter(x=lc.n, y=lc[col], name=nm_, error_y=dict(type="data", array=lc[col + "_sd"]));
        fig.update_layout(title="Learning curve (GradBoost, set C)", xaxis_title="training size", yaxis_title="AUC"); c[1].plotly_chart(fig, width="stretch")
        st.markdown("**ROC / PR (test, feature set above)**"); fig1, fig2 = go.Figure(), go.Figure()
        for m in t1.Model:
            p = pt["probs"][(fsel, m)]; fp, tp_, _ = roc_curve(pt["y_te"], p); fig1.add_scatter(x=fp, y=tp_, name=m); pr, rc, _ = precision_recall_curve(pt["y_te"], p); fig2.add_scatter(x=rc, y=pr, name=m)
        fig1.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="chance"); a, b = st.columns(2); a.plotly_chart(fig1, width="stretch"); b.plotly_chart(fig2, width="stretch")
        st.caption("Sens (abnormal recall) is sensitivity when 'abnormal' is the positive class; Spec (normal recall) is specificity. The chapter reverses these names.")

    # ------------------------------------------------------------------ F
    with tabs[5]:
        st.subheader("Is the benchmark leaking the label?")
        au = pt["audit"]; st.dataframe(fmt(au, 3), width="stretch")
        flagged = au[au.flag != ""]
        if len(flagged): st.error("Flagged: " + ", ".join(flagged.feature) + " — diagnostic flags that never occur in the normal class encode the label; use with care.")
        else: st.success("No feature flagged.")
        g = tab.groupby("Feature set")["AUC test"].agg(["mean", "min", "max"]).reset_index()
        st.plotly_chart(px.bar(g, x="Feature set", y="mean", error_y=g["max"] - g["mean"], error_y_minus=g["mean"] - g["min"], title="Mean test AUC (bars span best/worst model) by feature set", range_y=[.4, 1]), width="stretch")
        sc = pt["split_compare"]; st.dataframe(sc.groupby("split").agg(AUC=("AUC", "mean"), SD=("AUC", "std"), patients_in_both=("patients in both", "mean")).style.format("{:.3f}"))
        st.caption("In this synthetic table patient-specific offsets are modest, so the record-vs-patient gap is small; in real PTB-XL (multiple records per patient, up to dozens) it can be larger. "
                   "Always split by patient — this app's grouped CV and strat_fold test do so.")
        st.subheader("Age coding")
        a = df.age; st.write(f"Raw: min {a.min():.0f}, max {a.max():.0f}, mean {a.mean():.1f}, SD {a.std():.1f} → after treating 300 as missing: mean {tc.clean_age(a).mean():.1f}, SD {tc.clean_age(a).std():.1f}.")
        st.plotly_chart(px.histogram(tc.clean_age(a), nbins=40, title="Age after cleaning"), width="stretch")

    # ------------------------------------------------------------------ G
    with tabs[6]:
        st.subheader("PCA and K-means — do clusters recover Normal/Abnormal?")
        for key, title in [("unsup_A", "Chapter setup (A: metadata + diagnostic flags)"), ("unsup_C", "With measured ECG intervals (C)")]:
            u = pt[key]; st.markdown(f"**{title}**"); a, b, c = st.columns(3)
            a.plotly_chart(px.bar(x=np.arange(1, len(u["evr"]) + 1), y=u["evr"] * 100, labels=dict(x="PC", y="% variance"), title="Scree"), width="stretch")
            ut = u["table"]; fig = make_subplots(specs=[[{"secondary_y": True}]]); fig.add_scatter(x=ut.k, y=ut.Silhouette, name="Silhouette", secondary_y=False); fig.add_scatter(x=ut.k, y=ut["ARI vs label"], name="ARI vs label", secondary_y=True)
            b.plotly_chart(fig, width="stretch"); c.plotly_chart(px.scatter(x=u["pc"][:, 0], y=u["pc"][:, 1], color=np.where(u["y"] == 1, "Abnormal", "Normal"), labels=dict(x="PC1", y="PC2"), title="PC1 vs PC2"), width="stretch")
            st.dataframe(fmt(ut), width="stretch")
        st.caption("Silhouette measures cluster separation, not diagnostic relevance. With ten mostly-binary features many rows are identical, which inflates apparent cluster structure; "
                   "ARI/NMI against the label show whether the best-silhouette partition (often k=2) actually is Normal/Abnormal. The chapter's 'recovers the supervised class structure' requires this check.")

    # ------------------------------------------------------------------ H
    with tabs[7]:
        st.subheader("Tri-modal (ECG + PCG + Echo) fusion — five diagnoses")
        st.dataframe(fmt(tri["perf"]), width="stretch"); b = tri["boot"]
        st.write(f"Tri-modal vs best unimodal ({b['best_unimodal']}): ΔAUC = **{b['delta']:.3f}** (95% bootstrap CI {b['ci'][0]:.3f}–{b['ci'][1]:.3f}; two-sided bootstrap p ≈ {max(b['p'], 1 / 300):.3f}).")
        pf = tri["perf"]; fig = px.bar(pf, x="Modalities", y="Macro AUC (OvR)", color="k", title="Fusion benefit by modality subset", range_y=[.5, 1]); st.plotly_chart(fig, width="stretch")
        cls_cols = [c for c in pf.columns if c.startswith("AUC ") and c != "AUC (OvR)"]
        st.plotly_chart(px.imshow(pf.set_index("Modalities")[cls_cols], text_auto=".2f", color_continuous_scale="Viridis", aspect="auto", title="Per-class AUC by modality subset"), width="stretch")
        st.subheader("Modality attribution: data-derived exact Shapley vs chapter Table 17")
        al = tri["alpha"]; c = st.columns(2)
        fig = go.Figure()
        for m in al.columns: fig.add_bar(x=al.index, y=al[m], name=m)
        fig.update_layout(barmode="stack", title="Computed α_m (normalised positive Shapley values)"); c[0].plotly_chart(fig, width="stretch")
        c[1].plotly_chart(px.imshow(al, text_auto=".2f", color_continuous_scale="Blues", aspect="auto", title="Computed attribution heat-map"), width="stretch")
        st.dataframe(pd.concat([al.round(2).add_prefix("computed "), tc.CHAPTER_T17.round(2).add_prefix("chapter ")], axis=1), width="stretch"); st.dataframe(tri["alpha_cmp"].style.format({"mean |Δα| vs chapter Table 17": "{:.3f}"}))
        st.caption("Shapley value of a modality = average marginal gain in one-vs-rest AUC over all coalitions (exact; 3 players). The synthetic class shifts were set from textbook physiology, so agreement with Table 17 shows the pipeline recovers the planted structure — "
                   "it does not validate Table 17 on patients. Use real tri-modal data for that.")

    # ------------------------------------------------------------------ I
    with tabs[8]:
        st.subheader("Measured modality-dropout robustness (test-time masking)")
        dr = tri["drop"]; fig = px.line(dr, x="tau", y="AUC", color="model", error_y="sd", markers=True, title="Macro AUC vs test-time modality-dropout probability τ"); st.plotly_chart(fig, width="stretch")
        pv = dr.pivot(index="tau", columns="model", values="AUC"); st.dataframe(fmt(pv), width="stretch")
        gap = pv["Fusion, curriculum modality-dropout"] - pv["Fusion, standard training"]
        st.write(f"Curriculum-trained minus standard fusion: τ=0 → {gap.iloc[0]:+.3f}; τ=0.5 → {gap.iloc[-1]:+.3f}. Fusion minus best unimodal at τ=0.5: {pv.iloc[-1][['ECG-only', 'PCG-only', 'Echo-only']].pipe(lambda s: pv.iloc[-1]['Fusion, curriculum modality-dropout'] - s.max()):+.3f}.")
        e = np.arange(50); tau_curve = [tc.mc.tau_schedule(i, 50, .5) for i in e]
        st.plotly_chart(px.line(x=e, y=tau_curve, labels=dict(x="epoch / copy", y="τ"), title="Curriculum schedule (linear warm-up then cosine annealing)"), width="stretch")
        st.caption("Contrary to the chapter's Figure 32 narrative, measured ordering of unimodal models depends on the data: here Echo > ECG > PCG, and every model degrades with τ. "
                   "Fusion remains best because masking removes at most the less informative blocks on average; with τ near 1 it collapses toward the unimodal level.")

    # ------------------------------------------------------------------ J
    with tabs[9]:
        st.subheader("Benchmark table (labelled for comparability)")
        bt = tc.benchmark_table(pt, tri); st.dataframe(fmt(bt), width="stretch")
        st.plotly_chart(px.bar(bt, x="AUC", y="System", color="Like-for-like?", orientation="h", height=560, title="AUC — only 'computed here' rows were produced by this app"), width="stretch")
        ca = tab[tab["Feature set"].str.startswith("A")]["AUC test"]; cc = tab[tab["Feature set"].str.startswith("C")]["AUC test"]; cb = tab[tab["Feature set"].str.startswith("B")]["AUC test"]
        st.markdown(f"""
### Interpretation of this run
* **Metadata ≠ diagnosis.** With age/sex/anthropometrics alone the AUC is {cb.min():.2f}–{cb.max():.2f}; adding LVH/CLBB flags (the chapter's setup) gives {ca.min():.2f}–{ca.max():.2f}; adding *measured* ECG intervals gives {cc.min():.2f}–{cc.max():.2f}. The 0.13–0.18 'gap to deep learning' in the chapter is mostly an information gap (no waveform information) rather than a model-capacity gap.
* **Classifier ranking is statistically fragile.** Top models differ by ≤ 0.01 AUC while bootstrap CIs span about ±0.03 (Tab E). Declaring one 'best' needs DeLong/paired tests (see the M-AIM module).
* **Fusion helps in this simulation** (ΔAUC {tri['boot']['delta']:.3f} vs best unimodal, CI {tri['boot']['ci'][0]:.3f}–{tri['boot']['ci'][1]:.3f}), but the classes were built so each modality carries distinct information; real gains must be re-measured on synchronised patient data.
* **Robustness is measurable, not assumed** (Tab I); curriculum training changes the trade-off between clean and degraded performance.
* **Shapley attribution is computable from data** and should replace the literature-set Table 17 before the manuscript claims TC.5.

### Priority fixes before submission
1. Replace 'literature-informed' with computed attribution, or relabel Table 17 as 'prior'. 2. Split by patient; report folds. 3. Report leaky-flag ablation. 4. Correct sensitivity/specificity naming. 5. Remove 'projected AUC ≥ 0.95' as a result.
6. Compare with published systems only on matched dataset/task (e.g. PTB-XL benchmarking: Strodthoff et al. 2021). 7. Add CIs and paired tests for every ranking.
""")

    with tabs[10]: _refs()


def _refs():
    st.markdown("""
### References for this module (verify DOIs before submission)
Items marked ✓ appear in the chapter's reference list with a DOI; items marked ◇ were added here from memory and are *not* verified — check them.

**Data & ECG deep learning**
- ✓ Wagner P, et al. PTB-XL, a large publicly available electrocardiography dataset. Sci Data 2020;7:154. doi:10.1038/s41597-020-0495-6
- ◇ Strodthoff N, et al. Deep learning for ECG analysis: benchmarks and insights from PTB-XL. IEEE J Biomed Health Inform 2021;25:1519–1528. doi:10.1109/JBHI.2020.3022989
- ◇ Ribeiro AH, et al. Automatic diagnosis of the 12-lead ECG using a deep neural network. Nat Commun 2020;11:1760. doi:10.1038/s41467-020-15432-4
- ✓ Kligfield P, et al. Recommendations for the standardization and interpretation of the ECG, Part I. J Am Coll Cardiol 2007;49:1109–1127. doi:10.1016/j.jacc.2007.01.024
- ✓ Hancock EW, et al. AHA/ACCF/HRS recommendations … Part V. J Am Coll Cardiol 2009;53:992–1002. doi:10.1016/j.jacc.2008.12.015

**HRV & signal processing**
- ✓ Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. Front Public Health 2017;5:258. doi:10.3389/fpubh.2017.00258
- ◇ Task Force of the ESC and NASPE. Heart rate variability: standards of measurement. Circulation 1996;93:1043–1065. doi:10.1161/01.CIR.93.5.1043
- ◇ Pan J, Tompkins WJ. A real-time QRS detection algorithm. IEEE Trans Biomed Eng 1985;32:230–236. doi:10.1109/TBME.1985.325532

**Heart sounds, echo, fusion**
- ✓ Springer DB, Tarassenko L, Clifford GD. Logistic regression-HSMM-based heart sound segmentation. IEEE Trans Biomed Eng 2016;63:822–832. doi:10.1109/TBME.2015.2475278
- ✓ Zhao Q, et al. Deep learning in heart sound analysis. Health Data Sci 2024;4:0182. doi:10.34133/hds.0182
- ✓ Ouyang D, et al. Video-based AI for beat-to-beat assessment of cardiac function. Nature 2020;580:252–256. doi:10.1038/s41586-020-2145-8
- ✓ Deng M, et al. Use of multi-modal data and machine learning to improve cardiovascular disease care. Front Cardiovasc Med 2022;9:840262. doi:10.3389/fcvm.2022.840262
- ✓ Wu R, et al. Deep multimodal learning with missing modality: a survey. arXiv:2409.07825 (2024).
- ✓ Lin J, et al. Progressive attention-based cross-modal fusion network for CVD detection. PeerJ Comput Sci 2025. doi:10.7717/peerj-cs.3038 · ✓ Xiong N, et al. HCTG-Net. Bioengineering 2025;12:1268. doi:10.3390/bioengineering12111268
- ✓ Salih AM, et al. Evaluation approaches for explainable AI in cardiology. Artif Intell Rev 2024;57:240. doi:10.1007/s10462-024-10852-w

**Methods**
- ◇ Shapley LS. A value for n-person games. In: Contributions to the Theory of Games II, Princeton Univ Press, 1953.
- ✓ Chen T, Guestrin C. XGBoost. KDD 2016:785–794. doi:10.1145/2939672.2939785 · ✓ Rousseeuw PJ. Silhouettes. J Comput Appl Math 1987;20:53–65. doi:10.1016/0377-0427(87)90125-7
- ◇ Collins GS, et al. TRIPOD+AI. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378
- ◇ Hubert L, Arabie P. Comparing partitions. J Classification 1985;2:193–218. doi:10.1007/BF01908075
""")
