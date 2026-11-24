"""
TriCARD signal / audio / image toolkit (pure NumPy/SciPy/Pillow; no librosa or OpenCV needed).
Implements the chapter's processing chain: Butterworth band-pass (TC.2), Pan-Tompkins-style R-peak detection (TC.3),
HRV + Poincare, PSD/ACF/STFT, breathing/PCG audio features incl. MFCC (TC.17), and echo/strip image statistics.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats
from scipy.fft import dct


# --------------------------------------------------------------------------- ECG
def synth_ecg(fs=500, dur=5.0, n_leads=5, hr=80, hrv=0.03, irregular=False, noise=0.03, seed=0):
    r = np.random.default_rng(seed); n = int(fs * dur); t = np.arange(n) / fs
    bt = [r.uniform(0.1, 0.4)]
    while bt[-1] < dur + 1:
        rr = 60 / hr * (1 + (r.normal(0, 0.25) if irregular else r.normal(0, hrv)))
        bt.append(bt[-1] + max(rr, 0.3))
    comps = [(-0.2, 0.025, 0.0 if irregular else 0.12), (-0.04, 0.010, -0.10), (0, 0.012, 1.0), (0.035, 0.012, -0.22), (0.28, 0.05, 0.30)]
    gains = np.resize(np.array([1.0, 0.85, 0.6, 1.2, 0.5]), n_leads)
    ecg = np.zeros((n_leads, n))
    for tb in bt:
        for off, w, a in comps:
            ecg += gains[:, None] * a * np.exp(-((t - tb - off) ** 2) / (2 * w ** 2))
    ecg += 0.15 * np.sin(2 * np.pi * 0.25 * t + r.uniform(0, 6, (n_leads, 1))) + r.normal(0, noise, ecg.shape)
    return ecg


def bandpass(x, fs, lo=0.5, hi=40.0, order=4):
    """Zero-phase Butterworth band-pass (sosfiltfilt => effective order 2x, no phase distortion). Poles are inside the unit circle => BIBO stable."""
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def descriptive_stats(ecg):
    rows = []
    for i, x in enumerate(ecg):
        rows.append({"Lead": f"Lead {i + 1}", "Mean": x.mean(), "Std": x.std(ddof=1), "Min": x.min(), "Max": x.max(), "P-P": np.ptp(x),
                     "Skewness": stats.skew(x), "Excess kurtosis": stats.kurtosis(x), "RMS": np.sqrt(np.mean(x ** 2))})
    return pd.DataFrame(rows)


def psd(x, fs):
    f, p = signal.welch(x, fs, nperseg=min(len(x), int(4 * fs)))
    m = (f >= 0.5) & (f <= 5)
    dom = f[m][np.argmax(p[m])] if m.any() else np.nan
    return f, p, dom


def detect_r_peaks(x, fs):
    """Pan-Tompkins-style: 5-15 Hz band-pass -> derivative -> square -> 150 ms moving integration -> adaptive threshold -> refractory 250 ms."""
    sos = signal.butter(2, [5, 15], btype="band", fs=fs, output="sos"); f = signal.sosfiltfilt(sos, x)
    mwi = np.convolve(np.gradient(f) ** 2, np.ones(int(0.15 * fs)) / int(0.15 * fs), mode="same")
    pk, _ = signal.find_peaks(mwi, height=0.25 * np.percentile(mwi, 99.5), distance=int(0.25 * fs))
    hw = int(0.06 * fs); out = []
    for p in pk:
        lo, hi = max(0, p - hw), min(len(x), p + hw); out.append(lo + int(np.argmax(np.abs(f[lo:hi]))))
    return np.unique(out)


def clean_rr(rr_ms, tol=0.30):
    """Reject RR intervals deviating > tol from the running median (ectopic / mis-detection screen)."""
    if len(rr_ms) < 3: return rr_ms
    med = np.median(rr_ms); return rr_ms[np.abs(rr_ms - med) <= tol * med]


def hrv_metrics(rpeaks, fs, dur_s, clean=False):
    rr = np.diff(rpeaks) / fs * 1000
    if clean: rr = clean_rr(rr)
    if len(rr) < 3:
        return dict(N_beats=len(rpeaks), HR_mean=np.nan, HR_std=np.nan, RR_mean=np.nan, RMSSD=np.nan, SDNN=np.nan, pNN50=np.nan, SD1=np.nan, SD2=np.nan, reliable=False)
    d = np.diff(rr); sdnn = rr.std(ddof=1); rmssd = np.sqrt(np.mean(d ** 2)); sd1 = rmssd / np.sqrt(2)
    return dict(N_beats=len(rpeaks), HR_mean=60000 / rr.mean(), HR_std=np.std(60000 / rr, ddof=1), RR_mean=rr.mean(), RMSSD=rmssd, SDNN=sdnn,
                pNN50=100 * np.mean(np.abs(d) > 50), SD1=sd1, SD2=np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0)),
                reliable=bool(dur_s >= 60))  # heuristic: >=60 s for RMSSD-type short-term HRV (Shaffer & Ginsberg 2017; Task Force 1996 recommends 5 min)


def acf(x, nlags=300):
    x = np.asarray(x, float) - np.mean(x); f = np.fft.rfft(x, 2 * len(x)); a = np.fft.irfft(f * np.conj(f))[: nlags + 1]
    return a / a[0]


def stft_db(x, fs, nperseg=128, noverlap=96):
    f, t, S = signal.spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, 10 * np.log10(S + 1e-12)


# --------------------------------------------------------------------------- audio / PCG
def synth_audio(fs=4000, dur=40.0, hr=75, breaths_per_min=14, seed=0):
    """Heart-sound-like S1/S2 bursts + modulated breathing noise (demo only)."""
    r = np.random.default_rng(seed); n = int(fs * dur); t = np.arange(n) / fs; x = np.zeros(n)
    def burst(t0, f0, amp, w=0.03):
        return amp * np.exp(-((t - t0) ** 2) / (2 * w ** 2)) * np.sin(2 * np.pi * f0 * (t - t0))
    for tb in np.arange(0.3, dur, 60 / hr):
        x += burst(tb, 60, 1.0) + burst(tb + 0.32, 90, 0.7, 0.025)
    env = 0.5 * (1 + np.sin(2 * np.pi * breaths_per_min / 60 * t))
    x += 0.25 * env * signal.lfilter(*signal.butter(2, [150, 800], btype="band", fs=fs), r.normal(0, 1, n))
    return x / np.abs(x).max() * 0.5, fs


def load_wav(file):
    from scipy.io import wavfile
    fs, x = wavfile.read(file); x = x.astype(float)
    if x.ndim > 1: x = x.mean(1)
    if x.dtype.kind in "iu" or np.abs(x).max() > 1.5: x = x / (np.abs(x).max() + 1e-12)
    return x, fs


def breathing_analysis(x, fs):
    q = max(int(fs // 50), 1); m = len(x) // q * q
    env = np.abs(signal.hilbert(x[:m])).reshape(-1, q).mean(1); fe = fs / q
    env = signal.sosfiltfilt(signal.butter(2, 0.6, fs=fe, output="sos"), env)
    pk, _ = signal.find_peaks(env, distance=int(2.0 * fe), prominence=0.5 * env.std())
    rate_pk = (len(pk) - 1) / ((pk[-1] - pk[0]) / fe) * 60 if len(pk) > 2 else np.nan
    a = acf(env, int(15 * fe)); lo, hi = int(1.5 * fe), min(int(15 * fe), len(a) - 1)
    rate_acf = 60 / ((lo + int(np.argmax(a[lo:hi]))) / fe) if hi > lo else np.nan
    return dict(duration_s=len(x) / fs, n_cycles=len(pk), rate_peaks=rate_pk, rate_acf=rate_acf, env=env, fs_env=fe, peaks=pk)


def _mel(f): return 2595 * np.log10(1 + f / 700)
def _imel(m): return 700 * (10 ** (m / 2595) - 1)


def mfcc(x, fs, n_mfcc=20, n_mels=40, frame=0.025, hop=0.010):
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]); fl, hl = int(frame * fs), int(hop * fs); nfft = 1 << (fl - 1).bit_length()
    nfr = 1 + (len(x) - fl) // hl; idx = np.arange(fl)[None, :] + hl * np.arange(nfr)[:, None]
    P = np.abs(np.fft.rfft(x[idx] * np.hamming(fl), nfft)) ** 2 / nfft
    bins = np.floor((nfft + 1) * _imel(np.linspace(_mel(0), _mel(fs / 2), n_mels + 2)) / fs).astype(int)
    fb = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r_ = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1); fb[m - 1, c:r_] = (r_ - np.arange(c, r_)) / max(r_ - c, 1)
    return dct(np.log(P @ fb.T + 1e-10), type=2, axis=1, norm="ortho")[:, :n_mfcc].T


def audio_stats(x, fs):
    f, p = signal.welch(x, fs, nperseg=min(len(x), 8192)); tot = p.sum()
    return {"Sampling rate (Hz)": fs, "Duration (s)": len(x) / fs, "RMS": float(np.sqrt(np.mean(x ** 2))), "Peak": float(np.abs(x).max()),
            "Crest factor": float(np.abs(x).max() / np.sqrt(np.mean(x ** 2))), "Dominant frequency (Hz)": float(f[np.argmax(p)]),
            "Spectral centroid (Hz)": float((f * p).sum() / tot), "Energy < 200 Hz (%)": 100 * p[f < 200].sum() / tot,
            "Energy 200-800 Hz (%)": 100 * p[(f >= 200) & (f < 800)].sum() / tot,
            "Zero-crossing rate": float(np.mean(np.abs(np.diff(np.sign(x))) > 0))}


# --------------------------------------------------------------------------- images
def to_gray(img):
    a = np.asarray(img, float)
    return a if a.ndim == 2 else a[..., :3] @ np.array([0.299, 0.587, 0.114])


def load_image(file):
    from PIL import Image
    return np.asarray(Image.open(file).convert("RGB"))


def otsu(g):
    h, _ = np.histogram(g, bins=256, range=(0, 256)); p = h / max(h.sum(), 1); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256))
    return int(np.argmax((mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)))


def sobel_mag(g):
    return np.hypot(ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0))


def image_stats(img, edge_frac=0.12):
    g = to_gray(img); m = sobel_mag(g); mn = m / (m.max() + 1e-12)
    return {"H": g.shape[0], "W": g.shape[1], "Mean": g.mean(), "Std": g.std(), "Min": g.min(), "Max": g.max(), "Dark px %": 100 * np.mean(g < 80),
            "Edge density %": 100 * np.mean(mn > edge_frac), "Gradient mean (norm.)": mn.mean(), "md5": hashlib.md5(g.astype(np.uint8).tobytes()).hexdigest()[:10]}


def region_brightness(img):
    """Five fixed regions (quadrants + centre). Mapping of quadrants to LV/RV/LA/RA depends on display orientation convention - verify per vendor."""
    g = to_gray(img); H, W = g.shape; h, w = H // 2, W // 2
    return {"Upper-left": g[:h, :w].mean(), "Upper-right": g[:h, w:].mean(), "Lower-left": g[h:, :w].mean(), "Lower-right": g[h:, w:].mean(),
            "Centre": g[H // 4: 3 * H // 4, W // 4: 3 * W // 4].mean()}


def echo_quality(img):
    """Well-posed quality metrics. CNR: Otsu split of sector pixels (blood vs tissue); tissue SNR = mean/std in tissue; sharpness = Laplacian variance;
    Michelson contrast on 1st/99th percentiles (the chapter's 'contrast = 1.000' for every view is a saturated, uninformative metric)."""
    g = to_gray(img); roi = g > 8
    if roi.sum() < 100: return {"CNR": np.nan, "Tissue SNR (dB)": np.nan, "Sharpness (Lap. var.)": np.nan, "Michelson contrast": np.nan}
    t = otsu(g[roi]); hi, lo = g[roi & (g > t)], g[roi & (g <= t)]
    cnr = abs(hi.mean() - lo.mean()) / np.sqrt((hi.var() + lo.var()) / 2 + 1e-12)
    p1, p99 = np.percentile(g[roi], [1, 99])
    return {"CNR": cnr, "Tissue SNR (dB)": 20 * np.log10(hi.mean() / (hi.std() + 1e-12)), "Sharpness (Lap. var.)": ndimage.laplace(g).var(),
            "Michelson contrast": (p99 - p1) / (p99 + p1 + 1e-12)}


def synth_echo(seed=0, size=(360, 480)):
    r = np.random.default_rng(seed); H, W = size; yy, xx = np.mgrid[0:H, 0:W]
    ax, ay = W / 2, 15; ang = np.degrees(np.arctan2(xx - ax, yy - ay + 1e-9)); rad = np.hypot(xx - ax, yy - ay)
    sector = (np.abs(ang) < 42) & (rad < H - 15)
    img = r.rayleigh(30, size) * 1.0
    for cx, cy, a, b in [(0.60, 0.55, 70, 105), (0.36, 0.55, 55, 90), (0.62, 0.80, 55, 45), (0.34, 0.80, 50, 42)]:
        img *= np.where(((xx - cx * W) / a) ** 2 + ((yy - cy * H) / b) ** 2 < 1, 0.12, 1.0)
    img = ndimage.gaussian_filter(img, 1.2) * (1.4 + 0.6 * r.random()); img[~sector] = 0
    return np.clip(img * 2.2, 0, 255).astype(np.uint8)


def synth_strip(seed=0, size=(420, 768)):
    from PIL import Image, ImageDraw
    H, W = size; im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
    for x in range(0, W, 8): d.line([(x, 0), (x, H)], fill=(255, 215, 215), width=1)
    for y in range(0, H, 8): d.line([(0, y), (W, y)], fill=(255, 215, 215), width=1)
    e = synth_ecg(250, W / 250 * 1.0, 1, hr=70 + 10 * seed, seed=seed)[0]; e = (e - e.min()) / np.ptp(e)
    xs = np.linspace(0, W - 1, len(e)); ys = H * 0.75 - e * H * 0.45
    d.line(list(zip(xs, ys)), fill=(20, 20, 20), width=2)
    return np.asarray(im)
