"""
PPG feature extraction (time-domain, derivatives, frequency proxies)
Works on per-row JSON PPG arrays for simplicity.
"""
import numpy as np, pandas as pd, json
from scipy.signal import butter, filtfilt

def _butter_bandpass(low, high, fs, order=3):
    ny = 0.5*fs
    b,a = butter(order, [low/ny, high/ny], btype='band')
    return b,a

def clean_ppg(ppg, fs):
    b,a = _butter_bandpass(0.5, 8.0, fs)
    return filtfilt(b,a,ppg)

def basic_stats(x):
    return {
        'ppg_mean': float(np.mean(x)),
        'ppg_std': float(np.std(x)),
        'ppg_skew': float(((x - x.mean())**3).mean() / (x.std()+1e-6)**3),
        'ppg_kurt': float(((x - x.mean())**4).mean() / (x.std()+1e-6)**4),
        'ppg_p2p': float(x.max() - x.min()),
    }

def derivative_features(x, fs):
    dx = np.gradient(x)*fs
    ddx = np.gradient(dx)*fs
    return {
        'dx_std': float(np.std(dx)),
        'ddx_std': float(np.std(ddx)),
        'slope_pos_frac': float(np.mean(dx>0)),
        'slope_neg_frac': float(np.mean(dx<0)),
    }

def freq_features(x, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1/fs)
    power = (X.real**2 + X.imag**2)
    # band powers
    def band(lo, hi):
        idx = (f>=lo) & (f<hi)
        return float(power[idx].sum())
    return {
        'bp_0_1': band(0.0, 1.0),
        'bp_1_3': band(1.0, 3.0),
        'bp_3_8': band(3.0, 8.0),
        'dom_freq': float(f[np.argmax(power)]),
    }

def features_from_row(row):
    ppg = np.array(json.loads(row['ppg_json']), dtype=float)
    fs = float(row['fs'])
    x = clean_ppg(ppg, fs)
    feat = {}
    feat.update(basic_stats(x))
    feat.update(derivative_features(x, fs))
    feat.update(freq_features(x, fs))
    # concatenate metadata/demographics
    feat['age'] = float(row['age'])
    feat['bmi'] = float(row['bmi'])
    feat['sex_M'] = 1.0 if row['sex']=='M' else 0.0
    feat['sensor_group_A'] = 1.0 if row['sensor_group']=='A' else 0.0
    feat['skin_medium'] = 1.0 if row['skin_tone_proxy']=='medium' else 0.0
    feat['skin_dark'] = 1.0 if row['skin_tone_proxy']=='dark' else 0.0
    return feat

def featurize_dataframe(df):
    feats = [features_from_row(r) for _,r in df.iterrows()]
    X = pd.DataFrame(feats)
    y = df['sbp'].astype(float).values
    meta = df[['sensor_group','skin_tone_proxy']].copy()
    return X, y, meta
