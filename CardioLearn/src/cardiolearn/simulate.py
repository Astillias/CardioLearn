"""
Synthetic PPG + SBP generator.

We simulate PPG beats as damped sinusoids with added harmonics, HRV, motion/thermal noise,
and domain shifts. SBP is tied to morphology (e.g., shorter rise time -> higher SBP),
plus demographic covariates.
"""
import numpy as np
import pandas as pd
import argparse

def _ppg_waveform(t, hr_hz, alpha=0.6, beta=0.3, gamma=0.1):
    # base + first/second harmonic with damping to shape systolic peak and diastolic wave
    base = np.exp(-alpha * (t % (1/hr_hz))) * np.sin(2*np.pi*hr_hz*t)
    h1 = beta * np.exp(-0.8*alpha * (t % (1/hr_hz))) * np.sin(4*np.pi*hr_hz*t + 0.3)
    h2 = gamma * np.exp(-0.5*alpha * (t % (1/hr_hz))) * np.sin(6*np.pi*hr_hz*t + 0.6)
    return base + h1 + h2

def simulate_ppg_row(n_samples=256, fs=128, seed=None, sensor_group=None, skin_tone_proxy=None):
    rng = np.random.default_rng(seed)
    # Heart rate in Hz (60-100 bpm typical rest to light activity)
    hr = rng.uniform(60, 100)
    hr_hz = hr/60.0
    # Small HR variability
    fs = float(fs)
    t = np.arange(n_samples)/fs
    # Morphology params affect systolic peak sharpness -> correlate with SBP
    alpha = rng.uniform(0.4, 0.9)
    beta = rng.uniform(0.15, 0.45)
    gamma = rng.uniform(0.05, 0.2)

    ppg = _ppg_waveform(t, hr_hz, alpha, beta, gamma)

    # Sensor/domain shift: group A vs B have slightly different baseline & noise
    if sensor_group is None:
        sensor_group = rng.choice(['A', 'B'])
    if skin_tone_proxy is None:
        skin_tone_proxy = rng.choice(['light','medium','dark'])
    if sensor_group == 'A':
        ppg += rng.normal(0, 0.01, size=ppg.shape)
        baseline = 0.1
    else:
        ppg += rng.normal(0, 0.02, size=ppg.shape)
        baseline = -0.05
    ppg += baseline

    # Motion artifact bursts
    if rng.random() < 0.3:
        idx = rng.integers(0, n_samples-10)
        ppg[idx:idx+10] += rng.normal(0, 0.2, size=10)

    # Demographics (synthetic)
    age = rng.integers(18, 80)
    bmi = rng.uniform(18, 35)
    sex = rng.choice(['M','F'])

    # SBP model (synthetic ground-truth): sharper systolic (lower alpha) and higher HR -> higher SBP
    sbp = 90 + (100 - 60)*(hr-60)/40 + (0.9-alpha)*35 + 0.8*(bmi-22) + 0.15*(age-45)
    # group shift + skin tone proxy mild effect (simulated bias)
    sbp += {'A':0, 'B':5}[sensor_group]
    sbp += {'light':0, 'medium':2, 'dark':4}[skin_tone_proxy]
    sbp += rng.normal(0, 5)

    row = {
        'ppg': ppg.astype(np.float32),
        'fs': fs,
        'age': age,
        'bmi': float(bmi),
        'sex': sex,
        'sensor_group': sensor_group,
        'skin_tone_proxy': skin_tone_proxy,
        'sbp': float(sbp)
    }
    return row

def generate_dataset(n=1000, fs=128, seed=0):
    rng = np.random.default_rng(seed)
    rows = [simulate_ppg_row(seed=int(rng.integers(0, 1e9))) for _ in range(n)]
    # Store waveform as JSON array string for portability
    df = pd.DataFrame(rows)
    df['ppg_json'] = df['ppg'].apply(lambda x: json_dumps(x.tolist()))
    df = df.drop(columns=['ppg'])
    return df

def json_dumps(x):
    import json
    return json.dumps(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--fs', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='data/ppg_synth.csv')
    args = ap.parse_args()
    df = generate_dataset(n=args.n, fs=args.fs, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == '__main__':
    main()
