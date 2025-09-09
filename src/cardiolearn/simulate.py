
import numpy as np, pandas as pd, json, argparse

def _ppg_waveform(t, hr_hz, alpha=0.6, beta=0.3, gamma=0.1):
    base = np.exp(-alpha * (t % (1/hr_hz))) * np.sin(2*np.pi*hr_hz*t)
    h1 = beta * np.exp(-0.8*alpha * (t % (1/hr_hz))) * np.sin(4*np.pi*hr_hz*t + 0.3)
    h2 = gamma * np.exp(-0.5*alpha * (t % (1/hr_hz))) * np.sin(6*np.pi*hr_hz*t + 0.6)
    return base + h1 + h2

def simulate_ppg_row(n_samples=256, fs=128, rng=None):
    if rng is None: rng = np.random.default_rng()
    hr = rng.uniform(60, 100); hr_hz = hr/60.0
    t = np.arange(n_samples)/fs
    alpha = rng.uniform(0.4, 0.9)
    beta = rng.uniform(0.15, 0.45)
    gamma = rng.uniform(0.05, 0.2)
    ppg = _ppg_waveform(t, hr_hz, alpha, beta, gamma)

    sensor_group = rng.choice(['A','B'])
    if sensor_group == 'A':
        ppg += rng.normal(0, 0.01, size=ppg.shape) + 0.1
    else:
        ppg += rng.normal(0, 0.02, size=ppg.shape) - 0.05

    if rng.random() < 0.3:
        idx = rng.integers(0, len(ppg)-10)
        ppg[idx:idx+10] += rng.normal(0, 0.2, size=10)

    age = int(rng.integers(18, 80))
    bmi = float(rng.uniform(18, 35))
    sex = rng.choice(['M','F'])
    skin = rng.choice(['light','medium','dark'])

    sbp = 90 + (100 - 60)*(hr-60)/40 + (0.9-alpha)*35 + 0.8*(bmi-22) + 0.15*(age-45)
    sbp += {'A':0,'B':5}[sensor_group] + {'light':0,'medium':2,'dark':4}[skin]
    sbp += rng.normal(0,5)

    return {
        'ppg_json': json.dumps(ppg.astype(float).tolist()),
        'fs': 128, 'age': age, 'bmi': bmi, 'sex': sex,
        'sensor_group': sensor_group, 'skin_tone_proxy': skin, 'sbp': float(sbp)
    }

def generate_dataset(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = [simulate_ppg_row(rng=rng) for _ in range(n)]
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='data/ppg_synth.csv')
    args = ap.parse_args()
    df = generate_dataset(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
