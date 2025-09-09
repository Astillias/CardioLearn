
import argparse, os, json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from cardiolearn.features import featurize_dataframe
from cardiolearn.fairness import group_mae
from cardiolearn.calibration import split_conformal_intervals, apply_interval, coverage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/ppg_synth.csv')
    ap.add_argument('--artifacts', type=str, default='artifacts')
    ap.add_argument('--alpha', type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X, y, meta = featurize_dataframe(df)
    Xtr, Xte, ytr, yte, mtr, mte = train_test_split(X, y, meta, test_size=0.2, random_state=1)

    gbr = joblib.load(os.path.join(args.artifacts, 'gbr.joblib'))
    yhat_tr = gbr.predict(Xtr)
    yhat_te = gbr.predict(Xte)

    q = split_conformal_intervals(ytr, yhat_tr, alpha=args.alpha)
    lo, hi = apply_interval(yhat_te, q)
    cov = coverage(yte, lo, hi)
    mae = float(np.mean(np.abs(yte - yhat_te)))

    mae_sensor = group_mae(yte, yhat_te, mte['sensor_group'])
    mae_skin = group_mae(yte, yhat_te, mte['skin_tone_proxy'])

    report = {
        'mae_overall': mae,
        'conformal_q': float(q),
        'cov_at_alpha': float(1-args.alpha),
        'empirical_coverage': float(cov),
        'group_mae_sensor': mae_sensor,
        'group_mae_skin': mae_skin
    }
    print(json.dumps(report, indent=2))
    os.makedirs(args.artifacts, exist_ok=True)
    with open(os.path.join(args.artifacts, 'eval_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()
