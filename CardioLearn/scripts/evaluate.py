import argparse, os, json, joblib
import pandas as pd, numpy as np
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
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(X, y, meta, test_size=0.2, random_state=1)

    gbr = joblib.load(os.path.join(args.artifacts, 'gbr.joblib'))
    yhat_tr = gbr.predict(X_tr)
    yhat_te = gbr.predict(X_te)

    q = split_conformal_intervals(y_tr, yhat_tr, alpha=args.alpha)
    lo, hi = apply_interval(yhat_te, q)
    cov = coverage(y_te, lo, hi)
    mae = float(np.mean(np.abs(y_te - yhat_te)))

    # group slices
    mae_sensor = group_mae(y_te, yhat_te, m_te['sensor_group'])
    mae_skin = group_mae(y_te, yhat_te, m_te['skin_tone_proxy'])

    report = {
        'mae_overall': mae,
        'conformal_q': float(q),
        'cov_at_alpha': float(1-args.alpha),
        'empirical_coverage': float(cov),
        'group_mae_sensor': mae_sensor,
        'group_mae_skin': mae_skin
    }
    print(json.dumps(report, indent=2))

    with open(os.path.join(args.artifacts, 'eval_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()
