"""
Group-based metrics for robustness/fairness slices.
"""
import numpy as np, pandas as pd

def group_mae(y_true, y_pred, groups: pd.Series):
    df = pd.DataFrame({'y': y_true, 'yp': y_pred, 'g': groups.values})
    out = df.groupby('g').apply(lambda d: np.mean(np.abs(d['y']-d['yp']))).to_dict()
    out['overall'] = float(np.mean(np.abs(y_true - y_pred)))
    return out
