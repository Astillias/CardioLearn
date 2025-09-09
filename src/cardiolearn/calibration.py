
import numpy as np
def split_conformal_intervals(y_true_cal, y_pred_cal, alpha=0.1):
    resid = np.abs(y_true_cal - y_pred_cal)
    q = np.quantile(resid, 1 - alpha)
    return q
def apply_interval(y_pred, q):
    lo = y_pred - q; hi = y_pred + q
    return lo, hi
def coverage(y_true, lo, hi):
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
