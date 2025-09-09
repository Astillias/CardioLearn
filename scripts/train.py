
import argparse, os, json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from cardiolearn.features import featurize_dataframe
from cardiolearn.models import build_models, save_models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/ppg_synth.csv')
    ap.add_argument('--out', type=str, default='artifacts')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X, y, meta = featurize_dataframe(df)
    Xtr, Xte, ytr, yte, mtr, mte = train_test_split(X, y, meta, test_size=0.2, random_state=0)

    models = build_models(X.columns.tolist())
    history = {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        mae = float(np.mean(np.abs(yte - mdl.predict(Xte))))
        history[name] = {'mae': mae}

    os.makedirs(args.out, exist_ok=True)
    save_models(models, args.out)
    with open(os.path.join(args.out, 'feature_names.json'), 'w') as f:
        json.dump(X.columns.tolist(), f, indent=2)
    with open(os.path.join(args.out, 'train_report.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print("Saved models and reports to", args.out)
    print(json.dumps(history, indent=2))

if __name__ == '__main__':
    main()
