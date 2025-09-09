
import joblib, os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def build_models(feature_names):
    num_features = [f for f in feature_names if not f.endswith(('_A','_M','_dark','_medium'))]
    pre = ColumnTransformer([('num', StandardScaler(), num_features)], remainder='passthrough')
    gbr = GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3, learning_rate=0.05)
    rf = RandomForestRegressor(random_state=0, n_estimators=400, n_jobs=-1)
    gbr_q10 = GradientBoostingRegressor(loss='quantile', alpha=0.10, n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    gbr_q90 = GradientBoostingRegressor(loss='quantile', alpha=0.90, n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)

    return {
        'gbr': Pipeline([('pre', pre), ('model', gbr)]),
        'rf': Pipeline([('pre', pre), ('model', rf)]),
        'q10': Pipeline([('pre', pre), ('model', gbr_q10)]),
        'q90': Pipeline([('pre', pre), ('model', gbr_q90)]),
    }

def save_models(models, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, mdl in models.items():
        joblib.dump(mdl, os.path.join(out_dir, f"{name}.joblib"))
