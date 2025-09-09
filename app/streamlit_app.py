import streamlit as st, pandas as pd, numpy as np, json, joblib, os
from cardiolearn.features import featurize_dataframe
from cardiolearn.calibration import apply_interval

st.set_page_config(page_title="CardioLearn Demo", layout="wide")
st.title("CardioLearn: Cuffless SBP Estimation with Uncertainty")

st.markdown("""Upload a CSV like **data/ppg_synth.csv** (ppg_json, fs, age, bmi, sex, sensor_group, skin_tone_proxy).
This demo loads a trained **GradientBoostingRegressor** and a conformal radius saved in artifacts.""")

art_dir = st.sidebar.text_input("Artifacts directory", "artifacts")
model_path = os.path.join(art_dir, "gbr.joblib")
q_path = os.path.join(art_dir, "eval_report.json")

uploaded = st.file_uploader("CSV file", type=['csv'])
if uploaded is not None and os.path.exists(model_path) and os.path.exists(q_path):
    df = pd.read_csv(uploaded)
    X, y, meta = featurize_dataframe(df)
    model = joblib.load(model_path)
    yhat = model.predict(X)
    q = json.load(open(q_path))['conformal_q']
    lo, hi = apply_interval(yhat, q)

    st.subheader("Predictions")
    out = pd.DataFrame({
        'sbp_pred': yhat,
        'lo': lo,
        'hi': hi,
        'sensor_group': meta['sensor_group'],
        'skin_tone_proxy': meta['skin_tone_proxy']
    })
    if 'sbp' in df.columns:
        out['sbp_true'] = df['sbp']
        out['abs_err'] = np.abs(out['sbp_true'] - out['sbp_pred'])
        st.write("Test MAE:", float(out['abs_err'].mean()))
    st.dataframe(out.head(20))
else:
    st.info("Upload CSV and ensure artifacts exist (run training & evaluation first).")
