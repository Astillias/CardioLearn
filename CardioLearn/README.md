# CardioLearn: Cuffless BP Estimation & Uncertainty with PPG (Recruiter-Ready Project)

**Pitch:** An end-to-end, clinically-inspired ML pipeline that estimates **systolic blood pressure (SBP)** from **PPG** waveforms and demographics, with **uncertainty quantification** (conformal prediction), **domain-shift checks**, and a **Streamlit demo**. It uses **fully synthetic** signals so it’s safe to share publicly.

> Why this is impressive
- Biomedical signal processing (PPG cleaning, feature engineering)
- ML system design (train/eval scripts, modular package)
- **Uncertainty**: distribution-free prediction intervals via conformal
- **Robustness/Fairness**: group-wise error slices + domain-shift experiment
- App layer: Streamlit demo for recruiters/PMs/clinicians

## Quickstart

```bash
python -m venv venv && source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
# (optional) generate fresh synthetic data
python -m cardiolearn.simulate --n 2000 --seed 7 --out data/ppg_synth.csv

# Train
python scripts/train.py --data data/ppg_synth.csv --out artifacts

# Evaluate (includes conformal intervals + group metrics)
python scripts/evaluate.py --data data/ppg_synth.csv --artifacts artifacts

# Run the demo app
streamlit run app/streamlit_app.py
```

## Repo layout
```
CardioLearn/
  ├─ src/cardiolearn/
  │   ├─ __init__.py
  │   ├─ simulate.py         # synthetic PPG + SBP generator
  │   ├─ features.py         # signal features (time/freq/derivative)
  │   ├─ models.py           # sklearn pipelines + quantile models
  │   ├─ calibration.py      # conformal intervals
  │   ├─ fairness.py         # group metrics + slicing
  ├─ scripts/
  │   ├─ train.py            # trains regressors + saves artifacts
  │   └─ evaluate.py         # eval + intervals + robustness
  ├─ app/streamlit_app.py    # interactive demo
  ├─ data/ppg_synth.csv      # small sample dataset
  ├─ requirements.txt
  └─ README.md
```

## What’s novel here?
- **Physiology-aware simulation:** Generates multi-morphology PPG with realistic heart-rate variability, noise, and *domain shifts* (sensor type group A vs B).
- **Feature set** spans pulse morphology (rise time, dicrotic notch proxy), variability, frequency content.
- **Uncertainty** via split-conformal prediction for **calibrated intervals** that maintain nominal coverage.
- **Bias/robustness** slices across `sensor_group` and `skin_tone_proxy` (synthetic), highlighting where the model under-performs.

## Notes
- All data are **synthetic**; no medical advice. This is a **research demo** to showcase engineering ability.
