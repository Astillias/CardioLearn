# CardioLearn: Cuffless BP Estimation & Uncertainty with PPG

End-to-end, clinically-inspired ML that estimates **systolic BP (SBP)** from **PPG** waveforms + demographics, with **uncertainty** (conformal) and **fairness/robustness slices**. All data are **synthetic**.

## Quickstart (Colab)
pip install -r requirements.txt
# Train
python scripts/train.py --data data/ppg_synth.csv --out artifacts
# Evaluate
python scripts/evaluate.py --data data/ppg_synth.csv --artifacts artifacts
