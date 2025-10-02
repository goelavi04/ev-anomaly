# src/train_dos.py
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest

# --- PATHS CORRECTED ---
LABELED_PATH = os.path.join('data','labeled_dataset.csv')
MODEL_PATH = os.path.join('models','if_dos.joblib')

def train():
    df = pd.read_csv(LABELED_PATH)
    features = ['cpu_percent','packets_per_sec','memory_percent']
    X = df[features].fillna(0).values
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    model.fit(X)
    os.makedirs('models', exist_ok=True) # --- PATH CORRECTED ---
    joblib.dump((model, features), MODEL_PATH)
    print("Saved DoS model to", MODEL_PATH)

if __name__ == '__main__':
    train()