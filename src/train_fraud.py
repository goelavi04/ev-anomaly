# src/train_fraud.py
import pandas as pd
import joblib
import os
# --- NEW MODEL IMPORTED ---
from sklearn.neighbors import LocalOutlierFactor

LABELED_PATH = os.path.join('data','labeled_dataset.csv')
MODEL_PATH = os.path.join('models','lof_fraud.joblib') # Renamed the output file

def train():
    df = pd.read_csv(LABELED_PATH)
    # features for fraud detection
    features = ['energy_kwh','payment_amount','duration_s','energy_to_payment']
    X = df[features].fillna(0).values

    # --- NEW MODEL CREATED ---
    # We use the settings that worked best in our test
    model = LocalOutlierFactor(n_neighbors=35, contamination=0.15, novelty=True)
    
    # In a real-world unsupervised script, you'd train on all data.
    # For LOF, it's often best to train only on what you assume is normal data.
    # Since our synthetic data has labels, we can do that here.
    y_true = df['is_free_charge']
    model.fit(X[y_true==0]) # Fitting only on normal data

    os.makedirs('models', exist_ok=True)
    joblib.dump((model, features), MODEL_PATH)
    print("Saved Fraud (LocalOutlierFactor) model to", MODEL_PATH)

if __name__ == '__main__':
    train()