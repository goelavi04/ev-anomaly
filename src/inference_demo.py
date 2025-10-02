# src/inference_demo.py
import pandas as pd
import joblib
import os
import numpy as np

LABELED_PATH = os.path.join('data','labeled_dataset.csv')
DOS_MODEL_PATH = os.path.join('models','if_dos.joblib')
FRAUD_MODEL_PATH = os.path.join('models','if_fraud.joblib')

def load_models():
    (if_dos, dos_feats) = joblib.load(DOS_MODEL_PATH)
    (if_fraud, fraud_feats) = joblib.load(FRAUD_MODEL_PATH)
    return (if_dos, dos_feats), (if_fraud, fraud_feats)

def score_row(row, model_tuple):
    model, features = model_tuple
    X = np.array(row[features].fillna(0)).reshape(1, -1)
    score = -model.decision_function(X)[0]
    flag = int(model.predict(X)[0] == -1)
    return score, flag

if __name__ == '__main__':
    df = pd.read_csv(LABELED_PATH)
    (dos_model, dos_feats), (fraud_model, fraud_feats) = load_models()
    
    print("--- Scoring first 10 rows for anomaly detection ---")
    for idx, row in df.head(10).iterrows():
        dos_score, dos_flag = score_row(row, (dos_model, dos_feats))
        fraud_score, fraud_flag = score_row(row, (fraud_model, fraud_feats))
        rule_flag = int((row['energy_kwh']>2) and (row['payment_amount']==0))
        
        # This is the corrected line
        print(f"Row {idx}: DoS Score={dos_score:.3f} (Flag: {dos_flag}) | Fraud Score={fraud_score:.3f} (Flag: {fraud_flag}) | Rule-Based Fraud: {rule_flag}")