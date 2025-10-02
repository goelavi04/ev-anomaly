# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- PATHS CORRECTED ---
DATA_PATH = os.path.join('data','ev_anomalies_full_synthetic.csv')
SCALER_PATH = os.path.join('models','scaler.joblib')
CLEAN_PATH = os.path.join('data','cleaned_dataset.csv')

def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    # parse datetimes
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    # duration in seconds
    df['duration_s'] = (df['end_time'] - df['start_time']).dt.total_seconds().fillna(0)
    # energy to payment ratio
    df['energy_to_payment'] = df['energy_kwh'] / (df['payment_amount'].replace(0, np.nan) + 1e-6)
    # normalize percentages to 0-1 if needed
    if df['cpu_percent'].max() > 1:
        df['cpu_percent'] = df['cpu_percent'] / 100.0
    if df['memory_percent'].max() > 1:
        df['memory_percent'] = df['memory_percent'] / 100.0
    # fill na
    df = df.fillna(0)
    return df

def add_derived(df):
    # simple derived features
    df['start_hour'] = df['start_time'].dt.hour
    # per-device seen count
    df['mac_count'] = df.groupby('mac_address')['mac_address'].transform('count')
    return df

def scale_features(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = ['cpu_percent','packets_per_sec','memory_percent','duration_s','energy_kwh','payment_amount','energy_to_payment','start_hour','mac_count']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    os.makedirs('models', exist_ok=True) # --- PATH CORRECTED ---
    joblib.dump(scaler, SCALER_PATH)
    return df, scaler

if __name__ == '__main__':
    df = load_and_clean()
    df = add_derived(df)
    df, scaler = scale_features(df)
    df.to_csv(CLEAN_PATH, index=False)
    print("Saved cleaned dataset to", CLEAN_PATH)
    print("Saved scaler to", SCALER_PATH)