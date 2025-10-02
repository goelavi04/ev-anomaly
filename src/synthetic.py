# src/synthetic.py
import pandas as pd
import numpy as np
import os

CLEAN_PATH = os.path.join('data','cleaned_dataset.csv')
LABELED_PATH = os.path.join('data','labeled_dataset.csv')

def inject_dos(df, frac=0.01, cpu_spike=1.5):
    df = df.copy()
    n = max(1, int(len(df)*frac))
    idx = np.random.choice(df.index, n, replace=False)
    df.loc[idx, 'cpu_percent'] = df.loc[idx, 'cpu_percent'] + cpu_spike
    df.loc[idx, 'is_dos'] = 1
    return df

def inject_free_charge(df, frac=0.01):
    df = df.copy()
    n = max(1, int(len(df)*frac))
    idx = np.random.choice(df.index, n, replace=False)
    df.loc[idx, 'payment_amount'] = 0.0
    df.loc[idx, 'is_free_charge'] = 1
    return df

def inject_multiuser_conflict(df, frac=0.005):
    df = df.copy()
    n = max(1, int(len(df)*frac))
    for _ in range(n):
        chargers = df['charger_id'].dropna().unique()
        if len(chargers) < 2:
            continue
        a, b = np.random.choice(chargers, 2, replace=False)
        t = df['start_time'].sample(1).iloc[0]
        user = 'SYN_USER_' + str(np.random.randint(1e6))
        row = {'session_id': 'SYN_'+user+'_A', 'user_id': user, 'charger_id': a, 'start_time': t, 'end_time': t + pd.Timedelta(minutes=30), 'cpu_percent': 0.01, 'packets_per_sec': 1, 'memory_percent': 0.01, 'energy_kwh': 5.0, 'payment_amount': 0.0, 'mac_address': 'SYN_MAC_'+user, 'duration_s': 1800, 'energy_to_payment': 5.0 }
        row2 = row.copy()
        row2['session_id'] = 'SYN_'+user+'_B'
        row2['charger_id'] = b
        row2['start_time'] = t + pd.Timedelta(minutes=5)
        row2['end_time'] = t + pd.Timedelta(minutes=35)
        df = pd.concat([df, pd.DataFrame([row]), pd.DataFrame([row2])], ignore_index=True)
    
    # --- THIS IS THE CORRECTED LOGIC ---
    df['is_multi_conflict'] = 0
    df.loc[df['user_id'].str.startswith('SYN_USER_', na=False), 'is_multi_conflict'] = 1
    return df

if __name__ == '__main__':
    df = pd.read_csv(CLEAN_PATH, parse_dates=['start_time','end_time'])
    df = inject_dos(df, frac=0.02)
    df = inject_free_charge(df, frac=0.10)
    df = inject_multiuser_conflict(df, frac=0.005)
    df[['is_dos','is_free_charge','is_multi_conflict']] = df[['is_dos','is_free_charge','is_multi_conflict']].fillna(0)
    df.to_csv(LABELED_PATH, index=False)
    print("Saved labeled dataset to", LABELED_PATH)