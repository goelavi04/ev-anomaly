# src/apply_simple_mapping.py
import pandas as pd
import os

SRC = os.path.join('data','ev_anomalies_preprocessed.csv')
OUT = os.path.join('data','ev_anomalies_renamed.csv')

print("Reading", SRC)
df = pd.read_csv(SRC)

# mapping: original name -> new canonical name
rename = {
    'dataenergy_kWh': 'energy_kwh',
    'amount_INR': 'payment_amount'
}

# apply renaming (only columns that exist will be renamed)
existing_rename = {k:v for k,v in rename.items() if k in df.columns}
if existing_rename:
    df = df.rename(columns=existing_rename)
    print("Renamed columns:", existing_rename)
else:
    print("No columns renamed (check column names).")

# convert duration minutes to seconds if present
if 'session_duration_min' in df.columns:
    df['duration_s'] = pd.to_numeric(df['session_duration_min'], errors='coerce').fillna(0) * 60
    print("Created duration_s from session_duration_min")
else:
    df['duration_s'] = 0
    print("session_duration_min not found; set duration_s=0")

# Ensure numeric types for key columns
for c in ['energy_kwh','payment_amount','duration_s']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

os.makedirs('data', exist_ok=True)
df.to_csv(OUT, index=False)

print("Wrote renamed CSV to", OUT)
print("Columns now:", list(df.columns))