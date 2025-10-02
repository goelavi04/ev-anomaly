# convert_to_json.py
import pandas as pd
import os

# --- Input and a simpler Output Path ---
CSV_IN_PATH = os.path.join('data', 'labeled_dataset.csv')
JSON_OUT_PATH = os.path.join('data', 'sessions.json') # <-- We'll save it here

print(f"Reading data from {CSV_IN_PATH}...")
df = pd.read_csv(CSV_IN_PATH)

# Convert dataframe to a list of objects (JSON)
df.to_json(JSON_OUT_PATH, orient='records', indent=2)

print(f"Successfully created sessions.json inside your 'data' folder!")