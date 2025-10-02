# src/detect_multiuser.py
import pandas as pd
import os

# --- PATHS CORRECTED ---
LABELED_PATH = os.path.join('data','labeled_dataset.csv')
OUT_PATH = os.path.join('data','multiuser_conflicts.csv')

def detect_overlaps(df):
    df = df.sort_values(['user_id','start_time'])
    conflict_rows = []
    for user, g in df.groupby('user_id'):
        g = g.sort_values('start_time')
        for i in range(len(g)-1):
            a = g.iloc[i]
            b = g.iloc[i+1]
            if pd.to_datetime(a['end_time']) > pd.to_datetime(b['start_time']):
                conflict_rows.append(a.to_dict())
                conflict_rows.append(b.to_dict())
    return pd.DataFrame(conflict_rows).drop_duplicates()

if __name__ == '__main__':
    df = pd.read_csv(LABELED_PATH, parse_dates=['start_time','end_time'])
    conflicts = detect_overlaps(df)
    conflicts.to_csv(OUT_PATH, index=False)
    print("Saved multiuser conflicts to", OUT_PATH, "rows:", len(conflicts))