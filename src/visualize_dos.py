# src/visualize_dos.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load the data and the trained DoS model
LABELED_PATH = os.path.join('data','labeled_dataset.csv')
DOS_MODEL_PATH = os.path.join('models','if_dos.joblib')

print("Loading data and DoS model for visualization...")
df = pd.read_csv(LABELED_PATH, parse_dates=['start_time'])
(model, features) = joblib.load(DOS_MODEL_PATH)

# Get the model's predictions (-1 for anomalies, 1 for normal)
predictions = model.predict(df[features].fillna(0))
df['is_anomaly_prediction'] = [1 if x == -1 else 0 for x in predictions]

# Separate the anomalies for plotting
anomalies = df[df['is_anomaly_prediction'] == 1]

print("Creating plot...")
# Create the plot
plt.figure(figsize=(15, 7))
# Plot all data points as a blue line
plt.plot(df['start_time'], df['cpu_percent'], label='Normal CPU Usage', color='blue', alpha=0.7)
# Plot the detected anomalies as red dots on top
plt.scatter(anomalies['start_time'], anomalies['cpu_percent'], color='red', s=50, label='Detected DoS Anomaly', zorder=5)

plt.title('Time Series of CPU Usage with Detected DoS Anomalies')
plt.xlabel('Time of Charging Session')
plt.ylabel('CPU Usage (%)')
plt.legend()
plt.grid(True)
plt.show()

print("Plot window opened. Close the plot window to finish the script.")