# src/visualize_fraud.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load the data and the trained model
LABELED_PATH = os.path.join('data','labeled_dataset.csv')
FRAUD_MODEL_PATH = os.path.join('models','if_fraud.joblib')
df = pd.read_csv(LABELED_PATH)
(model, features) = joblib.load(FRAUD_MODEL_PATH)

# Calculate anomaly scores for the whole dataset
scores = -model.decision_function(df[features].fillna(0))

# Create the plot
plt.figure(figsize=(12, 7))
scatter = plt.scatter(df['energy_kwh'], df['payment_amount'], c=scores, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Anomaly Score (Higher is more anomalous)')
plt.title('Fraud Detection: Energy vs. Payment')
plt.xlabel('Energy Consumed (kWh)')
plt.ylabel('Payment Amount')
plt.grid(True)
plt.show()

print("Plot window opened. Close the plot window to finish the script.")