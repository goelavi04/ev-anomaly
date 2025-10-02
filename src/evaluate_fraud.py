# src/evaluate_fraud.py
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the labeled dataset and the trained fraud model
LABELED_PATH = os.path.join('data','labeled_dataset.csv')
FRAUD_MODEL_PATH = os.path.join('models','if_fraud.joblib')

print("Loading data and model for evaluation...")
df = pd.read_csv(LABELED_PATH)
(model, features) = joblib.load(FRAUD_MODEL_PATH)

# --- Prepare Data for Evaluation ---

# The "ground truth" or the "answer key"
# We'll use the 'is_free_charge' column as our true label for fraud
y_true = df['is_free_charge']

# The data features the model will use to make predictions
X_test = df[features].fillna(0)

# --- Get Model's Predictions ---

# Use the trained model to predict which rows are anomalies (-1) or normal (1)
predictions_raw = model.predict(X_test)

# Convert the model's output (-1 for anomaly, 1 for normal) 
# to our format (1 for anomaly, 0 for normal)
y_pred = [1 if x == -1 else 0 for x in predictions_raw]

# --- Calculate and Display Results ---

print("\n--- Model Evaluation Results ---")
print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.2%}\n")

# Display the main classification metrics (Precision, Recall, F1-Score)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

# Display the Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
print("\nMatrix Explained:")
print(f"Top-Left (True Negatives):  {cm[0][0]} -> Normal sessions correctly identified as Normal.")
print(f"Top-Right (False Positives): {cm[0][1]} -> Normal sessions INCORRECTLY flagged as Fraud.")
print(f"Bottom-Left (False Negatives): {cm[1][0]} -> Real Fraud that the model MISSED.")
print(f"Bottom-Right (True Positives): {cm[1][1]} -> Real Fraud that the model CORRECTLY CAUGHT.")
print("---------------------------------")