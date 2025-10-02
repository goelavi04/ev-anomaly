# src/evaluate_dos.py
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the labeled dataset and the trained DoS model
LABELED_PATH = os.path.join('data','labeled_dataset.csv')
DOS_MODEL_PATH = os.path.join('models','if_dos.joblib')

print("Loading data and model for DoS evaluation...")
df = pd.read_csv(LABELED_PATH)
(model, features) = joblib.load(DOS_MODEL_PATH)

# --- Prepare Data for Evaluation ---

# The "ground truth" is the 'is_dos' column
y_true = df['is_dos']

# The data features the model will use to make predictions
X_test = df[features].fillna(0)

# --- Get Model's Predictions ---

# Use the trained model to predict anomalies (-1) vs normal (1)
predictions_raw = model.predict(X_test)

# Convert model's output (-1) to our format (1)
y_pred = [1 if x == -1 else 0 for x in predictions_raw]

# --- Calculate and Display Results ---

print("\n--- DoS Model Evaluation Results ---")
print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.2%}\n")

# Display the main classification metrics
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'DoS Attack (1)']))

# Display the Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
print("\nMatrix Explained:")
print(f"Top-Left (True Negatives):  {cm[0][0]} -> Normal sessions correctly identified as Normal.")
print(f"Top-Right (False Positives): {cm[0][1]} -> Normal sessions INCORRECTLY flagged as DoS.")
print(f"Bottom-Left (False Negatives): {cm[1][0]} -> Real DoS attacks that the model MISSED.")
print(f"Bottom-Right (True Positives): {cm[1][1]} -> Real DoS attacks that the model CORRECTLY CAUGHT.")
print("------------------------------------")