# src/test_unseen_data.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- NEW MODEL IMPORTED ---
from sklearn.neighbors import LocalOutlierFactor

# --- SETUP ---
# Load the full labeled dataset
LABELED_PATH = os.path.join('data','labeled_dataset.csv')
df = pd.read_csv(LABELED_PATH)

# Define the features (X) and the true labels (y) for the fraud model
features = ['energy_kwh','payment_amount','duration_s','energy_to_payment']
X = df[features].fillna(0)
y = df['is_free_charge'] # This is our "answer key"

# --- THE TRAIN-TEST SPLIT ---
# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data split into: {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- MODEL TRAINING ---
# IMPORTANT: The model is trained ONLY on the training data (X_train)
print("\nTraining the new LocalOutlierFactor model...")
# --- NEW MODEL CREATED ---
model = LocalOutlierFactor(n_neighbors=35, contamination=0.15, novelty=True)
# Note: For LOF, we fit it only on the "normal" data for better performance
# In our training data, y_train == 0 are the normal samples
model.fit(X_train[y_train == 0])
print("Model training complete.")

# --- TESTING ON UNSEEN DATA ---
# Now, we test the trained model on the unseen testing data (X_test)
print("\nTesting the model on UNSEEN data...")
predictions_raw = model.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in predictions_raw]

# --- RESULTS ---
print("\n--- Evaluation Results on Unseen Data (LOF Model) ---")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("-------------------------------------------------------")