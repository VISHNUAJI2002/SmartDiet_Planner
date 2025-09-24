import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os

# Make sure model directory exists
os.makedirs("model", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/synthetic_diet_dataset_encoded.csv")

# --- Handle missing values ---
df.replace("", np.nan, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# 2. Encode categorical fields if they exist
if "gender" in df.columns:
    df["gender"] = LabelEncoder().fit_transform(df["gender"].astype(str))

if "chronic_disease" in df.columns:
    df["chronic_disease"] = LabelEncoder().fit_transform(df["chronic_disease"].astype(str))

# 3. Features (X) and Target (y)
X = df.drop(["Meal_Plan", "Meal_Plan_Encoded"], axis=1)
y = df["Meal_Plan_Encoded"]

# Encode target labels
meal_encoder = LabelEncoder()
y = meal_encoder.fit_transform(y)

# Save encoder for Flask app
joblib.dump(meal_encoder, "model/meal_encoder.pkl")

# 4. Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 6. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

# 7. Build neural network model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 8. Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1
)

# 9. Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Convert to string labels for readability
class_names = [str(c) for c in meal_encoder.classes_]
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# 10. Save model
# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Save in new Keras format (.keras)
try:
    model.save("model/diet_model.keras")   # no save_format needed
    print("✅ Model saved in Keras format: model/diet_model.keras")
except Exception as e:
    print("⚠️ Keras format save failed:", e)

# Save fallback in .h5 format
try:
    model.save("model/diet_model.h5")
    print("✅ Model also saved in HDF5 format: model/diet_model.h5")
except Exception as e:
    print("❌ HDF5 save failed:", e)


