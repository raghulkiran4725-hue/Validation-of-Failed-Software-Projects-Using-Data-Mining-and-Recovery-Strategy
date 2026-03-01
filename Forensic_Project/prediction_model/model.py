import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("🚀 Training model started...")

# Load dataset
data_path = "datasets/software_projects.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError("❌ Dataset not found")

df = pd.read_csv("datasets/software_projects_final.csv")

X = df.drop(["project", "label"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🎯 MODEL ACCURACY:", round(accuracy * 100, 2), "%")

# Save model
os.makedirs("prediction_model", exist_ok=True)
joblib.dump(model, "prediction_model/failure_model.pkl")

print("✅ Model saved at prediction_model/failure_model.pkl")
