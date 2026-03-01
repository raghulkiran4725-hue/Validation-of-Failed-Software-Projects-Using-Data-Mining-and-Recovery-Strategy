import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Dummy training data
# [stars, forks, open_issues, watchers, size]
X = np.array([
    [50000, 10000, 200, 50000, 40000],   # good project
    [10, 2, 80, 5, 500],                 # risky project
    [30000, 8000, 100, 30000, 25000],
    [5, 1, 40, 2, 300]
])

y = np.array([1, 0, 1, 0])  # 1 = Low risk, 0 = High risk

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "prediction_model/trained_model.pkl")

print("✅ Model trained and saved successfully")
