import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Simulated but structured data (acceptable for hackathon)
X_human = np.random.normal(loc=0.6, scale=0.15, size=(200, 8))
X_ai = np.random.normal(loc=0.3, scale=0.1, size=(200, 8))

X = np.vstack([X_human, X_ai])
y = np.array([0]*200 + [1]*200)  # 0=HUMAN, 1=AI

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/voice_detector.pkl")

print("Model trained successfully (stable features)")
