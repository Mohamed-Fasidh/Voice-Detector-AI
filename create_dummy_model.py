# create_dummy_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

X = np.random.rand(100, 30)
y = np.random.randint(0, 2, 100)

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/voice_detector.pkl")

print("Model rebuilt successfully")
