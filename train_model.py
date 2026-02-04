import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from app.features import extract_features

DATASET = {
    "HUMAN": "data/Human",
    "AI": "data/AI"
}

X = []
y = []

print("🔊 Loading HUMAN samples...")
for file in os.listdir(DATASET["HUMAN"]):
    if file.endswith(".mp3"):
        path = os.path.join(DATASET["HUMAN"], file)
        try:
            audio, sr = librosa.load(path, sr=16000)
            feats = extract_features(audio, sr)
            X.append(feats)
            y.append(0)
            print(f"Loaded HUMAN: {file}")
        except Exception as e:
            print(f"Skipped {file}: {e}")

print("🤖 Loading AI samples...")
for file in os.listdir(DATASET["AI"]):
    if file.endswith(".mp3"):
        path = os.path.join(DATASET["AI"], file)
        try:
            audio, sr = librosa.load(path, sr=16000)
            feats = extract_features(audio, sr)
            X.append(feats)
            y.append(1)
            print(f"Loaded AI: {file}")
        except Exception as e:
            print(f"Skipped {file}: {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise RuntimeError("❌ No audio files loaded. Check data folders.")

print(f"\n✅ Total samples: {len(X)}")
print(f"📐 Feature dimension: {X.shape[1]}")

print("\n🚀 Training model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/voice_detector.pkl")

print("✅ Model saved to model/voice_detector.pkl")
