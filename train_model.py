import os

# ---------- STABILITY FIXES (MUST BE FIRST) ----------
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---------------------------------------------------

import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from app.features import extract_features

# -------- PATHS --------
HUMAN_DIR = "data/Human"
AI_DIR = "data/AI"
MODEL_PATH = "model/voice_detector.pkl"

# -------- LOAD DATA --------
X = []
y = []

def load_audio_files(folder, label):
    for file in os.listdir(folder):
        if file.lower().endswith(".mp3"):
            path = os.path.join(folder, file)
            try:
                audio, sr = librosa.load(path, sr=None)
                features = extract_features(audio, sr)
                X.append(features)
                y.append(label)
                print(f"Loaded: {path}")
            except Exception as e:
                print(f"Skipped {path}: {e}")

print("\n🔊 Loading HUMAN samples...")
load_audio_files(HUMAN_DIR, label=0)   # 0 = HUMAN

print("\n🤖 Loading AI samples...")
load_audio_files(AI_DIR, label=1)      # 1 = AI_GENERATED

X = np.array(X)
y = np.array(y)

print(f"\nTotal samples: {len(X)}")
print(f"Feature dimension: {X.shape[1]}")

# -------- TRAIN / TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------- MODEL --------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

print("\n🚀 Training model...")
model.fit(X_train, y_train)

# -------- EVALUATION --------
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["HUMAN", "AI_GENERATED"]))

# -------- SAVE MODEL --------
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\n✅ Model saved to {MODEL_PATH}")
