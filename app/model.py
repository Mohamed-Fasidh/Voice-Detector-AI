import joblib
import numpy as np
from app.config import MODEL_PATH

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_voice(features):
    model = get_model()

    features = features.reshape(1, -1)
    proba = model.predict_proba(features)[0]

    ai_prob = float(proba[1])
    human_prob = float(proba[0])

    if ai_prob > human_prob:
        return "AI_GENERATED", ai_prob, "Synthetic speech patterns detected"
    else:
        return "HUMAN", human_prob, "Natural human voice variations detected"
