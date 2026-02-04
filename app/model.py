import joblib
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
    probs = model.predict_proba(features)[0]

    human_prob = probs[0]
    ai_prob = probs[1]

    if ai_prob > human_prob:
        return (
            "AI_GENERATED",
            float(ai_prob),
            "Unnatural pitch consistency and spectral smoothness detected"
        )
    else:
        return (
            "HUMAN",
            float(human_prob),
            "Natural human speech variability detected"
        )

