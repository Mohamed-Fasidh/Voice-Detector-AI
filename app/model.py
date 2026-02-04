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

    proba = model.predict_proba(features)[0]
    human_prob = float(proba[0])
    ai_prob = float(proba[1])

    if ai_prob > human_prob:
        classification = "AI_GENERATED"
        confidence = round(ai_prob, 2)
        explanation = "Synthetic speech characteristics detected"
    else:
        classification = "HUMAN"
        confidence = round(human_prob, 2)
        explanation = "Natural human speech variability detected"

    return classification, confidence, explanation
