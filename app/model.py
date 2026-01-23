import joblib

MODEL_PATH = "model/voice_detector.pkl"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_voice(features):
    model = get_model()

    prob = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]

    label = "AI_GENERATED" if prediction == 1 else "HUMAN"
    confidence = float(max(prob))

    explanation = (
        "Unnatural pitch consistency and spectral patterns detected"
        if label == "AI_GENERATED"
        else "Natural human speech variability detected"
    )

    return label, confidence, explanation
