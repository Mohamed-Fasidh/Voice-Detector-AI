import os

# Windows stability fixes (MUST be first)
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from app.auth import validate_api_key
from app.audio_utils import decode_base64_mp3
from app.features import extract_features
from app.model import predict_voice
from app.config import SUPPORTED_LANGUAGES
from typing import Optional
app = FastAPI(title="AI Voice Detection API")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str
    
@app.post("/api/voice-detection")
def detect_voice(data: VoiceRequest, api_key=Depends(validate_api_key)):

    if data.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    y, sr = decode_base64_mp3(data.audioBase64)
    features = extract_features(y, sr)

    label, confidence, explanation = predict_voice(features)

    return {
        "status": "success",
        "language": data.language,
        "classification": label,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
