
import os

API_KEY = "sk_cf5fd6256a1bf6609ae3529a8aa8a1edc85db4007d00772b"

SUPPORTED_LANGUAGES = [
    "Tamil", "English", "Hindi", "Malayalam", "Telugu"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "voice_detector.pkl")