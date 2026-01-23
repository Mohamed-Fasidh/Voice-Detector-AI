import base64
import tempfile
import subprocess
import numpy as np
import soundfile as sf
import os

def decode_base64_mp3(audio_base64: str):
    """
    Decodes a Base64-encoded MP3 file into a waveform and sample rate.
    Uses FFmpeg for MP3 decoding (stable on Windows).
    """

    # Decode base64 to raw bytes
    audio_bytes = base64.b64decode(audio_base64)

    # Create temporary MP3 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
        mp3_file.write(audio_bytes)
        mp3_path = mp3_file.name

    # Prepare temporary WAV file path
    wav_path = mp3_path.replace(".mp3", ".wav")

    try:
        # Convert MP3 → WAV using ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path, wav_path],
            check=True
        )

        # Read WAV file
        y, sr = sf.read(wav_path)

        # Convert stereo → mono if needed
        if y.ndim > 1:
            y = y.mean(axis=1)

        return y.astype(np.float32), sr

    finally:
        # Cleanup temp files
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
