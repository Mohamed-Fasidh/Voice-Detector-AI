import numpy as np
import librosa
def extract_features(y, sr):
    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Delta MFCC (important for AI smoothness)
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[pitches > 0]
    features.append(np.var(pitch) if len(pitch) else 0)
    features.append(np.mean(pitch) if len(pitch) else 0)

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.var(rms))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    # Spectral features (AI voices fail here)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(spec_centroid))
    features.append(np.var(spec_centroid))

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(spec_rolloff))

    return np.array(features)

