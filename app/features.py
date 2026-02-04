import numpy as np
import librosa

def extract_features(y, sr):
    features = []

    # 1️⃣ MFCCs (SAFE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2️⃣ RMS Energy (SAFE)
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # 3️⃣ Spectral Centroid (SAFE)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # 4️⃣ Spectral Bandwidth (SAFE)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    return np.array(features, dtype=np.float32)
