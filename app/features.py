import numpy as np
import librosa

def extract_features(y, sr):
    features = []

    # 1️⃣ MFCCs (SAFE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2️⃣ Energy (SAFE)
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # 3️⃣ Signal Variance (AI voices are more uniform)
    features.append(np.var(y))

    # 4️⃣ Zero-crossing (PURE NUMPY — NO NUMBA)
    zero_crossings = np.mean(np.abs(np.diff(np.sign(y))))
    features.append(zero_crossings)

    return np.array(features, dtype=np.float32)
