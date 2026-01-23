import numpy as np
from scipy.signal import welch

def extract_features(y, sr):
    y = y.astype(np.float32)

    features = []

    # 1. Signal energy
    energy = np.mean(y ** 2)
    features.append(energy)

    # 2. Energy variance
    features.append(np.var(y ** 2))

    # 3. Zero Crossing Rate
    zcr = np.mean(np.abs(np.diff(np.sign(y)))) / 2
    features.append(zcr)

    # 4. Spectral features (Welch PSD)
    freqs, psd = welch(y, sr)

    features.append(np.mean(psd))      # Spectral mean
    features.append(np.std(psd))       # Spectral variance
    features.append(np.max(psd))       # Spectral peak

    # 5. Spectral entropy
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    features.append(spectral_entropy)

    # 6. Pitch proxy (peak frequency)
    peak_freq = freqs[np.argmax(psd)]
    features.append(peak_freq)

    # Pad to fixed length (important for model)
    while len(features) < 30:
        features.append(0.0)

    return np.array(features).reshape(1, -1)
