import numpy as np

def extract_features(y, sr):
    y = y.astype(np.float32)

    features = []

    # 1️⃣ Energy
    energy = np.mean(y ** 2)
    features.append(energy)

    # 2️⃣ Energy variance
    features.append(np.var(y ** 2))

    # 3️⃣ Zero crossing rate (manual, NO librosa)
    zero_crossings = np.mean(np.abs(np.diff(np.sign(y)))) / 2
    features.append(zero_crossings)

    # 4️⃣ Amplitude statistics
    features.append(np.mean(np.abs(y)))
    features.append(np.std(y))
    features.append(np.max(np.abs(y)))

    # 5️⃣ Simple spectral proxy (FFT)
    spectrum = np.abs(np.fft.rfft(y))
    features.append(np.mean(spectrum))
    features.append(np.var(spectrum))

    return np.array(features).reshape(1, -1)
