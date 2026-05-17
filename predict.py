import json
import sys
import numpy as np
import librosa
import onnxruntime as ort

MODEL_PATH = "model/model.onnx"
CONFIG_PATH = "model/cardiag_config.json"

# -------------------------
# Load config
# -------------------------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

labels = config["label_names"]
feature_mean = np.array(config["feature_mean"])
feature_scale = np.array(config["feature_scale"])
thresholds = np.array(config["thresholds"])

# -------------------------
# Feature extraction
# -------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = []

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    features = np.array(features, dtype=np.float32)

    # pad/truncate to expected size
    input_size = len(feature_mean)

    if len(features) < input_size:
        features = np.pad(features, (0, input_size - len(features)))

    features = features[:input_size]

    # normalize
    features = (features - feature_mean) / feature_scale

    return features.reshape(1, -1).astype(np.float32)

# -------------------------
# Predict
# -------------------------
def predict(audio_path):
    session = ort.InferenceSession(MODEL_PATH)

    x = extract_features(audio_path)

    input_name = session.get_inputs()[0].name

    logits = session.run(None, {input_name: x})[0]

    probs = 1 / (1 + np.exp(-logits))

    print("\nPredictions:\n")

    for label, prob, thr in zip(labels, probs[0], thresholds):
        detected = prob >= thr

        status = "[ALERT]" if detected else "[OK]"

        print(f"{status:8} {label:20} {prob:.1%}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python predict.py audio.wav")
        sys.exit(1)

    predict(sys.argv[1])