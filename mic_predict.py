import json
import time
import queue

import librosa
import numpy as np
import onnxruntime as ort
import sounddevice as sd

MODEL_PATH = "best_model/model.onnx"
CONFIG_PATH = "best_model/cardiag_config.json"

SAMPLE_RATE = 16000
RECORD_SECONDS = 3

# -------------------------
# Load config
# -------------------------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

labels = config["label_names"]

feature_mean = np.array(
    config["feature_mean"],
    dtype=np.float32
)

feature_scale = np.array(
    config["feature_scale"],
    dtype=np.float32
)

thresholds = np.array(
    config["thresholds"],
    dtype=np.float32
)

# -------------------------
# ONNX session
# -------------------------
session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name

# -------------------------
# Feature extraction
# -------------------------
def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=20
    )

    features = []

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    features = np.array(
        features,
        dtype=np.float32
    )

    input_size = len(feature_mean)

    if len(features) < input_size:
        features = np.pad(
            features,
            (0, input_size - len(features))
        )

    features = features[:input_size]

    features = (
        features - feature_mean
    ) / feature_scale

    return features.reshape(1, -1)

# -------------------------
# Predict
# -------------------------
def predict(audio):

    x = extract_features(audio)

    logits = session.run(
        None,
        {input_name: x}
    )[0]

    probs = 1 / (1 + np.exp(-logits))

    print("\n=============================")

    for label, prob, thr in zip(
        labels,
        probs[0],
        thresholds
    ):

        detected = prob >= thr

        status = "[ALERT]" if detected else "[OK]"

        bar = "#" * int(prob * 20)

        print(
            f"{status:8} "
            f"{label:20} "
            f"{prob:.1%} "
            f"{bar}"
        )

# -------------------------
# Record loop
# -------------------------
def main():

    print("\nCar Diagnostics Mic Mode")
    print("Press CTRL+C to stop.\n")

    while True:

        print(
            f"\nRecording {RECORD_SECONDS} seconds..."
        )

        recording = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = recording.flatten()

        predict(audio)

        time.sleep(0.25)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    main()