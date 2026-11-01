import numpy as np
import pandas as pd
import tensorflow as tf
import json
from collections import Counter
import os

# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = "models/autoencoder.h5"
DATA_PATH = "data/processed/44_processed.csv"
MAP_PATH = "data/sensor_cluster_map.json"
OUTPUT_PATH = "data/processed/anomaly_with_root_cause.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
print("✅ Loading data...")
df = pd.read_csv(DATA_PATH, index_col=0)
feature_names = df.columns.tolist()
X = df.values

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
print("✅ Loading trained autoencoder...")
autoencoder = tf.keras.models.load_model(MODEL_PATH,compile=False)

# -----------------------------
# RECONSTRUCTION
# -----------------------------
print("✅ Running inference...")
X_reconstructed = autoencoder.predict(X, verbose=0)

# -----------------------------
# RECONSTRUCTION ERROR
# -----------------------------
reconstruction_error = np.mean(np.square(X - X_reconstructed), axis=1)

# -----------------------------
# ANOMALY THRESHOLD (99.5 PERCENTILE)
# -----------------------------
threshold = np.mean(reconstruction_error) + 4 * np.std(reconstruction_error)
anomalies = reconstruction_error > threshold

print(f"✅ Anomaly threshold set to: {threshold:.6f}")
print(f"✅ Total anomalies detected: {np.sum(anomalies)}")

# -----------------------------
# LOAD SENSOR → SUBSYSTEM MAP
# -----------------------------
if not os.path.exists(MAP_PATH):
    raise FileNotFoundError("❌ sensor_cluster_map.json not found!")

with open(MAP_PATH, "r") as f:
    SENSOR_TO_SUBSYSTEM = json.load(f)

# -----------------------------
# RCA DECODER
# -----------------------------
def decode_root_cause(sensor_list):
    """
    Converts sensor list → dominant physical subsystems
    """
    subsystems = [
        SENSOR_TO_SUBSYSTEM.get(s, "UNKNOWN")
        for s in sensor_list
    ]

    dominant = Counter(subsystems).most_common(3)
    return " + ".join([x[0] for x in dominant])

# -----------------------------
# RCA + ANOMALY ANALYSIS
# -----------------------------
results = []

for i in range(len(anomalies)):
    if anomalies[i]:

        timestamp = df.index[i]

        # reconstruction error vector for time i
        error_vector = np.abs(X[i] - X_reconstructed[i])

        # top 5 contributing sensors
        top_idx = np.argsort(error_vector)[-5:]
        root_sensors = [feature_names[j] for j in top_idx]

        # decode physical RCA
        physical_root_cause = decode_root_cause(root_sensors)

        print(f"\n🚨 ANOMALY DETECTED at {timestamp}")
        print("Top sensors:", root_sensors)
        print("✅ Physical RCA:", physical_root_cause)

        results.append({
            "timestamp": timestamp,
            "anomaly": True,
            "reconstruction_error": reconstruction_error[i],
            "root_cause_sensors": ",".join(root_sensors),
            "root_cause_physical": physical_root_cause
        })

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_PATH, index=False)

print("\n✅ RCA results saved to:", OUTPUT_PATH)
print("✅ Inference + Root Cause Analysis completed successfully.")
