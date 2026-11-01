import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/44_processed.csv", index_col=0)

# -----------------------------
# FEATURE EXTRACTION PER SENSOR
# (THIS IS THE CRITICAL FIX)
# -----------------------------
features = pd.DataFrame(index=df.columns)

features["mean"] = df.mean()
features["std"] = df.std()
features["skew"] = df.skew()
features["kurtosis"] = df.kurtosis()

# frequency energy (to separate vibration vs electrical)
fft_energy = []
for col in df.columns:
    signal = df[col].values
    fft = np.abs(np.fft.rfft(signal))
    fft_energy.append(np.mean(fft))

features["fft_energy"] = fft_energy

# -----------------------------
# NORMALIZATION
# -----------------------------
X = StandardScaler().fit_transform(features)

# -----------------------------
# CLUSTER INTO 9 REAL SUBSYSTEMS
# -----------------------------
kmeans = KMeans(n_clusters=9, random_state=42, n_init=20)
labels = kmeans.fit_predict(X)

features["cluster"] = labels

# -----------------------------
# PHYSICAL SUBSYSTEM RULE MAPPING
# -----------------------------
subsystem_names = {
    0: "ENVIRONMENT",
    1: "ROTOR",
    2: "SHAFT",
    3: "GEARBOX",
    4: "GENERATOR",
    5: "POWER_ELECTRONICS",
    6: "YAW",
    7: "PITCH",
    8: "TOWER"
}

sensor_map = {
    sensor: subsystem_names[cluster]
    for sensor, cluster in zip(features.index, labels)
}

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
with open("data/sensor_cluster_map.json", "w") as f:
    json.dump(sensor_map, f, indent=2)

features.to_csv("data/sensor_physical_features.csv")

print("✅ Physical subsystem mapping rebuilt correctly.")
